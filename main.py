import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import combinations


class TSPSolver:
    def __init__(self, distance_matrix=None, cities=None):
        """
        Inicializa el solucionador TSP con una matriz de distancias o coordenadas de ciudades.

        Args:
            distance_matrix: Matriz de distancias entre ciudades (opcional)
            cities: Diccionario con nombres de ciudades como claves y tuplas (x,y) como valores (opcional)
        """
        self.distance_matrix = distance_matrix
        self.cities = cities
        self.best_route = None
        self.best_distance = float('inf')

        # Si se proporcionan ciudades pero no matriz de distancias, crear la matriz
        if cities is not None and distance_matrix is None:
            self._create_distance_matrix()

    def _create_distance_matrix(self):
        """Crea una matriz de distancias a partir de coordenadas de ciudades"""
        n = len(self.cities)
        self.distance_matrix = np.zeros((n, n))
        city_names = list(self.cities.keys())

        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = self.cities[city_names[i]]
                    x2, y2 = self.cities[city_names[j]]
                    self.distance_matrix[i, j] = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def calculate_route_distance(self, route):
        """
        Calcula la distancia total de una ruta

        Args:
            route: Lista de índices de ciudades que representan la ruta

        Returns:
            Distancia total de la ruta
        """
        total_distance = 0
        n = len(route)

        for i in range(n):
            from_city = route[i]
            to_city = route[(i + 1) % n]
            total_distance += self.distance_matrix[from_city, to_city]

        return total_distance

    def nearest_neighbor_initial_route(self):
        """
        Genera una ruta inicial utilizando el algoritmo del vecino más cercano

        Returns:
            Lista de índices de ciudades (ruta)
        """
        n = len(self.distance_matrix)
        unvisited = set(range(n))
        current = 0  # Comenzamos desde la primera ciudad
        route = [current]
        unvisited.remove(current)

        while unvisited:
            nearest = min(unvisited, key=lambda city: self.distance_matrix[current, city])
            route.append(nearest)
            current = nearest
            unvisited.remove(nearest)

        return route

    def subpath_inversion(self, route, max_iterations=1000):
        """
        Aplica el algoritmo del subviaje inverso para mejorar una ruta

        Args:
            route: Ruta inicial
            max_iterations: Número máximo de iteraciones

        Returns:
            Ruta mejorada y distancia final
        """
        n = len(route)
        best_route = route.copy()
        best_distance = self.calculate_route_distance(best_route)

        iterations = 0
        improvement = True

        while improvement and iterations < max_iterations:
            improvement = False
            iterations += 1

            # Probar todas las combinaciones posibles de aristas no adyacentes
            for i, j in combinations(range(n), 2):
                if j == (i + 1) % n or i == (j + 1) % n:
                    # Aristas adyacentes, saltar
                    continue

                # Crear una nueva ruta invirtiendo el subviaje
                new_route = best_route.copy()
                # Aseguramos que i < j para la inversión
                if i > j:
                    i, j = j, i

                # Invertir el subviaje entre i y j
                new_route[i + 1:j + 1] = reversed(new_route[i + 1:j + 1])

                # Calcular la distancia de la nueva ruta
                new_distance = self.calculate_route_distance(new_route)

                # Si la nueva ruta es mejor, actualizar
                if new_distance < best_distance:
                    best_distance = new_distance
                    best_route = new_route.copy()
                    improvement = True
                    print("iteration", iterations, "route:", new_route, "distance:", best_distance)
                    #break  # Primera mejora encontrada

        return best_route, best_distance

    def solve(self, initial_route=None, max_iterations=1000):
        """
        Resuelve el problema TSP utilizando el algoritmo del subviaje inverso

        Args:
            initial_route: Ruta inicial (opcional)
            max_iterations: Número máximo de iteraciones

        Returns:
            Mejor ruta encontrada y su distancia
        """
        if initial_route is None:
            initial_route = self.nearest_neighbor_initial_route()

        start_time = time.time()
        best_route, best_distance = self.subpath_inversion(initial_route, max_iterations)
        end_time = time.time()

        self.best_route = best_route
        self.best_distance = best_distance

        print(f"Ruta óptima encontrada con distancia: {best_distance:.2f}")
        print(f"Tiempo de ejecución: {end_time - start_time:.4f} segundos")

        return best_route, best_distance

    def plot_route(self, route=None, title="Ruta del Agente Viajero"):
        """
        Visualiza la ruta del TSP

        Args:
            route: Ruta a visualizar (opcional, usa la mejor ruta si no se proporciona)
            title: Título del gráfico
        """
        if route is None:
            route = self.best_route

        if self.cities is None:
            print("No se pueden visualizar las ciudades sin coordenadas")
            return

        plt.figure(figsize=(10, 8))
        city_names = list(self.cities.keys())

        # Extraer coordenadas x e y
        x_coords = [self.cities[city_names[i]][0] for i in route]
        y_coords = [self.cities[city_names[i]][1] for i in route]

        # Agregar el retorno a la primera ciudad
        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])

        # Trazar la ruta
        plt.plot(x_coords, y_coords, 'b-', linewidth=0.7)
        plt.scatter(x_coords, y_coords, c='red', s=50)

        # Etiquetar las ciudades
        for i, txt in enumerate(route):
            plt.annotate(city_names[txt], (x_coords[i], y_coords[i]), fontsize=10)

        plt.title(f"{title}\nDistancia total: {self.best_distance:.2f}")
        plt.grid(True)
        plt.axis('equal')
        plt.show()


# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo con 5 ciudades
    cities = {
        '1': (0, 6),
        '2': (5, 13),
        '3': (7, 7),
        '4': (14, 11.5),
        '5': (8, 5),
        '6': (12.5, 3.5),
        '7': (7, 0),
    }

    # Ejemplo con matriz de distancias
    distance_matrix = np.array([
        [  0,  12,  10, 100, 100, 100,  12],
        [ 12,   0,   8,  12, 100, 100, 100],
        [ 10,   8,   0,  11,   3, 100,   9],
        [100,  12,  11,   0,  11,  10, 100],
        [100, 100,   3,  11,   0,   6,   7],
        [100, 100, 100,  10,   6,   0,   9],
        [ 12, 100,   9, 100,   7,   9,   0]
    ])

    # Resolver usando matriz de distancias
    print("\nResolviendo TSP con matriz de distancias:")
    solver1 = TSPSolver(distance_matrix=distance_matrix)

    # Ruta inicial: A → B → C → D → E → A (índices: 0 → 1 → 2 → 3 → 4 → 0)
    initial_route = [0, 1, 2, 3, 4, 5, 6, 0]
    #initial_route = {0, 1, 3, 2, 4, 5, 6, 0}

    best_route, best_distance = solver1.solve(initial_route=initial_route)

    # Mostrar solución
    city_names = ['1', '2', '3', '4', '5', '6', '7']
    print("Ruta inicial:", [city_names[i] for i in initial_route])
    print("Mejor ruta encontrada:", [city_names[i] for i in best_route])
    print("Distancia total:", best_distance)

    # Podemos también visualizar la solución del primer caso si definimos coordenadas
    solver1.cities = cities
    solver1.plot_route(title="Solución TSP con matriz de distancias")