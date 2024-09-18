import random
from itertools import product


def gen_shapes(start_shape, end_shape, interval, num_samples="all"):
    def align_to_interval(x, start_val, interval):
        dx = x - start_val
        dx = (dx // interval) * interval
        return start_val + dx

    shapes = []

    num_dims = len(start_shape)

    if num_samples == "all":
        dim_ranges = [range(start_shape[i], end_shape[i] + interval[i], interval[i]) for i in range(num_dims)]

        for shape in product(*dim_ranges):
            shapes.append(shape)

    else:
        samples_id = 0

        while samples_id < num_samples:
            shape = []

            for i in range(-num_dims, 0):
                x = random.randint(start_shape[i], end_shape[i])
                shape.append(align_to_interval(x, start_shape[i], interval[i]))

            samples_id += 1
            shapes.append(shape)

    return shapes
