# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import random
from loguru import logger
from itertools import product
import torch
import ttnn


def tensor_to_dtype(x, dtype):
    if dtype == ttnn.bfloat16:
        x = x.to(torch.bfloat16)

    elif dtype == ttnn.bfloat8_b:
        tt_tensor = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=None, memory_config=None)

        x = ttnn.to_torch(tt_tensor)

    elif dtype == ttnn.bfloat4_b:
        tt_tensor = ttnn.from_torch(x, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=None, memory_config=None)

        x = ttnn.to_torch(tt_tensor)

    elif dtype == ttnn.uint16:
        x = x.to(torch.int16)

    elif dtype == ttnn.uint32:
        x = x.to(torch.int32)

    elif dtype == ttnn.int32:
        x = x.to(torch.int32)

    elif dtype == ttnn.float32:
        pass

    else:
        logger.warning(f"Unknown dtype {dtype} passed to gen_func_with_cast_tt")

    return x


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


def gen_low_high_scalars(low=-100, high=100, dtype=torch.bfloat16):
    low = torch.tensor(1, dtype=dtype).uniform_(low, high).item()
    high = torch.tensor(1, dtype=dtype).uniform_(low, high).item()

    if low >= high:
        low, high = high, low

    return low, high


def gen_rand_exclude_range(size, excluderange=None, low=0, high=100):
    res = torch.Tensor(size=size).uniform_(low, high)
    if excluderange is None:
        return res

    upper = excluderange[1]
    lower = excluderange[0]
    assert upper < high
    assert lower > low

    while torch.any((res > lower) & (res < upper)):
        res = torch.where((res < lower) & (res > upper), res, random.uniform(low, high))

    return res
