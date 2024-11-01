# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import random
from loguru import logger
from itertools import product
import torch
import ttnn
import math
import itertools
from typing import Optional, List


def sanitize_shape_rm(input_shape):
    if input_shape[-1] % 2 != 0:
        input_shape[-1] = input_shape[-1] + input_shape[-1] % 2
    return input_shape


def tensor_to_dtype(x, dtype):
    if x.dtype == torch.bool:
        x = x.to(torch.bfloat16)

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
        TIMEOUT = 5
        num_tries = 0

        while samples_id < num_samples and num_tries < num_samples * 10:
            shape = []
            num_tries += 1

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


def gen_rand_bitwise_left_shift(size, shift_bits=None, low=-2147483647, high=2147483648):
    res = torch.randint(low, high, size, dtype=torch.int32)
    if shift_bits is None:
        return res

    changebit = 31 - shift_bits
    exludebit_mask = torch.bitwise_not(torch.tensor(2**changebit).to(torch.int32))
    includebit_mask = torch.tensor(2**changebit).to(torch.int32)

    res = torch.where(res < 0, torch.bitwise_or(res, includebit_mask), torch.bitwise_and(res, exludebit_mask))

    return res


def gen_with_zeroes(size, probabilityzeroes=0.5, low=-100, high=100, dtype=torch.bfloat16):
    element_count = 1
    if probabilityzeroes == "random":
        probabilityzeroes = random.uniform(0.0, 0.9)
    for i in size:
        element_count = element_count * i
    raw = torch.zeros(element_count).to(dtype)
    raw[: int(probabilityzeroes * element_count)] = torch.Tensor(
        size=(1, int(probabilityzeroes * element_count))
    ).uniform_(low, high)
    ridx = torch.randperm(element_count)  # a random permutation of the entries
    mask = torch.reshape(raw[ridx], size)
    return mask


# at the moment, topk only works on last dim
# last dim must be a multiple of 64 and a pow of 2
def santize_topk_shape(input_shape):
    num_dims = len(input_shape)
    last_dim = input_shape[num_dims - 1]
    if not (last_dim & (last_dim - 1) == 0) and last_dim != 0:
        last_dim = 2 ** math.ceil(math.log2(last_dim))
        last_dim = last_dim + last_dim % 64

    input_shape[num_dims - 1] = last_dim

    return input_shape


def gen_rand_integers(low, high, num_samples):
    for i in range(num_samples):
        yield random.randint(low, high)


def gen_split_qkv_heads_spec(
    batch_size_list: List[int],
    sequence_size_list: List[int],
    num_heads_list: List[int],
    transpose_key_list: List[bool],
    num_kv_heads_list: List[int] = [None],
    kv_input_tensor_list: List[bool] = [False],
    use_invalid_hidden_size=False,
):
    for batch_size, sequence_size, num_heads, num_kv_heads, kv_input_tensor, transpose_key in itertools.product(
        batch_size_list, sequence_size_list, num_heads_list, num_kv_heads_list, kv_input_tensor_list, transpose_key_list
    ):
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if use_invalid_hidden_size is False:
            head_size = 32 * random.randint(1, 3)
            hidden_size = head_size * (num_heads + num_kv_heads * 2)
        else:
            hidden_size = random.randint(num_heads + num_kv_heads * 2, 2048)

        yield {
            "batch_size": batch_size,
            "sequence_size": sequence_size,
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "kv_input_tensor": kv_input_tensor,
            "transpose_key": transpose_key,
        }
