# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import os
import random
from loguru import logger
from itertools import product
import torch
import ttnn
import math
import itertools
from typing import Optional, List, Callable
import copy
from functools import partial

from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from models.utility_functions import torch_random


def sanitize_shape_rm(input_shape):
    if input_shape[-1] % 2 != 0:
        input_shape[-1] = input_shape[-1] + 1
    return input_shape


def sanitize_shape(shape, method, **kwargs):
    if method == "topk":
        num_dims = len(shape)
        last_dim = shape[num_dims - 1]
        if not (last_dim & (last_dim - 1) == 0) and last_dim != 0:
            last_dim = 2 ** math.ceil(math.log2(last_dim))
            if last_dim < 64:
                last_dim = 64
        shape[num_dims - 1] = last_dim

    if method == "split_query_key_value_and_split_heads":
        assert len(shape) == 3

        hidden_size = shape[2]
        num_heads = kwargs.pop("num_heads")
        num_kv_heads = kwargs.pop("num_kv_heads")

        if num_kv_heads is None:
            min_sum_heads_size = 32 * (3 * num_heads)
        else:
            min_sum_heads_size = 32 * (num_heads + 2 * num_kv_heads)
            hidden_size = 4672

        if hidden_size % min_sum_heads_size != 0:
            if hidden_size < min_sum_heads_size:
                hidden_size = min_sum_heads_size
            else:
                hidden_size = (hidden_size // min_sum_heads_size) * min_sum_heads_size

        shape[2] = hidden_size

    return shape


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


def gen_rand_integers(low, high, num_samples):
    for i in range(num_samples):
        yield random.randint(low, high)


def gen_split_qkv_heads_spec(
    input_shape_list: List[int],
    transpose_key_list: List[bool],
    num_heads_list: List[int],
    num_kv_heads_list: List[int] = [None],
    kv_input_tensor_list: List[bool] = [False],
    use_invalid_hidden_size=False,
):
    for input_shape, num_heads, num_kv_heads, kv_input_tensor, transpose_key in itertools.product(
        input_shape_list, num_heads_list, num_kv_heads_list, kv_input_tensor_list, transpose_key_list
    ):
        input_shape_ = input_shape.copy()

        if use_invalid_hidden_size is False:
            input_shape_ = sanitize_shape(
                input_shape_,
                "split_query_key_value_and_split_heads",
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
            )

        yield {
            "batch_size": input_shape_[0],
            "sequence_size": input_shape_[1],
            "hidden_size": input_shape_[2],
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "kv_input_tensor": kv_input_tensor,
            "transpose_key": transpose_key,
        }


def gen_rotary_embedding_spec(
    input_shape_list,
    cache_size_list,
    use_token_idx_list=[True, False],
):
    for input_shape, cache_size, use_token_idx in itertools.product(
        input_shape_list, cache_size_list, use_token_idx_list
    ):
        input_shape_ = input_shape.copy()
        if use_token_idx is True:
            token_idx = random.randint(1, cache_size - 1)
            input_shape_[0] = 1
        else:
            token_idx = None

        yield {
            "input_shape": input_shape_,
            "cache_size": cache_size,
            "token_idx": token_idx,
        }


def gen_complex_tensor(input_shape, low, high, dtype=ttnn.bfloat16):
    torch_real = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), dtype)(
        input_shape
    ).to(torch.float32)
    torch_imag = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), dtype)(
        input_shape
    ).to(torch.float32)

    return torch.complex(torch_real, torch_imag)


def complex_from_torch(torch_tensor, dtype, layout, memory_config, device):
    tt_real = ttnn.from_torch(
        torch_tensor.real,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
    )
    tt_imag = ttnn.from_torch(
        torch_tensor.imag,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
    )
    return ttnn.complex_tensor(tt_real, tt_imag)


def get_device_grid_size():
    device = ttnn.open_device(device_id=0)
    if ttnn.device.is_wormhole_b0(device):
        y, x = 8, 8
    elif ttnn.device.is_grayskull(device):
        y, x = 9, 12
    else:
        y, x = 8, 8
    ttnn.close_device(device)
    del device
    return y, x


def gen_pytest_parametrize_args(parameters: dict, invalidate_function: Callable | None = None) -> dict:
    """Generate pytest parametrize arguments from a dictionary of parameters."
    Args:
        parameters: A dictionary of parameters. The keys are the config names and the values are dictionaries of parameter values.
                    The keys of the inner dictionaries are the function argument names.
                    The values of the inner dictionaries are lists of parameter values to be used for sweep.
        invalidate_function (optional): A function that takes a dictionary of parameter values and returns a boolean indicating
                    whether to skip the test. The function should return (should_skip, reason) where should_skip is a boolean and
                    reason is a string. If should_skip is True, the specific test case is skipped.
    Returns:
        A dictionary of pytest parametrize arguments.
    Example:
        parameters = {
            "cfg1": {
                "arg1": [1, 2],
                "arg2": [3, 4],
            },
            "cfg2": {
                "arg1": [5, 6],
                "arg2": [7],
            },
        }
        The function returns:
        {'argnames': ['arg1', 'arg2'], 'argvalues': { ... }, 'ids': { ... }} where
        argvalues = [(1, 3), (1, 4), (2, 3), (2, 4), (5, 7), (6, 7)]
        ids = ['cfg1-arg1_1-arg2_3', 'cfg1-arg1_1-arg2_4', ... , 'cfg2-arg1_6-arg2_7']
    """
    test_argnames, test_id2values = [], {}
    for param_name, values_dict in parameters.items():
        test_argnames = list(values_dict)
        all_arg_combs = itertools.product(*[values_dict[arg_name] for arg_name in test_argnames])

        for arg_vals in all_arg_combs:
            if invalidate_function is not None:
                test_vector = {name: val for name, val in zip(test_argnames, arg_vals)}
                should_skip, _reason = invalidate_function(test_vector)
                if should_skip:
                    continue
            id_str = (
                str(param_name)
                + "-"
                + "-".join(str(name) + "_" + str(val) for name, val in zip(test_argnames, arg_vals))
            )
            test_id2values[id_str] = arg_vals
    return {"argnames": test_argnames, "argvalues": test_id2values.values(), "ids": test_id2values.keys()}
