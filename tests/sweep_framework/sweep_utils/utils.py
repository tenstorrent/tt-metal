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
from typing import Optional, List
import copy
from functools import partial

from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from models.utility_functions import torch_random


unary_ops_dict = {
    # "unaryop" : {golden_func, unary_op_type, unary_op_param}
    "exp": (ttnn.get_golden_function(ttnn.exp), ttnn.UnaryOpType.EXP, None),
    "reciprocal": (ttnn.get_golden_function(ttnn.reciprocal), ttnn.UnaryOpType.RECIP, None),
    "gelu": (ttnn.get_golden_function(ttnn.gelu), ttnn.UnaryOpType.GELU, None),
    "relu": (ttnn.get_golden_function(ttnn.relu), ttnn.UnaryOpType.RELU, None),
    "sqrt": (ttnn.get_golden_function(ttnn.sqrt), ttnn.UnaryOpType.SQRT, None),
    "sigmoid": (ttnn.get_golden_function(ttnn.sigmoid), ttnn.UnaryOpType.SIGMOID, None),
    "log": (ttnn.get_golden_function(ttnn.log), ttnn.UnaryOpType.LOG, None),
    "tanh": (ttnn.get_golden_function(ttnn.tanh), ttnn.UnaryOpType.TANH, None),
    "log2": (ttnn.get_golden_function(ttnn.log2), ttnn.UnaryOpType.LOG2, None),
    "log10": (ttnn.get_golden_function(ttnn.log10), ttnn.UnaryOpType.LOG10, None),
    "sin": (ttnn.get_golden_function(ttnn.sin), ttnn.UnaryOpType.SIN, None),
    "cos": (ttnn.get_golden_function(ttnn.cos), ttnn.UnaryOpType.COS, None),
    "abs": (ttnn.get_golden_function(ttnn.abs), ttnn.UnaryOpType.ABS, None),
    "sign": (ttnn.get_golden_function(ttnn.sign), ttnn.UnaryOpType.SIGN, None),
    "square": (ttnn.get_golden_function(ttnn.square), ttnn.UnaryOpType.SQUARE, None),
    "eqz": (ttnn.get_golden_function(ttnn.eqz), ttnn.UnaryOpType.EQZ, None),
    "nez": (ttnn.get_golden_function(ttnn.nez), ttnn.UnaryOpType.NEZ, None),
    "gtz": (ttnn.get_golden_function(ttnn.gtz), ttnn.UnaryOpType.GTZ, None),
    "ltz": (ttnn.get_golden_function(ttnn.ltz), ttnn.UnaryOpType.LTZ, None),
    "gez": (ttnn.get_golden_function(ttnn.gez), ttnn.UnaryOpType.GEZ, None),
    "lez": (ttnn.get_golden_function(ttnn.lez), ttnn.UnaryOpType.LEZ, None),
    "relu_max": (
        ttnn.get_golden_function(ttnn.relu_max),
        ttnn.UnaryOpType.RELU_MAX,
        partial(torch_random, shape=(1,), low=0, high=100, dtype=torch.bfloat16),
    ),
    "relu_min": (
        ttnn.get_golden_function(ttnn.relu_min),
        ttnn.UnaryOpType.RELU_MIN,
        partial(torch_random, shape=(1,), low=0, high=100, dtype=torch.bfloat16),
    ),
    "power": (
        ttnn.get_golden_function(ttnn.pow),
        ttnn.UnaryOpType.POWER,
        partial(torch_random, shape=(1,), low=1, high=16, dtype=torch.int32),
    ),
    "leaky_relu": (
        ttnn.get_golden_function(ttnn.leaky_relu),
        ttnn.UnaryOpType.LEAKY_RELU,
        partial(torch_random, shape=(1,), low=0, high=10.0, dtype=torch.bfloat16),
    ),
    "elu": (
        ttnn.get_golden_function(ttnn.elu),
        ttnn.UnaryOpType.ELU,
        partial(torch_random, shape=(1,), low=-10, high=10, dtype=torch.bfloat16),
    ),
    "exp2": (ttnn.get_golden_function(ttnn.exp2), ttnn.UnaryOpType.EXP2, None),
    "heaviside": (
        ttnn.get_golden_function(ttnn.heaviside),
        ttnn.UnaryOpType.HEAVISIDE,
        partial(torch_random, shape=(1,), low=-100, high=100, dtype=torch.bfloat16),
    ),
    "expm1": (ttnn.get_golden_function(ttnn.expm1), ttnn.UnaryOpType.EXPM1, None),
    "signbit": (ttnn.get_golden_function(ttnn.signbit), ttnn.UnaryOpType.SIGNBIT, None),
    "asin": (torch.asin, ttnn.UnaryOpType.ASIN, None),
    "acos": (torch.acos, ttnn.UnaryOpType.ACOS, None),
    "rsqrt": (ttnn.get_golden_function(ttnn.rsqrt), ttnn.UnaryOpType.RSQRT, None),
    "relu6": (ttnn.get_golden_function(ttnn.relu6), ttnn.UnaryOpType.RELU6, None),
    "atan": (ttnn.get_golden_function(ttnn.atan), ttnn.UnaryOpType.ATAN, None),
    "erf": (ttnn.get_golden_function(ttnn.erf), ttnn.UnaryOpType.ERF, None),
    "erfc": (ttnn.get_golden_function(ttnn.erfc), ttnn.UnaryOpType.ERFC, None),
    "isinf": (ttnn.get_golden_function(ttnn.isinf), ttnn.UnaryOpType.ISINF, None),
    "isposinf": (ttnn.get_golden_function(ttnn.isposinf), ttnn.UnaryOpType.ISPOSINF, None),
    "isneginf": (ttnn.get_golden_function(ttnn.isneginf), ttnn.UnaryOpType.ISNEGINF, None),
    "isnan": (ttnn.get_golden_function(ttnn.isnan), ttnn.UnaryOpType.ISNAN, None),
    "logical_not_unary": (ttnn.get_golden_function(ttnn.logical_not), ttnn.UnaryOpType.LOGICAL_NOT_UNARY, None),
    "isfinite": (ttnn.get_golden_function(ttnn.isfinite), ttnn.UnaryOpType.ISFINITE, None),
    "erfinv": (ttnn.get_golden_function(ttnn.erfinv), ttnn.UnaryOpType.ERFINV, None),
    "i0": (ttnn.get_golden_function(ttnn.i0), ttnn.UnaryOpType.I0, None),
    "tan": (ttnn.get_golden_function(ttnn.tan), ttnn.UnaryOpType.TAN, None),
    "rsub": (
        ttnn.get_golden_function(ttnn.rsub),
        ttnn.UnaryOpType.RSUB,
        partial(torch_random, shape=(1,), low=-100, high=100, dtype=torch.bfloat16),
    ),
    "rdiv": (
        ttnn.get_golden_function(ttnn.rdiv),
        ttnn.UnaryOpType.RDIV,
        partial(torch_random, shape=(1,), low=0.1, high=100, dtype=torch.bfloat16),
    ),
    "silu": (ttnn.get_golden_function(ttnn.silu), ttnn.UnaryOpType.SILU, None),
    "identity": (ttnn.get_golden_function(ttnn.identity), ttnn.UnaryOpType.IDENTITY, None),
    "neg": (ttnn.get_golden_function(ttnn.neg), ttnn.UnaryOpType.NEG, None),
    "add_unary_sfpu": (
        ttnn.get_golden_function(ttnn.add),
        ttnn.UnaryOpType.ADD_UNARY_SFPU,
        partial(torch_random, shape=(1,), low=-100, high=100, dtype=torch.bfloat16),
    ),
    "sub_unary_sfpu": (
        ttnn.get_golden_function(ttnn.sub),
        ttnn.UnaryOpType.SUB_UNARY_SFPU,
        partial(torch_random, shape=(1,), low=-100, high=100, dtype=torch.bfloat16),
    ),
    "mul_unary_sfpu": (
        ttnn.get_golden_function(ttnn.multiply),
        ttnn.UnaryOpType.MUL_UNARY_SFPU,
        partial(torch_random, shape=(1,), low=-100, high=100, dtype=torch.bfloat16),
    ),
    "div_unary_sfpu": (
        ttnn.get_golden_function(ttnn.div),
        ttnn.UnaryOpType.DIV_UNARY_SFPU,
        partial(torch_random, shape=(1,), low=0.1, high=100, dtype=torch.bfloat16),
    ),
    "unary_ne": (
        ttnn.get_golden_function(ttnn.ne),
        ttnn.UnaryOpType.UNARY_NE,
        partial(torch_random, shape=(1,), low=-100, high=100, dtype=torch.bfloat16),
    ),
    "unary_gt": (
        ttnn.get_golden_function(ttnn.gt),
        ttnn.UnaryOpType.UNARY_GT,
        partial(torch_random, shape=(1,), low=-100, high=100, dtype=torch.bfloat16),
    ),
    "unary_lt": (
        ttnn.get_golden_function(ttnn.lt),
        ttnn.UnaryOpType.UNARY_LT,
        partial(torch_random, shape=(1,), low=-100, high=100, dtype=torch.bfloat16),
    ),
    # "tiled_prod": (ttnn.get_golden_function(ttnn.prod), ttnn.UnaryOpType.TILED_PROD, None),
    "floor": (ttnn.get_golden_function(ttnn.floor), ttnn.UnaryOpType.FLOOR, None),
    "floor_float32": (ttnn.get_golden_function(ttnn.floor), ttnn.UnaryOpType.FLOOR_FLOAT32, None),
    "ceil": (ttnn.get_golden_function(ttnn.ceil), ttnn.UnaryOpType.CEIL, None),
    "ceil_float32": (ttnn.get_golden_function(ttnn.ceil), ttnn.UnaryOpType.CEIL_FLOAT32, None),
    "remainder": (
        ttnn.get_golden_function(ttnn.remainder),
        ttnn.UnaryOpType.REMAINDER,
        partial(torch_random, shape=(1,), low=-100, high=100, dtype=torch.bfloat16),
    ),
    "fmod": (
        ttnn.get_golden_function(ttnn.fmod),
        ttnn.UnaryOpType.FMOD,
        partial(torch_random, shape=(1,), low=-100, high=100, dtype=torch.bfloat16),
    ),
    # "dropout": (ttnn.get_golden_function(ttnn.ceil), ttnn.UnaryOpType.DROPOUT, partial(torch_random, shape=(1,), low=0.1, high=1.0, dtype=torch.bfloat16))
    "fill": (
        ttnn.get_golden_function(ttnn.fill),
        ttnn.UnaryOpType.FILL,
        partial(torch_random, shape=(1,), low=-100, high=100, dtype=torch.bfloat16),
    ),
    "prelu": (
        ttnn.get_golden_function(ttnn.prelu),
        ttnn.UnaryOpType.PRELU_SFPU,
        partial(torch_random, shape=(1,), low=0, high=10, dtype=torch.bfloat16),
    ),
}


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


# Helper function to get the golden function and ttnn.UnaryWithParam(unary_op_type, param) from unary_op_dict
def get_unary_op_args(op, input_tensor, param):
    golden_function = unary_ops_dict[op][0]
    unary_op_type = unary_ops_dict[op][1]

    if param is not None:
        golden_function = partial(golden_function, input_tensor, param)
        unary_op_type = ttnn.UnaryWithParam(unary_op_type, param)
    else:
        golden_function = partial(golden_function, input_tensor)
        unary_op_type = ttnn.UnaryWithParam(unary_op_type)

    return (golden_function, unary_op_type)


def gen_unary_chain_spec(unary_ops_lists="default", sequence_length=4, num_samples=8):
    if unary_ops_lists == "default":
        unary_ops_lists = []
        assert (
            sequence_length is not None and sequence_length > 3
        ), "Sequence length must be defined and above 3 when using default unary_chain spec for sweep test"

        starting_ops = ["acos", "asin", "atan"]
        ending_ops = ["isinf", "isfinite", "isposinf", "isneginf", "unary_ne", "unary_gt", "unary_lt", "isnan"]
        all_ops = unary_ops_dict.keys()

        for i in range(num_samples):
            unary_ops_list = []
            unary_ops_list.append(random.choice([op for op in all_ops if op not in ending_ops]))
            unary_ops_list.extend(
                random.sample([op for op in all_ops if op not in starting_ops + ending_ops], sequence_length - 2)
            )
            unary_ops_list.append(random.choice([op for op in all_ops if op not in starting_ops]))
            unary_ops_lists.append(unary_ops_list)

    # unary_op_lists contains lists where each contains one unary_op. All unary_ops are included.
    if unary_ops_lists == "all_single_op":
        unary_ops_lists = []
        for op in unary_ops_dict.keys():
            unary_ops_lists.append([op])

    for unary_ops in unary_ops_lists:
        param_list = []

        for unary_op in unary_ops:
            param_gen_func = unary_ops_dict[unary_op][-1]
            if param_gen_func is not None:
                param = param_gen_func().item()
            else:
                param = None
            param_list.append(param)

        if unary_ops[0] in ["atan", "acos", "asin"]:
            input_range = (-1, 1)
        else:
            input_range = (-100, 100)

        yield {"unary_ops": unary_ops, "param_list": param_list, "input_range": input_range}
