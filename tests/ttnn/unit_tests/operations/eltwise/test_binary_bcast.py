# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    compare_pcc,
)
from models.utility_functions import skip_for_grayskull
from itertools import product as parameters


binary_fns = (
    "gte",
    "gt",
    "lte",
    "lt",
    "eq",
    "ne",
    "logical_and",
    "logical_or",
    "logical_xor",
    "ldexp",
    "logaddexp",
    "logaddexp2",
    "squared_difference",
    "add",
    "sub",
    "mul",
    "div",
    "bias_gelu",
)
activation_fns = {
    "EXP": torch.exp,
    "GELU": torch.nn.functional.gelu,
    "RELU": torch.relu,
    "SQRT": torch.sqrt,
    "SIGMOID": torch.sigmoid,
    "LOG": torch.log,
    "TANH": torch.tanh,
    "LOG2": torch.log2,
    "LOG10": torch.log10,
    "SIN": torch.sin,
    "COS": torch.cos,
    "ABS": torch.abs,
    "SIGN": torch.sign,
    "SQUARE": torch.square,
    "EQZ": lambda x: torch.eq(x, 0),
    "NEZ": lambda x: torch.not_equal(x, 0),
    "GTZ": lambda x: torch.greater(x, 0),
    "LTZ": lambda x: torch.less(x, 0),
    "GEZ": lambda x: torch.greater_equal(x, 0),
    "LEZ": lambda x: torch.less_equal(x, 0),
    "EXP2": torch.exp2,
    "EXPM1": torch.expm1,
    "SIGNBIT": torch.signbit,
    "RSQRT": torch.rsqrt,
    "RELU6": torch.nn.functional.relu6,
    "ATAN": torch.atan,
    "ERF": torch.erf,
    "ERFC": torch.erfc,
    "ISINF": torch.isinf,
    "ISPOSINF": torch.isposinf,
    "ISNEGINF": torch.isneginf,
    "ISNAN": torch.isnan,
    "LOGICAL_NOT_UNARY": torch.logical_not,
    "ISFINITE": torch.isfinite,
    "ERFINV": torch.erfinv,
    "I0": torch.i0,
    "I1": torch.special.i1,
    "TAN": torch.tan,
    "SILU": torch.nn.functional.silu,
    "NEG": torch.neg,
    "BITWISE_NOT": torch.bitwise_not,
    "FLOOR": torch.floor,
    "CEIL": torch.ceil,
}
square_lhs = (["SQUARE"], [], [])
sin_rhs = ([], ["SIN"], [])
floor_lhs_ceil_rhs_cos_post = (["FLOOR"], ["CEIL"], ["COS"])
exp_floor_lhs_exp_rhs = (["FLOOR", "EXP"], ["EXP"], [])
log_lhs_sqrt_abs_post = (["LOG"], [], ["ABS", "SQRT"])
exp_post = ([], [], ["EXP"])
sqrt_post = ([], [], ["SQRT"])
sigmoid_post = ([], [], ["SIGMOID"])
log_post = ([], [], ["LOG"])
tanh_post = ([], [], ["TANH"])
log2_post = ([], [], ["LOG2"])
log10_post = ([], [], ["LOG10"])
sin_post = ([], [], ["SIN"])
exp2_post = ([], [], ["EXP2"])
expm1_post = ([], [], ["EXPM1"])
rsqrt_post = ([], [], ["RSQRT"])
atan_post = ([], [], ["ATAN"])
erf_post = ([], [], ["ERF"])
erfc_post = ([], [], ["ERFC"])
erfinv_post = ([], [], ["ERFINV"])
i0_post = ([], [], ["I0"])
tan_post = ([], [], ["TAN"])
silu_post = ([], [], ["SILU"])
floor_post = ([], [], ["FLOOR"])
ceil_post = ([], [], ["CEIL"])


@skip_for_grayskull("Possible accuracy issues with grayskull")
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 1, 1]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 1, 64, 1]), torch.Size([1, 3, 1, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([1, 3, 128, 1])),
    ),
)
@pytest.mark.parametrize(
    "ttnn_fn",
    binary_fns,
)
@pytest.mark.parametrize(
    "activations",
    (
        ([], [], []),
        square_lhs,
        sin_rhs,
        floor_lhs_ceil_rhs_cos_post,
        exp_floor_lhs_exp_rhs,
        log_lhs_sqrt_abs_post,
        *(([], [], [op]) for op in activation_fns.keys()),
    ),
)
def test_binary_scalar_ops(input_shapes, ttnn_fn, activations, device):
    a_shape, b_shape = input_shapes
    ttnn_op = getattr(ttnn.experimental, ttnn_fn)
    lhs, rhs, post = ([ttnn.UnaryOpType[activation] for activation in chain] for chain in activations)

    def compare(tt, pt):
        if (ttnn_fn, activations) in (
            *parameters(("eq", "ne"), (square_lhs, sin_rhs, exp_floor_lhs_exp_rhs, log_lhs_sqrt_abs_post)),
            *parameters(
                ("logaddexp", "logaddexp2"),
                (floor_lhs_ceil_rhs_cos_post, tanh_post, erfinv_post, tan_post, floor_post, ceil_post),
            ),
            *parameters(
                ("bias_gelu"),
                (
                    floor_lhs_ceil_rhs_cos_post,
                    log_lhs_sqrt_abs_post,
                    tanh_post,
                    erfinv_post,
                    tan_post,
                    floor_post,
                    ceil_post,
                ),
            ),
            *parameters(
                ("gte", "lt", "lte"),
                (
                    exp_floor_lhs_exp_rhs,
                    log_lhs_sqrt_abs_post,
                    tanh_post,
                    erfinv_post,
                    tan_post,
                    floor_post,
                    ceil_post,
                ),
            ),
            *parameters(
                ("logical_and", "logical_or"),
                (
                    log_lhs_sqrt_abs_post,
                    exp_post,
                    sqrt_post,
                    sigmoid_post,
                    log_post,
                    tanh_post,
                    log2_post,
                    log10_post,
                    expm1_post,
                    atan_post,
                    erf_post,
                    erfc_post,
                    silu_post,
                ),
            ),
            *parameters(("logical_xor"), (log_lhs_sqrt_abs_post, erf_post, erfc_post)),
            *parameters(("div"), (exp_post, exp2_post, expm1_post, i0_post, tan_post)),
            *parameters(("sub"), (log_post, log2_post, log10_post)),
            *parameters(("add"), (tanh_post, tan_post)),
            *parameters(("ldexp"), (erfinv_post, tan_post, floor_post, ceil_post)),
            ("logaddexp2", sin_post),
            ("squared_difference", i0_post),
        ):
            pytest.skip("precision error")

        if (ttnn_fn, activations) in (
            *parameters(("bias_gelu"), (square_lhs, rsqrt_post)),
            *parameters(("gte", "lte", "lt"), (sin_rhs)),
            *parameters(("logaddexp2"), (exp2_post, expm1_post, atan_post, erf_post)),
            ("squared_difference", erfinv_post),
            ("div", erfinv_post),
        ):
            return compare_pcc(tt, pt, 0.977)

        return compare_pcc(tt, pt)

    a_pt = torch.rand(a_shape).bfloat16()
    b_pt = torch.rand(b_shape).bfloat16()

    a_tt = ttnn.from_torch(a_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b_tt = ttnn.from_torch(b_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cq_id = 0
    out_tt = ttnn_op(a_tt, b_tt, queue_id=cq_id, lhs_activations=lhs, rhs_activations=rhs, post_activations=post)

    for op in lhs:
        a_pt = activation_fns[op](a_pt).bfloat16()

    for op in rhs:
        b_pt = activation_fns[op](b_pt).bfloat16()

    golden_fn = ttnn.get_golden_function(ttnn_op)
    out_pt = golden_fn(a_pt, b_pt).bfloat16()

    for op in post:
        out_pt = activation_fns[op](out_pt).bfloat16()

    comp_pass = compare([out_tt], [out_pt])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 31, 32]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 2, 64, 1]), torch.Size([1, 3, 1, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([2, 3, 128, 1])),
    ),
)
@pytest.mark.parametrize(
    "ttnn_fn",
    binary_fns,
)
def test_binary_scalar_ops_invalid_bcast(input_shapes, ttnn_fn, device):
    a_shape, b_shape = input_shapes
    a_pt = torch.rand(a_shape).bfloat16()
    b_pt = torch.rand(b_shape).bfloat16()

    a_tt = ttnn.from_torch(a_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b_tt = ttnn.from_torch(b_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    with pytest.raises(RuntimeError) as e:
        cq_id = 0
        _ = ttnn_fn(a_tt, b_tt, queue_id=cq_id)
        assert "Broadcasting rule violation" in str(e.value)


@pytest.mark.parametrize(
    "shapes",
    [
        [[1, 71, 7, 7], [7, 7]],
        [[920, 1, 256], [256]],
        [[4, 12, 64, 64], [12, 1, 1]],
        [[4, 16, 64, 64], [16, 1, 1]],
        [[64, 3, 64, 64], [3, 1, 1]],
        [[64, 4, 64, 64], [4, 1, 1]],
        [[16, 6, 64, 64], [6, 1, 1]],
        [[16, 8, 64, 64], [8, 1, 1]],
        [[16, 1], [1, 1, 32]],
    ],
)
def test_unequal_ranks(device, shapes):
    torch.manual_seed(0)
    torch_input_tensor_a = torch.rand(shapes[0], dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shapes[1], dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    output_tensor = ttnn.experimental.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988


@pytest.mark.parametrize(
    "data",
    [
        ([], [], []),
        ([1], [2], [3]),
        ([1], [], []),
        ([], [1], []),
        ([1, 2], [3], [4, 5]),
        ([1], [2, 3], [3, 4]),
        ([1, 2], [3, 4], [4, 6]),
    ],
)
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_01_volume_tensors(device, data, memory_config):
    (a, b, c_golden) = data
    a = torch.BFloat16Tensor(a)
    b = torch.BFloat16Tensor(b)
    assert torch.add(a, b).tolist() == c_golden

    ttnn_a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    ttnn_b = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    ttnn_c = ttnn.experimental.add(ttnn_a, ttnn_b)
    c = ttnn.to_torch(ttnn_c).reshape((-1))

    assert c.tolist() == c_golden
