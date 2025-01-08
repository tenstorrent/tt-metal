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


binary_fns = {
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
    "add_",
    "sub_",
    "mul_",
    "gt_",
    "lt_",
    "lte_",
    "gte_",
    "eq_",
    "ne_",
}
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
    "TAN": torch.tan,
    "SILU": torch.nn.functional.silu,
    "NEG": torch.neg,
    "FLOOR": torch.floor,
    "CEIL": torch.ceil,
}
no_activations = ((), (), ())
square_lhs = (("SQUARE",), (), ())
sin_rhs = ((), ("SIN",), ())
floor_lhs_ceil_rhs_cos_post = (("FLOOR",), ("CEIL",), ("COS",))
exp_floor_lhs_exp_rhs = (("FLOOR", "EXP"), ("EXP",), ())
log_lhs_sqrt_abs_post = (("LOG",), (), ("ABS", "SQRT"))
exp_post = ((), (), ("EXP",))
log_post = ((), (), ("LOG",))
tanh_post = ((), (), ("TANH",))
log2_post = ((), (), ("LOG2",))
log10_post = ((), (), ("LOG10",))
exp2_post = ((), (), ("EXP2",))
expm1_post = ((), (), ("EXPM1",))
erfinv_post = ((), (), ("ERFINV",))
i0_post = ((), (), ("I0",))
tan_post = ((), (), ("TAN",))
floor_post = ((), (), ("FLOOR",))
ceil_post = ((), (), ("CEIL",))


def rand_bf16_gen(shape, device, *, min=0, max=1):
    pt = torch.rand(shape, dtype=torch.bfloat16) * (max - min) + min
    tt = ttnn.from_torch(pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return pt, tt


@skip_for_grayskull("Possible accuracy issues with grayskull")
@pytest.mark.parametrize(
    "a_shape, b_shape",
    (
        (torch.Size([1, 1, 1, 1]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 1, 64, 1]), torch.Size([1, 3, 1, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([1, 3, 128, 1])),
    ),
)
@pytest.mark.parametrize(
    "ttnn_fn, activations",
    {
        *parameters(
            binary_fns,
            {
                no_activations,
                square_lhs,
                sin_rhs,
                floor_lhs_ceil_rhs_cos_post,
                exp_floor_lhs_exp_rhs,
                log_lhs_sqrt_abs_post,
            },
        ),
        *parameters({"add"}, {((), (), (op,)) for op in activation_fns.keys()}),
    }.difference(
        parameters({"eq", "ne", "ne_"}, {square_lhs, sin_rhs, exp_floor_lhs_exp_rhs, log_lhs_sqrt_abs_post}),
        parameters({"eq_"}, {square_lhs, sin_rhs, exp_floor_lhs_exp_rhs}),
        parameters({"logaddexp", "logaddexp2"}, {floor_lhs_ceil_rhs_cos_post}),
        parameters({"gte", "lt", "lte", "lt_"}, {exp_floor_lhs_exp_rhs, log_lhs_sqrt_abs_post}),
        parameters({"lte_"}, {sin_rhs, log_lhs_sqrt_abs_post}),
        parameters({"gte_"}, {exp_floor_lhs_exp_rhs}),
        parameters({"gt_"}, {sin_rhs}),
        parameters({"logical_and", "logical_or", "logical_xor", "bias_gelu"}, {log_lhs_sqrt_abs_post}),
        parameters({"div"}, {exp_post, tanh_post, exp2_post, expm1_post, i0_post, tan_post}),
        parameters({"sub"}, {log_post, log2_post, log10_post}),
        parameters({"ldexp"}, {erfinv_post, tan_post, floor_post, ceil_post}),
        parameters({"squared_difference"}, {erfinv_post, i0_post}),
        parameters({"add"}, {tan_post, tanh_post}),
        {("mul", log_lhs_sqrt_abs_post)},
        {("mul_", log_lhs_sqrt_abs_post)},
    ),
)
def test_binary_scalar_ops(a_shape, b_shape, ttnn_fn, activations, device):
    torch.manual_seed(0)
    ttnn_op = getattr(ttnn.experimental, ttnn_fn)
    lhs, rhs, post = ([getattr(ttnn.UnaryOpType, op) for op in ops] for ops in activations)
    golden_lhs, golden_rhs, golden_post = ((activation_fns[op] for op in ops) for ops in activations)
    # make 0 exclusive for rhs of div
    min, max = (1, 0) if ttnn_fn == "div" else (0, 1)

    a_pt, a_tt = rand_bf16_gen(a_shape, device)
    b_pt, b_tt = rand_bf16_gen(b_shape, device, min=min, max=max)

    cq_id = 0
    out_tt = ttnn_op(a_tt, b_tt, queue_id=cq_id, lhs_activations=lhs, rhs_activations=rhs, post_activations=post)

    for golden_activation in golden_lhs:
        a_pt = golden_activation(a_pt).bfloat16()

    for golden_activation in golden_rhs:
        b_pt = golden_activation(b_pt).bfloat16()

    golden_fn = ttnn.get_golden_function(ttnn_op)
    out_pt = golden_fn(a_pt, b_pt).bfloat16()

    for golden_activation in golden_post:
        out_pt = golden_activation(out_pt).bfloat16()

    def compare(tt, pt):
        imprecise_cases = {
            *parameters({"bias_gelu"}, {square_lhs, floor_lhs_ceil_rhs_cos_post}),
            *parameters({"gte", "gt", "lte", "lt"}, {sin_rhs}),
        }
        return compare_pcc(tt, pt, 0.98) if (ttnn_fn, activations) in imprecise_cases else compare_pcc(tt, pt)

    assert compare([out_tt], [out_pt])


@pytest.mark.parametrize(
    "a_shape, b_shape",
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
def test_binary_scalar_ops_invalid_bcast(a_shape, b_shape, ttnn_fn, device):
    torch.manual_seed(0)
    ttnn_op = getattr(ttnn.experimental, ttnn_fn)

    _, a_tt = rand_bf16_gen(a_shape, device)
    _, b_tt = rand_bf16_gen(b_shape, device)

    with pytest.raises(RuntimeError) as e:
        cq_id = 0
        _ = ttnn_op(a_tt, b_tt, queue_id=cq_id)
        assert "Broadcasting rule violation" in str(e.value)


@pytest.mark.parametrize(
    "a_shape, b_shape",
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
def test_unequal_ranks(a_shape, b_shape, device):
    torch.manual_seed(0)

    torch_input_tensor_a, input_tensor_a = rand_bf16_gen(a_shape, device)
    torch_input_tensor_b, input_tensor_b = rand_bf16_gen(b_shape, device)

    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    output_tensor = ttnn.experimental.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988


@pytest.mark.parametrize(
    "a, b, c_golden",
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
def test_01_volume_tensors(device, a, b, c_golden, memory_config):
    a = torch.BFloat16Tensor(a)
    b = torch.BFloat16Tensor(b)
    assert torch.add(a, b).tolist() == c_golden

    ttnn_a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    ttnn_b = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    ttnn_c = ttnn.experimental.add(ttnn_a, ttnn_b)
    c = ttnn.to_torch(ttnn_c).reshape((-1))

    assert c.tolist() == c_golden
