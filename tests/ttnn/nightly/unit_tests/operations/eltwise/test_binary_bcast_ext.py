# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    compare_pcc,
)
from models.common.utility_functions import torch_random
from models.common.utility_functions import divup
from itertools import product as parameters
from functools import partial
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.utils_for_testing import assert_allclose, assert_with_ulp

binary_fns = {
    "ge",
    "gt",
    "le",
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
    "rsub",
    "mul",
    "divide",
    "bias_gelu",
}


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
    ttnn_op = getattr(ttnn, ttnn_fn)

    _, a_tt = rand_bf16_gen(a_shape, device)
    _, b_tt = rand_bf16_gen(b_shape, device)

    with pytest.raises(RuntimeError) as e:
        cq_id = 0
        _ = ttnn_op(a_tt, b_tt, queue_id=cq_id, use_legacy=None)
        assert "Broadcasting rule violation" in str(e.value)


@pytest.mark.parametrize(
    "a_shape, b_shape, out_shape",
    (
        (torch.Size([1, 1, 1, 1]), torch.Size([5, 3, 32, 32]), torch.Size([5, 3, 1, 32])),
        (torch.Size([1, 1, 1, 1]), torch.Size([5, 3, 128, 64]), torch.Size([5, 3, 1, 64])),
        (torch.Size([5, 2, 64, 1]), torch.Size([1, 3, 1, 128]), torch.Size([5, 3, 64, 128])),
    ),
)
@pytest.mark.parametrize(
    "ttnn_fn",
    binary_fns,
)
def test_binary_opt_output_invalid_bcast(a_shape, b_shape, out_shape, ttnn_fn, device):
    torch.manual_seed(0)
    ttnn_op = getattr(ttnn, ttnn_fn)

    _, input_tensor_a = rand_bf16_gen(a_shape, device)
    _, input_tensor_b = rand_bf16_gen(b_shape, device)
    _, out_tt = rand_bf16_gen(out_shape, device)

    with pytest.raises(
        RuntimeError, match=r"Shape of Output tensor.+ provided does not match the broadcasted output shape .+"
    ):
        cq_id = 0
        ttnn_op(input_tensor_a, input_tensor_b, queue_id=cq_id, output_tensor=out_tt, use_legacy=None)


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


def rand_bf16_gen(shape, device, *, min=0, max=1, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    pt = torch.rand(shape, dtype=torch.bfloat16) * (max - min) + min
    tt = ttnn.from_torch(pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)
    return pt, tt


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
        parameters({"eq", "ne"}, {square_lhs, sin_rhs, exp_floor_lhs_exp_rhs, log_lhs_sqrt_abs_post}),
        parameters({"logaddexp", "logaddexp2"}, {floor_lhs_ceil_rhs_cos_post}),
        parameters({"ge", "lt", "le"}, {exp_floor_lhs_exp_rhs, log_lhs_sqrt_abs_post}),
        parameters({"logical_and", "logical_or", "logical_xor", "bias_gelu"}, {log_lhs_sqrt_abs_post}),
        parameters({"divide"}, {exp_post, tanh_post, exp2_post, expm1_post, i0_post, tan_post}),
        parameters({"sub"}, {log_post, log2_post, log10_post}),
        parameters({"ldexp"}, {erfinv_post, tan_post, floor_post, ceil_post}),
        parameters({"squared_difference"}, {erfinv_post, i0_post}),
        parameters({"add"}, {tan_post, tanh_post}),
        {("mul", log_lhs_sqrt_abs_post)},
    ),
)
def test_binary_scalar_ops(a_shape, b_shape, ttnn_fn, activations, device):
    torch.manual_seed(0)
    ttnn_op = getattr(ttnn, ttnn_fn)
    lhs, rhs, post = ([getattr(ttnn.UnaryOpType, op) for op in ops] for ops in activations)
    golden_lhs, golden_rhs, golden_post = ((activation_fns[op] for op in ops) for ops in activations)
    # make 0 exclusive for rhs of div
    min, max = (1, 0) if ttnn_fn == "divide" else (0, 1)

    a_pt, a_tt = rand_bf16_gen(a_shape, device)
    b_pt, b_tt = rand_bf16_gen(b_shape, device, min=min, max=max)

    try:
        out_tt = ttnn_op(
            a_tt,
            b_tt,
            input_tensor_a_activations=lhs,
            input_tensor_b_activations=rhs,
            activations=post,
            use_legacy=None,
        )
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
                *parameters({"ge", "gt", "le", "lt"}, {sin_rhs}),
            }
            return compare_pcc(tt, pt, 0.98) if (ttnn_fn, activations) in imprecise_cases else compare_pcc(tt, pt)

        assert compare([out_tt], [out_pt])
    except RuntimeError as e:
        allowed = ("lhs_activations.size() <= 1", "rhs_activations.empty()")
        if any(msg in str(e) for msg in allowed):
            pass
        else:
            raise


activation_with_param_fns = {
    "ADD_UNARY_SFPU": torch.add,
    "SUB_UNARY_SFPU": torch.sub,
    "MUL_UNARY_SFPU": torch.mul,
    "DIV_UNARY_SFPU": torch.div,
    "POWER": torch.pow,
}


@pytest.mark.parametrize(
    "a_shape, b_shape",
    (
        (torch.Size([1, 1, 1, 1]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 1, 64, 1]), torch.Size([1, 3, 1, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([1, 3, 128, 1])),
    ),
)
@pytest.mark.parametrize("ttnn_fn", ("add", "sub", "mul", "divide"))
@pytest.mark.parametrize(
    "post_activations",
    (
        (),
        (("ADD_UNARY_SFPU", 7),),
        (("SUB_UNARY_SFPU", 6),),
        (("MUL_UNARY_SFPU", 5),),
        (("DIV_UNARY_SFPU", 4),),
        (("POWER", 3),),
    ),
)
def test_binary_scalar_ops_with_unary_param(a_shape, b_shape, ttnn_fn, post_activations, device):
    torch.manual_seed(0)
    ttnn_op = getattr(ttnn, ttnn_fn)
    post = [(getattr(ttnn.UnaryOpType, op), param) for op, param in post_activations]
    golden_post = ((lambda x: activation_with_param_fns[op](x, param)) for op, param in post_activations)
    # make 0 exclusive for rhs of div
    min, max = (1, 0) if ttnn_fn == "divide" else (0, 1)

    a_pt, a_tt = rand_bf16_gen(a_shape, device)
    b_pt, b_tt = rand_bf16_gen(b_shape, device, min=min, max=max)

    out_tt = ttnn_op(a_tt, b_tt, activations=post, use_legacy=None)

    golden_fn = ttnn.get_golden_function(ttnn_op)
    out_pt = golden_fn(a_pt, b_pt).bfloat16()

    for golden_activation in golden_post:
        out_pt = golden_activation(out_pt).bfloat16()

    assert compare_pcc([out_tt], [out_pt])


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 1, 1]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 1, 64, 1]), torch.Size([1, 3, 1, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([1, 3, 128, 1])),
        (torch.Size([5, 1, 1]), torch.Size([1, 32, 128])),
    ),
)
@pytest.mark.parametrize(
    "ttnn_fn",
    [
        ttnn.add,
        ttnn.sub,
        ttnn.mul,
        ttnn.divide,
        ttnn.rsub,
        ttnn.eq,
        ttnn.ne,
        ttnn.gt,
        ttnn.ge,
        ttnn.lt,
        ttnn.le,
        ttnn.logical_or,
        ttnn.logical_xor,
        ttnn.logical_and,
        ttnn.ldexp,
        ttnn.logaddexp,
        ttnn.logaddexp2,
        ttnn.squared_difference,
        ttnn.bias_gelu,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    ([ttnn.float32]),
)
def test_binary_sfpu_ops(input_shapes, dtype, ttnn_fn, device):
    torch.manual_seed(0)
    a_shape, b_shape = input_shapes

    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.float32), dtype)(a_shape)
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.float32), dtype)(b_shape)

    a_tt = ttnn.from_torch(
        a_pt,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b_tt = ttnn.from_torch(
        b_pt,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cq_id = 0
    out_tt = ttnn_fn(a_tt, b_tt, queue_id=cq_id, use_legacy=None)
    tt_out = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn_fn)
    out_pt = golden_fn(a_pt, b_pt)
    status = ttnn.pearson_correlation_coefficient(out_pt, tt_out)
    assert status >= 0.999


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 1, 1]), torch.Size([5, 3, 32, 32]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 1, 64, 1]), torch.Size([1, 3, 1, 128]), torch.Size([5, 3, 64, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([1, 3, 128, 1]), torch.Size([5, 3, 128, 64])),
        (torch.Size([5, 1, 1]), torch.Size([1, 32, 128]), torch.Size([5, 32, 128])),
    ),
)
@pytest.mark.parametrize(
    "ttnn_fn",
    [
        ttnn.add,
        ttnn.sub,
        ttnn.mul,
        ttnn.divide,
        ttnn.rsub,
        ttnn.eq,
        ttnn.ne,
        ttnn.gt,
        ttnn.ge,
        ttnn.lt,
        ttnn.le,
        ttnn.logical_or,
        ttnn.logical_xor,
        ttnn.logical_and,
        ttnn.ldexp,
        ttnn.logaddexp,
        ttnn.logaddexp2,
        ttnn.squared_difference,
        ttnn.bias_gelu,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    ([ttnn.float32]),
)
def test_binary_sfpu_opt_out(input_shapes, dtype, ttnn_fn, device):
    torch.manual_seed(0)
    a_shape, b_shape, out_shape = input_shapes

    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.float32), dtype)(a_shape)
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.float32), dtype)(b_shape)
    out = gen_func_with_cast_tt(partial(torch_random, low=0, high=1, dtype=torch.float32), dtype)(out_shape)

    a_tt = ttnn.from_torch(
        a_pt,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b_tt = ttnn.from_torch(
        b_pt,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_tt = ttnn.from_torch(
        out,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cq_id = 0
    ttnn_fn(a_tt, b_tt, queue_id=cq_id, output_tensor=out_tt, use_legacy=None)
    tt_out = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn_fn)
    out_pt = golden_fn(a_pt, b_pt)
    status = ttnn.pearson_correlation_coefficient(out_pt, tt_out)
    assert status >= 0.999


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 1, 1]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 1, 64, 1]), torch.Size([1, 3, 1, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([1, 3, 128, 1])),
        (torch.Size([5, 1, 1]), torch.Size([1, 32, 128])),
    ),
)
@pytest.mark.parametrize(
    "ttnn_fn",
    [
        ttnn.bitwise_and,
        ttnn.bitwise_or,
        ttnn.bitwise_xor,
        ttnn.bitwise_left_shift,
        ttnn.bitwise_right_shift,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    ([ttnn.int32]),
)
def test_binary_sfpu_bitwise_ops(input_shapes, dtype, ttnn_fn, device):
    torch.manual_seed(0)
    a_shape, b_shape = input_shapes

    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.int32), dtype)(a_shape)
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=0, high=31, dtype=torch.int32), dtype)(b_shape)

    a_tt = ttnn.from_torch(
        a_pt,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b_tt = ttnn.from_torch(
        b_pt,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cq_id = 0
    out_tt = ttnn_fn(a_tt, b_tt, queue_id=cq_id, use_legacy=None)
    tt_out = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn_fn)
    out_pt = golden_fn(a_pt, b_pt)

    status = ttnn.pearson_correlation_coefficient(out_pt, tt_out)
    assert status >= 0.999


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 1, 1]), torch.Size([5, 3, 32, 32]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 1, 64, 1]), torch.Size([1, 3, 1, 128]), torch.Size([5, 3, 64, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([1, 3, 128, 1]), torch.Size([5, 3, 128, 64])),
        (torch.Size([5, 1, 1]), torch.Size([1, 32, 128]), torch.Size([5, 32, 128])),
    ),
)
@pytest.mark.parametrize(
    "ttnn_fn",
    [
        ttnn.bitwise_and,
        ttnn.bitwise_or,
        ttnn.bitwise_xor,
        ttnn.bitwise_left_shift,
        ttnn.bitwise_right_shift,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    ([ttnn.int32]),
)
def test_bitwise_opt_output(input_shapes, dtype, ttnn_fn, device):
    torch.manual_seed(0)
    a_shape, b_shape, out_shape = input_shapes

    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.int32), dtype)(a_shape)
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=0, high=31, dtype=torch.int32), dtype)(b_shape)
    out = gen_func_with_cast_tt(partial(torch_random, low=0, high=1, dtype=torch.int32), dtype)(out_shape)

    a_tt = ttnn.from_torch(
        a_pt,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b_tt = ttnn.from_torch(
        b_pt,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_tt = ttnn.from_torch(
        out,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cq_id = 0
    ttnn_fn(a_tt, b_tt, queue_id=cq_id, output_tensor=out_tt, use_legacy=None)
    tt_out = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn_fn)
    out_pt = golden_fn(a_pt, b_pt)

    status = ttnn.pearson_correlation_coefficient(out_pt, tt_out)
    assert status >= 0.999


binary_inplace_fns = {
    "add_",
    "sub_",
    "mul_",
    "divide_",
    "rsub_",
    "gt_",
    "lt_",
    "le_",
    "ge_",
    "eq_",
    "ne_",
    "logical_and_",
    "logical_or_",
    "logical_xor_",
    "ldexp_",
    "logaddexp_",
    "logaddexp2_",
    "squared_difference_",
    "bias_gelu_",
}


@pytest.mark.parametrize(
    "a_shape, b_shape",
    (
        (torch.Size([5, 3, 128, 64]), torch.Size([1, 3, 128, 1])),
        (torch.Size([5, 3, 32, 32]), torch.Size([1, 1, 1, 1])),
        (torch.Size([5, 1, 1, 128]), torch.Size([5, 1, 1, 1])),
    ),
)
@pytest.mark.parametrize(
    "ttnn_fn, activations",
    {
        *parameters(
            binary_inplace_fns,
            {
                no_activations,
                square_lhs,
                sin_rhs,
                floor_lhs_ceil_rhs_cos_post,
                exp_floor_lhs_exp_rhs,
                log_lhs_sqrt_abs_post,
            },
        )
    }.difference(
        parameters({"eq_", "ne_"}, {square_lhs, sin_rhs, exp_floor_lhs_exp_rhs}),
        parameters({"lt_", "ge_"}, {exp_floor_lhs_exp_rhs}),
        parameters({"le_"}, {sin_rhs, log_lhs_sqrt_abs_post}),
        parameters({"bias_gelu_"}, {log_lhs_sqrt_abs_post}),
        parameters({"mul_"}, {log_lhs_sqrt_abs_post}),
    ),
)
def test_inplace_binary_ops_with_tensor(a_shape, b_shape, ttnn_fn, activations, device):
    torch.manual_seed(0)

    ttnn_op = getattr(ttnn, ttnn_fn)
    lhs, rhs, post = ([getattr(ttnn.UnaryOpType, op) for op in ops] for ops in activations)
    golden_lhs, golden_rhs, golden_post = ((activation_fns[op] for op in ops) for ops in activations)
    min, max = (1, 0) if ttnn_fn == "divide_" else (0, 1)

    torch_input_tensor_a, input_tensor_a = rand_bf16_gen(a_shape, device)
    torch_input_tensor_b, input_tensor_b = rand_bf16_gen(b_shape, device, min=min, max=max)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    for golden_activation in golden_lhs:
        torch_input_tensor_a = golden_activation(torch_input_tensor_a).bfloat16()

    for golden_activation in golden_rhs:
        torch_input_tensor_b = golden_activation(torch_input_tensor_b).bfloat16()

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b).bfloat16()

    for golden_activation in golden_post:
        torch_output_tensor = golden_activation(torch_output_tensor).bfloat16()

    ttnn_op(
        input_tensor_a,
        input_tensor_b,
        input_tensor_a_activations=lhs,
        input_tensor_b_activations=rhs,
        activations=post,
        use_legacy=None,
    )
    output_tensor = ttnn.to_torch(input_tensor_a)
    assert output_tensor.shape == torch_output_tensor.shape

    def compare(output_tensor, torch_output_tensor):
        imprecise_cases = {
            *parameters(
                {"logaddexp2_"},
                {exp_floor_lhs_exp_rhs, no_activations, sin_rhs, log_lhs_sqrt_abs_post, square_lhs},
            ),
            *parameters({"bias_gelu_"}, {no_activations, sin_rhs, square_lhs}),
            *parameters({"gt_", "le_", "ge_", "lt_"}, {sin_rhs, square_lhs}),
        }

        return (
            ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.98
            if (ttnn_fn, activations) in imprecise_cases
            else ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.999
        )

    assert compare(output_tensor, torch_output_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([5, 3, 32, 32]), torch.Size([1, 1, 1, 1])),
        (torch.Size([5, 3, 64, 128]), torch.Size([1, 3, 1, 128])),
        (torch.Size([5, 3, 128, 64]), torch.Size([1, 1, 128, 1])),
        (torch.Size([5, 32, 128]), torch.Size([5, 1, 1])),
    ),
)
@pytest.mark.parametrize(
    "ttnn_fn",
    binary_inplace_fns,
)
def test_inplace_binary_ops_fp32(input_shapes, ttnn_fn, device):
    torch.manual_seed(0)
    a_shape, b_shape = input_shapes
    ttnn_op = getattr(ttnn, ttnn_fn)
    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-50, high=50, dtype=torch.float32), ttnn.float32
    )(a_shape)
    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-50, high=50, dtype=torch.float32), ttnn.float32
    )(b_shape)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    cq_id = 0
    ttnn_op(input_tensor_a, input_tensor_b, queue_id=cq_id, use_legacy=None)
    output_tensor = ttnn.to_torch(input_tensor_a)

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


@pytest.mark.parametrize(
    "ttnn_fn",
    [
        "add_",
        "sub_",
        "mul_",
        "divide_",
        "rsub_",
        "gt_",
        "lt_",
        "le_",
        "ge_",
        "eq_",
        "ne_",
        "squared_difference_",
    ],
)
@pytest.mark.parametrize(
    "a_shape",
    (
        torch.Size([5, 3, 128, 64]),
        torch.Size([1, 1, 1, 1]),
        torch.Size([5, 3, 32, 32]),
        torch.Size([16, 1]),
        torch.Size([1, 1, 32]),
        torch.Size([920, 1, 256]),
    ),
)
@pytest.mark.parametrize("scalar", [-0.25, -16.5, 0.0, 0.05, 1.7, 19.0])
def test_inplace_binary_with_scalar(a_shape, scalar, ttnn_fn, device):
    torch.manual_seed(0)

    ttnn_op = getattr(ttnn, ttnn_fn)
    torch_input_tensor_a, input_tensor_a = rand_bf16_gen(a_shape, device)
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, scalar)

    ttnn_op(input_tensor_a, scalar, use_legacy=None)
    output_tensor = ttnn.to_torch(input_tensor_a)
    assert output_tensor.shape == torch_output_tensor.shape
    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99


profile_a_b_shape_pairs = [
    # [[8192, 8192], [8192, 8192]],
    [[1, 8192], [8192, 8192]],
    # [[8192, 8192], [1, 8192]],
    # [8192, 1], [8192, 8192]],
    # [[8192, 8192], [8192, 1]],
    # [[1, 8192], [8192, 1]],
    # [[8192, 1], [1, 8192]],
    # [[1, 1], [8192, 8192]],
    # [[8192, 8192], [1, 1]],
]


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    (
        (torch.bfloat16, ttnn.bfloat16),
        # (torch.float32, ttnn.float32)
    ),
)
@pytest.mark.parametrize(
    "memory_config_input",
    [ttnn.DRAM_MEMORY_CONFIG],
)
@pytest.mark.parametrize(
    "use_legacy",
    [
        # True,
        False,
    ],
)
@pytest.mark.parametrize("a_and_b_shape", profile_a_b_shape_pairs)
def test_binary_bcast_profile(device, dtype_pt, dtype_tt, a_and_b_shape, memory_config_input, use_legacy):
    torch.manual_seed(0)
    a_shape, b_shape = a_and_b_shape

    torch_input_tensor_a = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(
        a_shape
    )
    torch_input_tensor_b = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(
        b_shape
    )

    torch_result = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config_input
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config_input
    )
    for _ in range(2):
        output = ttnn.add(input_tensor_a, input_tensor_b, memory_config=memory_config_input, use_legacy=use_legacy)
        output = ttnn.to_torch(output)

        assert (
            output.shape == torch_result.shape
        ), f"Output shape {output.shape} does not match torch shape {torch_result.shape}"

        # use assert_allclose in the future, needs sutiable tolerance
        # assert_allclose(torch_result, output, rtol=1e-02, atol=1e-02)
        assert_with_pcc(torch_result, output)
        ttnn.synchronize_device(device)


height_sharded_memory_config_1 = ttnn.create_sharded_memory_config(
    [1024, 256],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (7, 7))}),
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

height_sharded_memory_config_2 = ttnn.create_sharded_memory_config(
    [1024, 128],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (7, 7))}),
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
height_sharded_memory_config_3 = ttnn.create_sharded_memory_config(
    [4096, 64],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (7, 7))}),
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
height_sharded_memory_config_4 = ttnn.create_sharded_memory_config(
    [4096, 32],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (7, 7))}),
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_decoder_program_cache(dtype_pt, dtype_tt, device):
    compute_grid_size = device.compute_with_storage_grid_size()
    if compute_grid_size.x < 8 or compute_grid_size.y < 8:
        pytest.skip("Test is skipped because the device does not have full coregrid 8x8")

    torch.manual_seed(0)
    # device.disable_and_clear_program_cache()

    input_tensors = (
        (torch.Size([1, 1, 65536, 256]), torch.Size([1, 1, 65536, 256]), height_sharded_memory_config_1),
        (torch.Size([1, 1, 65536, 128]), torch.Size([1, 1, 65536, 128]), height_sharded_memory_config_2),
        (torch.Size([1, 1, 262144, 64]), torch.Size([1, 1, 262144, 64]), height_sharded_memory_config_3),
        (torch.Size([1, 1, 262144, 3]), torch.Size([1, 1, 262144, 3]), height_sharded_memory_config_4),
    )
    for _i in range(2):
        for a_shape, b_shape, sharded_config in input_tensors:
            input_combinations = (
                (ttnn.DRAM_MEMORY_CONFIG, sharded_config),
                (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
            )
            for src_a_config, src_b_config in input_combinations:
                a_pt = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=dtype_pt), dtype_tt)(
                    a_shape
                )
                b_pt = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=dtype_pt), dtype_tt)(
                    b_shape
                )

                a_tt = ttnn.from_torch(
                    a_pt,
                    dtype=dtype_tt,
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=src_a_config,
                )
                b_tt = ttnn.from_torch(
                    b_pt,
                    dtype=dtype_tt,
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=src_b_config,
                )

                out_pt = torch.add(a_pt, b_pt)
                ttnn.add(a_tt, b_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, output_tensor=a_tt, use_legacy=False)
                out_tt_interleaved = ttnn.to_torch(a_tt)

                pcc = ttnn.pearson_correlation_coefficient(out_tt_interleaved, out_pt)
                # print(f"Pearson correlation coefficient: {pcc}")
                print(f"device.num_program_cache_entries(): {device.num_program_cache_entries()}")
                assert pcc >= 0.99988
    assert (
        device.num_program_cache_entries() == 5
    ), f"device.num_program_cache_entries(): {device.num_program_cache_entries()}"


@pytest.mark.parametrize(
    "shapes",
    [
        # no subtile bcast
        [[1, 16, 32], [8, 16, 32]],
        # scalar bcast
        [[8, 16, 32], [1, 1, 1]],
        [[1, 1, 1], [8, 16, 32]],
        # col bcast
        [[1, 16, 1], [8, 16, 32]],
        [[8, 16, 32], [8, 16, 1]],
        # row bcast
        [[8, 16, 32], [8, 1, 32]],
        [[8, 1, 32], [8, 16, 32]],
        # row col mixed bcast
        [[1, 1, 32], [8, 16, 1]],
        [[8, 16, 1], [1, 1, 32]],
    ],
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.int32, ttnn.int32),
        (torch.float32, ttnn.float32),
    ],
)
def test_remainder_implicit_broadcast(device, shapes, torch_dtype, ttnn_dtype):
    torch.manual_seed(0)

    if torch_dtype == torch.int32:
        torch_input_tensor_a = torch.randint(
            torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max, shapes[0], dtype=torch.int32
        )
        torch_input_tensor_b = torch.randint(
            torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max, shapes[1], dtype=torch.int32
        )
    else:
        torch_input_tensor_a = torch.empty(shapes[0], dtype=torch.float32).uniform_(
            torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max
        )
        torch_input_tensor_b = torch.empty(shapes[1], dtype=torch.float32).uniform_(
            torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max
        )

    golden_function = ttnn.get_golden_function(ttnn.remainder)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.remainder(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


height_sharded_memory_config = ttnn.create_sharded_memory_config(
    [32 * 2, 32 * 4 * 8],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 3))}),
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

width_sharded_memory_config = ttnn.create_sharded_memory_config(
    [32 * 4 * 8, 32 * 2],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 3))}),
    strategy=ttnn.ShardStrategy.WIDTH,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

block_sharded_memory_config = ttnn.create_sharded_memory_config(
    [32 * 8, 32 * 8],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 3))}),
    strategy=ttnn.ShardStrategy.BLOCK,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)


@pytest.mark.parametrize(
    "a_shape, b_shape",
    ((torch.Size([1, 1, 32 * 4 * 8, 32 * 4 * 8]), torch.Size([1, 1, 32 * 4 * 8, 32 * 4 * 8])),),
)
@pytest.mark.parametrize(
    "a_config, b_config, out_config",
    [
        # [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        # [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, height_sharded_memory_config],
        # [ttnn.DRAM_MEMORY_CONFIG, height_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG],
        # [ttnn.DRAM_MEMORY_CONFIG, height_sharded_memory_config, height_sharded_memory_config],
        # [height_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        # [height_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG, height_sharded_memory_config],
        # [height_sharded_memory_config, height_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG],
        [height_sharded_memory_config, height_sharded_memory_config, height_sharded_memory_config],
        # [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, width_sharded_memory_config],
        # [ttnn.DRAM_MEMORY_CONFIG, width_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG],
        # [ttnn.DRAM_MEMORY_CONFIG, width_sharded_memory_config, width_sharded_memory_config],
        # [width_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        # [width_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG, width_sharded_memory_config],
        # [width_sharded_memory_config, width_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG],
        # [width_sharded_memory_config, width_sharded_memory_config, width_sharded_memory_config],
        # [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, block_sharded_memory_config],
        # [ttnn.DRAM_MEMORY_CONFIG, block_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG],
        # [ttnn.DRAM_MEMORY_CONFIG, block_sharded_memory_config, block_sharded_memory_config],
        # [block_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        # [block_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG, block_sharded_memory_config],
        # [block_sharded_memory_config, block_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG],
        # [block_sharded_memory_config, block_sharded_memory_config, block_sharded_memory_config],
    ],
)
@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_no_profile(a_shape, b_shape, a_config, b_config, out_config, dtype_pt, dtype_tt, device):
    torch.manual_seed(0)
    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=dtype_pt), dtype_tt)(a_shape)
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=dtype_pt), dtype_tt)(b_shape)

    a_tt = ttnn.from_torch(
        a_pt,
        dtype=dtype_tt,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=a_config,
    )
    b_tt = ttnn.from_torch(
        b_pt,
        dtype=dtype_tt,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=b_config,
    )

    out_pt = torch.add(a_pt, b_pt)
    out_tt = ttnn.add(a_tt, b_tt, memory_config=out_config, use_legacy=None)
    assert_with_pcc(ttnn.to_torch(out_tt), out_pt)


@pytest.mark.parametrize(
    "test_shapes",
    ([[1, 1, 2304, 1792], [1, 1, 2112, 1792]],),
)
# HEIGHT SHARDING test - tests program cache with different shapes
def test_inplace_sub_height_sharded_different_shapes(test_shapes, device):
    grid_size = device.compute_with_storage_grid_size()

    if grid_size.x < 5 or grid_size.y < 4:
        pytest.skip(
            f"This test is intended to run on devices with at least 5x4 core grid. Core grid: {grid_size.x}x{grid_size.y}"
        )

    import math

    for iteration, shape in enumerate(test_shapes):
        # Generate random tensors
        torch_input_tensor_a = torch.rand(shape, dtype=torch.bfloat16)
        torch_input_tensor_b = torch.rand(shape, dtype=torch.bfloat16)

        # Calculate shard dimensions
        total_rows = math.prod(shape[:-1])  # 2304 or 2112
        total_cols = shape[-1]  # 1792

        num_cores = 20

        shard_height = math.ceil(total_rows / num_cores / 32) * 32
        shard_width = math.ceil(total_cols / 32) * 32

        # Create HEIGHT_SHARDED memory config
        sharded_memory_config = ttnn.create_sharded_memory_config(
            shape=(shard_height, shard_width),
            core_grid=ttnn.CoreGrid(y=4, x=5),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # Create tensors on device as HEIGHT_SHARDED
        input_tensor_a = ttnn.from_torch(
            torch_input_tensor_a,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=sharded_memory_config,
        )
        input_tensor_b = ttnn.from_torch(
            torch_input_tensor_b,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=sharded_memory_config,
        )

        ttnn.sub_(input_tensor_a, input_tensor_b)
        output_tensor = ttnn.to_torch(input_tensor_a)

        # Validate results
        golden_fn = ttnn.get_golden_function(ttnn.sub_)
        torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
        assert_allclose(torch_output_tensor, output_tensor)

        # Cleanup before next iteration
        ttnn.deallocate(input_tensor_a)
        ttnn.deallocate(input_tensor_b)
