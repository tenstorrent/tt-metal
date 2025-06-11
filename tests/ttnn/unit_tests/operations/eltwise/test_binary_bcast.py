# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    compare_pcc,
)
from models.utility_functions import torch_random
from itertools import product as parameters
from functools import partial
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt


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

    out_tt = ttnn_op(
        a_tt, b_tt, input_tensor_a_activations=lhs, input_tensor_b_activations=rhs, activations=post, use_legacy=False
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

    out_tt = ttnn_op(a_tt, b_tt, activations=post, use_legacy=False)

    golden_fn = ttnn.get_golden_function(ttnn_op)
    out_pt = golden_fn(a_pt, b_pt).bfloat16()

    for golden_activation in golden_post:
        out_pt = golden_activation(out_pt).bfloat16()

    assert compare_pcc([out_tt], [out_pt])


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
        _ = ttnn_op(a_tt, b_tt, queue_id=cq_id, use_legacy=False)
        assert "Broadcasting rule violation" in str(e.value)


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        [[1, 71, 7, 7], [1]],
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

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=False)
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
@pytest.mark.parametrize(
    "memory_config_a, memory_config_b",
    [
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    ],
)
def test_01_volume_tensors(device, a, b, c_golden, memory_config_a, memory_config_b):
    a = torch.BFloat16Tensor(a)
    b = torch.BFloat16Tensor(b)
    assert torch.add(a, b).tolist() == c_golden

    ttnn_a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config_a)
    ttnn_b = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config_b)
    ttnn_c = ttnn.add(ttnn_a, ttnn_b, use_legacy=False)
    c = ttnn.to_torch(ttnn_c).reshape((-1))

    assert c.tolist() == c_golden


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        [[2, 4, 12, 64, 64], [12, 1, 1]],
        [[12, 1, 1], [2, 4, 12, 64, 64]],
        [[3, 4, 8, 6, 32, 64], [1, 1, 8, 6, 32, 64]],
    ],
)
def test_binary_invalid_rank(device, a_shape, b_shape):
    torch.manual_seed(0)
    pt_a, tt_a = rand_bf16_gen(a_shape, device)
    pt_b, tt_b = rand_bf16_gen(b_shape, device)

    with pytest.raises(RuntimeError):
        tt_c = ttnn.add(tt_a, tt_b, use_legacy=False)


height_sharded_memory_config = ttnn.create_sharded_memory_config(
    # [320, 128], # 7 cores
    # core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
    # [160, 128],  # 14 cores
    [128, 160],
    # config 1 single rectangle start from 0, 0
    # core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (1, 6))}),
    # config 2 single rectangle not start from 0, 0
    # core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((1, 0), (2, 6))}),
    # config 3 two grids any
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((1, 0), (1, 6)), ttnn.CoreRange((3, 0), (3, 6))}),
    # [32, 128] should work with 70 cores
    # [64, 128], # 35 cores
    # core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (4, 6))}),
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.COL_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

# width sharding is not good for large and tall (w is small) tensors
# because each core may ends up with a large tensor as well, then out of L1 space
width_sharded_memory_config = ttnn.create_sharded_memory_config(
    # [2240, 64],
    # core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 1))}),
    [2240, 32],
    # core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),
    # core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((1, 0), (1, 3))}),
    # core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 1)), ttnn.CoreRange((2, 2), (2, 3))}),
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((2, 2), (2, 3)), ttnn.CoreRange((0, 0), (0, 1))}),
    strategy=ttnn.ShardStrategy.WIDTH,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

block_sharded_memory_config = ttnn.create_sharded_memory_config(
    # [320, 64], # 128 / 64 = 2, core grid is 2x6
    # core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (1, 6))}),
    # following is better, more cores
    [320, 32],  # 128 / 32 = 4, core grid is 4x6
    # core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 6))}),
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((1, 0), (4, 6))}),
    # [160, 32] will not work, because it needs core grid 4x14
    strategy=ttnn.ShardStrategy.BLOCK,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)


@pytest.mark.parametrize(
    "a_shape, b_shape",
    ((torch.Size([5, 7, 64, 128]), torch.Size([5, 7, 64, 128])),),
)
@pytest.mark.parametrize(
    "sharded_config",
    [
        height_sharded_memory_config,
        width_sharded_memory_config,
        block_sharded_memory_config,
    ],
)
@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    (
        [torch.bfloat16, ttnn.bfloat16],
        [torch.int32, ttnn.int32],
        [torch.float32, ttnn.float32],
    ),
)
def test_binary_sharded(a_shape, b_shape, sharded_config, dtype_pt, dtype_tt, device):
    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, sharded_config),
        (sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (sharded_config, sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    )

    for src_config, dst_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=dtype_pt), dtype_tt)(b_shape)

        a_tt = ttnn.from_torch(
            a_pt,
            dtype=dtype_tt,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=src_config,
        )
        b_tt = ttnn.from_torch(
            b_pt,
            dtype=dtype_tt,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=dst_config,
        )

        out_pt = torch.add(a_pt, b_pt)
        out_tt_interleaved = ttnn.add(a_tt, b_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=False)
        out_tt_interleaved = ttnn.to_torch(out_tt_interleaved)
        assert ttnn.pearson_correlation_coefficient(out_tt_interleaved, out_pt) >= 0.99988

        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=sharded_config, use_legacy=False)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert ttnn.pearson_correlation_coefficient(out_tt_sharded, out_pt) >= 0.99988


@pytest.mark.parametrize(
    "a_shape, b_shape",
    ((torch.Size([5, 7, 64, 128]), torch.Size([5, 7, 64, 128])),),
)
@pytest.mark.parametrize(
    "memory_lay_out",
    [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
)
@pytest.mark.parametrize(
    "sharded_core_grid",
    (
        ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (1, 6))}),
        ttnn.CoreRangeSet({ttnn.CoreRange((1, 0), (2, 6))}),
        ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6)), ttnn.CoreRange((1, 0), (1, 6))}),
        ttnn.CoreRangeSet({ttnn.CoreRange((1, 0), (1, 6)), ttnn.CoreRange((3, 0), (3, 6))}),
    ),
)
def test_binary_sharded_core_grid(device, a_shape, b_shape, sharded_core_grid, memory_lay_out):
    sharded_config = ttnn.create_sharded_memory_config(
        [160, 128],  # 14 cores
        core_grid=sharded_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16)(a_shape)
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16)(b_shape)

    a_tt = ttnn.from_torch(
        a_pt,
        dtype=ttnn.bfloat16,
        device=device,
        layout=memory_lay_out,
        memory_config=sharded_config,
    )
    b_tt = ttnn.from_torch(
        b_pt,
        dtype=ttnn.bfloat16,
        device=device,
        layout=memory_lay_out,
        memory_config=sharded_config,
    )

    out_pt = torch.add(a_pt, b_pt)

    out_tt_interleaved = ttnn.add(a_tt, b_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=False)
    out_tt_interleaved = ttnn.to_torch(out_tt_interleaved)
    assert ttnn.pearson_correlation_coefficient(out_tt_interleaved, out_pt) >= 0.99988

    out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=sharded_config, use_legacy=False)
    out_tt_sharded = ttnn.to_torch(out_tt_sharded)
    assert ttnn.pearson_correlation_coefficient(out_tt_sharded, out_pt) >= 0.99988


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
    out_tt = ttnn_fn(a_tt, b_tt, queue_id=cq_id, use_legacy=False)
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
    ttnn_fn(a_tt, b_tt, queue_id=cq_id, output_tensor=out_tt, use_legacy=False)
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
    out_tt = ttnn_fn(a_tt, b_tt, queue_id=cq_id, use_legacy=False)
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
    ttnn_fn(a_tt, b_tt, queue_id=cq_id, output_tensor=out_tt, use_legacy=False)
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
        use_legacy=False,
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
    "a_shape, b_shape",
    (
        (torch.Size([5, 3, 128, 64]), torch.Size([1, 3, 128, 1])),
        (torch.Size([5, 3, 32, 32]), torch.Size([1, 1, 1, 1])),
        (torch.Size([5, 1, 1, 128]), torch.Size([5, 1, 1, 1])),
        (torch.Size([1, 71, 7, 7]), torch.Size([7, 7])),
        (torch.Size([920, 1, 256]), torch.Size([256])),
        (torch.Size([4, 12, 64, 64]), torch.Size([12, 1, 1])),
    ),
)
@pytest.mark.parametrize(
    "input_dtype, pcc",
    (
        (ttnn.bfloat4_b, 0.97),
        (ttnn.bfloat8_b, 0.999),
    ),
)
@pytest.mark.parametrize("ttnn_fn", ["add", "sub", "mul", "add_", "sub_", "mul_"])
def test_bf4b_bf8b(a_shape, b_shape, input_dtype, pcc, ttnn_fn, device):
    torch.manual_seed(0)

    torch_input_tensor_a, input_tensor_a = rand_bf16_gen(a_shape, device, min=-1e3, max=1e3)
    torch_input_tensor_b, input_tensor_b = rand_bf16_gen(b_shape, device, min=-1e3, max=1e3)
    ttnn_op = getattr(ttnn, ttnn_fn)
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=input_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_input_tensor_a = ttnn.to_torch(input_tensor_a)
    torch_input_tensor_b = ttnn.to_torch(input_tensor_b)

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b)

    output_tensor = ttnn_op(input_tensor_a, input_tensor_b, use_legacy=False)
    output_tensor = ttnn.to_torch(input_tensor_a if ttnn_fn.endswith("_") else output_tensor)
    assert output_tensor.shape == torch_output_tensor.shape
    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= pcc


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
    ttnn_op(input_tensor_a, input_tensor_b, queue_id=cq_id, use_legacy=False)
    output_tensor = ttnn.to_torch(input_tensor_a)

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


@pytest.mark.parametrize(
    "a_shape, b_shape",
    (
        (torch.Size([1, 1, 31, 32]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 2, 64, 1]), torch.Size([1, 3, 1, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([2, 3, 128, 1])),
        (torch.Size([2, 2, 3, 128, 1]), torch.Size([2, 3, 128, 1])),
    ),
)
@pytest.mark.parametrize(
    "ttnn_fn",
    binary_inplace_fns,
)
def test_inplace_binary_ops_invalid_bcast(a_shape, b_shape, ttnn_fn, device):
    torch.manual_seed(0)
    ttnn_op = getattr(ttnn, ttnn_fn)

    _, input_tensor_a = rand_bf16_gen(a_shape, device)
    _, input_tensor_b = rand_bf16_gen(b_shape, device)

    with pytest.raises(RuntimeError):
        cq_id = 0
        ttnn_op(input_tensor_a, input_tensor_b, queue_id=cq_id, use_legacy=False)


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

    ttnn_op(input_tensor_a, scalar, use_legacy=False)
    output_tensor = ttnn.to_torch(input_tensor_a)
    assert output_tensor.shape == torch_output_tensor.shape
    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99


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
        ttnn_op(input_tensor_a, input_tensor_b, queue_id=cq_id, output_tensor=out_tt, use_legacy=False)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    (
        [torch.bfloat16, ttnn.bfloat16],
        [torch.int32, ttnn.int32],
        [torch.float32, ttnn.float32],
    ),
)
def test_binary_sharded_bcast_w(device, dtype_pt, dtype_tt):
    a_shape = torch.Size([5, 7, 2 * 32, 4 * 32])
    b_shape = torch.Size([5, 7, 2 * 32, 1])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [10 * 32, 4 * 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [10 * 32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config),
    )

    for src_config, dst_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

        a_tt = ttnn.from_torch(
            a_pt,
            dtype=dtype_tt,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=src_config,
        )
        b_tt = ttnn.from_torch(
            b_pt,
            dtype=dtype_tt,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=dst_config,
        )

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=False)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        torch.testing.assert_close(out_tt_sharded, out_pt)

        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=a_sharded_config, use_legacy=False)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        torch.testing.assert_close(out_tt_sharded, out_pt)


@pytest.mark.parametrize(
    "a_shape, b_shape, a_shard_size, b_shard_size, core_range",
    (
        [
            torch.Size([7, 5, 2 * 32, 4 * 32]),
            torch.Size([5, 7, 2 * 32, 1]),
            [10 * 32, 4 * 32],
            [10 * 32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        ],
        [
            torch.Size([5, 6, 1, 2]),
            torch.Size([5, 6, 1, 1]),
            [32, 32],
            [32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (4, 5))}),
        ],
    ),
)
def test_binary_sharded_invalid_bcast(a_shape, b_shape, a_shard_size, b_shard_size, core_range, device):
    a_sharded_config = ttnn.create_sharded_memory_config(
        a_shard_size,
        core_grid=core_range,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        b_shard_size,
        core_grid=core_range,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    a_pt, a_tt = rand_bf16_gen(a_shape, device, memory_config=a_sharded_config)
    b_pt, b_tt = rand_bf16_gen(b_shape, device, memory_config=b_sharded_config)

    with pytest.raises(RuntimeError):
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=a_sharded_config, use_legacy=False)


@pytest.mark.parametrize(
    "a_shape, b_shape",
    ((torch.Size([1, 5, 7, 2, 35]), torch.Size([1, 5, 7, 2, 35])),),
)
@pytest.mark.parametrize(
    "shard_type, shard_size, core_range",
    (
        [ttnn.ShardStrategy.HEIGHT, [32, 64], ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (4, 6))})],
        [ttnn.ShardStrategy.WIDTH, [35 * 32, 32], ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 1))})],
        [ttnn.ShardStrategy.BLOCK, [32 * 5, 32], ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (1, 6))})],
    ),
)
def test_binary_sharded_small_tile(a_shape, b_shape, shard_type, shard_size, core_range, device):
    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16)(a_shape)
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16)(b_shape)

    shard_config = ttnn.create_sharded_memory_config(
        shard_size,
        core_grid=core_range,
        strategy=shard_type,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    a_tt = ttnn.from_torch(
        a_pt,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=shard_config,
    )
    b_tt = ttnn.from_torch(
        b_pt,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=shard_config,
    )

    out_pt = torch.add(a_pt, b_pt)
    out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=shard_config, use_legacy=False)
    out_tt_sharded = ttnn.to_torch(out_tt_sharded)
    assert ttnn.pearson_correlation_coefficient(out_tt_sharded, out_pt) >= 0.99988


@pytest.mark.parametrize(
    "ttnn_fn",
    [
        ttnn.add,
        ttnn.sub,
        ttnn.mul,
        # ttnn.divide,
        # ttnn.rsub,
        ttnn.eq,
        ttnn.ne,
        ttnn.gt,
        ttnn.ge,
        ttnn.lt,
        # ttnn.le,
        ttnn.logical_or,
        # ttnn.logical_xor,
        ttnn.logical_and,
        # ttnn.ldexp,
        # ttnn.logaddexp,
        # ttnn.logaddexp2,
        # ttnn.squared_difference,
        # ttnn.bias_gelu,
    ],
)
@pytest.mark.parametrize(
    "a_shape, b_shape, shard_type, shard_size, core_range",
    (
        [
            torch.Size([5, 7, 2, 35]),
            torch.Size([5, 7, 2, 35]),
            ttnn.ShardStrategy.HEIGHT,
            [64, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (4, 6))}),
        ],
        [
            torch.Size([5, 7, 2, 35]),
            torch.Size([5, 7, 2, 35]),
            ttnn.ShardStrategy.WIDTH,
            [32, 35 * 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 1))}),
        ],
        [
            torch.Size([5, 7, 2, 35]),
            torch.Size([5, 7, 2, 35]),
            ttnn.ShardStrategy.BLOCK,
            [32, 32 * 5],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (6, 1))}),
        ],
        [
            torch.Size([1, 1, 1024, 1024]),
            torch.Size([1, 1, 1024, 1024]),
            ttnn.ShardStrategy.HEIGHT,
            [1024, 128],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (7, 0))}),
        ],
        [
            torch.Size([1, 1, 1024, 1024]),
            torch.Size([1, 1, 1024, 1024]),
            ttnn.ShardStrategy.WIDTH,
            [128, 1024],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (7, 0))}),
        ],
        [
            torch.Size([1, 1, 1024, 1024]),
            torch.Size([1, 1, 1024, 1024]),
            ttnn.ShardStrategy.BLOCK,
            [256, 256],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 3))}),
        ],
        # with broadcasting on w
        [
            torch.Size([1, 7, 32, 32]),
            torch.Size([1, 7, 32, 1]),
            ttnn.ShardStrategy.HEIGHT,
            [32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (6, 0))}),
        ],
    ),
)
def test_binary_sharded_col_major(a_shape, b_shape, shard_type, shard_size, core_range, ttnn_fn, device):
    golden_function = ttnn.get_golden_function(ttnn_fn)

    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16)(a_shape)
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16)(b_shape)

    shard_config = ttnn.create_sharded_memory_config(
        shard_size,
        core_grid=core_range,
        strategy=shard_type,
        orientation=ttnn.ShardOrientation.COL_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, shard_config),
        (shard_config, ttnn.DRAM_MEMORY_CONFIG),
        (shard_config, shard_config),
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    )

    for src_config, dst_config in input_combinations:
        a_tt = ttnn.from_torch(
            a_pt,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=src_config,
        )
        b_tt = ttnn.from_torch(
            b_pt,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=dst_config,
        )

        out_pt = golden_function(a_pt, b_pt)

        out_tt_sharded = ttnn_fn(a_tt, b_tt, memory_config=shard_config, use_legacy=False)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert ttnn.pearson_correlation_coefficient(out_tt_sharded, out_pt) >= 0.99988

        out_tt_interleaved = ttnn_fn(a_tt, b_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=False)
        out_tt_interleaved = ttnn.to_torch(out_tt_interleaved)
        assert ttnn.pearson_correlation_coefficient(out_tt_interleaved, out_pt) >= 0.99988


@pytest.mark.parametrize(
    "a_shape, b_shape",
    ((torch.Size([5, 7, 32, 64]), torch.Size([5, 7, 32, 64])),),
)
@pytest.mark.parametrize(
    "shard_type, core_coord",
    (
        [ttnn.ShardStrategy.HEIGHT, ttnn.CoreGrid(x=5, y=7)],  # shard [32, 64],
        [ttnn.ShardStrategy.WIDTH, ttnn.CoreGrid(x=2, y=1)],  # shard [35 * 32, 32],
        [ttnn.ShardStrategy.BLOCK, ttnn.CoreGrid(x=2, y=5)],  # shard [32 * 7, 32],
    ),
)
def test_binary_sharded_auto(a_shape, b_shape, shard_type, core_coord, device):
    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16)(a_shape)
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16)(b_shape)

    shard_config = ttnn.create_sharded_memory_config(
        a_shape,
        core_grid=core_coord,
        strategy=shard_type,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=False,
    )

    a_tt = ttnn.from_torch(
        a_pt,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=shard_config,
    )
    b_tt = ttnn.from_torch(
        b_pt,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=shard_config,
    )

    out_pt = torch.add(a_pt, b_pt)
    out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=shard_config, use_legacy=False)
    out_tt_sharded = ttnn.to_torch(out_tt_sharded)
    assert ttnn.pearson_correlation_coefficient(out_tt_sharded, out_pt) >= 0.99988


@pytest.mark.parametrize(
    "a_shape, b_shape",
    ((torch.Size([1, 5, 7, 32, 96]), torch.Size([1, 5, 7, 32, 96])),),
)
@pytest.mark.parametrize(
    "shard_type, shard_size, core_range",
    ([ttnn.ShardStrategy.HEIGHT, [64, 96], ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (2, 5))})],),
)
def test_binary_sharded_uneven(a_shape, b_shape, shard_type, shard_size, core_range, device):
    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16)(a_shape)
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16)(b_shape)

    shard_config = ttnn.create_sharded_memory_config(
        shard_size,
        core_grid=core_range,
        strategy=shard_type,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    a_tt = ttnn.from_torch(
        a_pt,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=shard_config,
    )
    b_tt = ttnn.from_torch(
        b_pt,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=shard_config,
    )

    out_pt = torch.add(a_pt, b_pt)
    out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=shard_config, use_legacy=False)
    out_tt_sharded = ttnn.to_torch(out_tt_sharded)
    assert ttnn.pearson_correlation_coefficient(out_tt_sharded, out_pt) >= 0.99988


@pytest.mark.parametrize(
    "a_shape, b_shape",
    ((torch.Size([1, 5, 7, 32, 96]), torch.Size([1, 5, 7, 32, 96])),),
)
@pytest.mark.parametrize(
    "shard_type, shard_size, core_range",
    (
        [ttnn.ShardStrategy.WIDTH, [35 * 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 1))})],
        [ttnn.ShardStrategy.BLOCK, [32 * 7, 64], ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (1, 4))})],
    ),
)
def test_binary_sharded_uneven_invalid(a_shape, b_shape, shard_type, shard_size, core_range, device):
    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16)(a_shape)
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16)(b_shape)

    shard_config = ttnn.create_sharded_memory_config(
        shard_size,
        core_grid=core_range,
        strategy=shard_type,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    a_tt = ttnn.from_torch(
        a_pt,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=shard_config,
    )
    b_tt = ttnn.from_torch(
        b_pt,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=shard_config,
    )

    with pytest.raises(RuntimeError) as e:
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=shard_config, use_legacy=False)


@pytest.mark.parametrize("scalar", [-0.25, -16.5, 0.0, 0.05, 1.7, 19.0])
@pytest.mark.parametrize(
    "a_shape, shard_type, shard_size, core_range",
    (
        [
            torch.Size([1, 4 * 32]),
            ttnn.ShardStrategy.HEIGHT,
            [32, 4 * 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))}),
        ],
        [
            torch.Size([1, 4 * 32]),
            ttnn.ShardStrategy.WIDTH,
            [32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),
        ],
        [
            torch.Size([1, 4 * 32]),
            ttnn.ShardStrategy.BLOCK,
            [32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 0))}),
        ],
    ),
)
def test_binary_sharded_scalar(scalar, a_shape, shard_type, shard_size, core_range, device):
    torch.manual_seed(0)
    a_sharded_config = ttnn.create_sharded_memory_config(
        shard_size,
        core_grid=core_range,
        strategy=shard_type,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16)(a_shape)
    a_tt = ttnn.from_torch(
        a_pt,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=a_sharded_config,
    )

    out_pt = torch.add(a_pt, scalar)
    out_tt_sharded = ttnn.add(a_tt, scalar, memory_config=a_sharded_config, use_legacy=False)
    out_tt_sharded = ttnn.to_torch(out_tt_sharded)
    torch.testing.assert_close(out_tt_sharded, out_pt)

    out_tt_interleaved = ttnn.add(a_tt, scalar, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=False)
    out_tt_interleaved = ttnn.to_torch(out_tt_interleaved)
    torch.testing.assert_close(out_tt_interleaved, out_pt)


@pytest.mark.parametrize("scalar", [-0.25])
@pytest.mark.parametrize(
    "a_shape, shard_type, shard_size, core_range",
    (
        # only support HEIGHT
        [
            torch.Size([1, 4 * 32]),
            ttnn.ShardStrategy.WIDTH,
            [1, 4 * 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))}),
        ],
        [
            torch.Size([1, 4 * 32]),
            ttnn.ShardStrategy.BLOCK,
            [1, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 0))}),
        ],
        [
            torch.Size([1, 31]),
            ttnn.ShardStrategy.HEIGHT,
            [1, 31],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))}),
        ],
    ),
)
def test_binary_sharded_scalar_invalid_row_major(scalar, a_shape, shard_type, shard_size, core_range, device):
    torch.manual_seed(0)
    a_sharded_config = ttnn.create_sharded_memory_config(
        shard_size,
        core_grid=core_range,
        strategy=shard_type,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16)(a_shape)

    with pytest.raises(RuntimeError) as e:
        a_tt = ttnn.from_torch(
            a_pt,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=a_sharded_config,
        )

        tt_out = ttnn.add(a_tt, scalar, memory_config=a_sharded_config, use_legacy=False)


@pytest.mark.parametrize("scalar", [-0.25, -16.5, 0.0, 0.05, 1.7, 19.0])
@pytest.mark.parametrize(
    "a_shape, shard_type, shard_size, core_range",
    (
        [
            torch.Size([1, 4 * 32]),
            ttnn.ShardStrategy.WIDTH,
            [32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 1)), ttnn.CoreRange((0, 2), (0, 3))}),
        ],
    ),
)
def test_binary_sharded_scalar_col_major(scalar, a_shape, shard_type, shard_size, core_range, device):
    torch.manual_seed(0)
    a_sharded_config = ttnn.create_sharded_memory_config(
        shard_size,
        core_grid=core_range,
        strategy=shard_type,
        orientation=ttnn.ShardOrientation.COL_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16)(a_shape)

    with pytest.raises(RuntimeError) as e:
        a_tt = ttnn.from_torch(
            a_pt,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=a_sharded_config,
        )

        tt_out = ttnn.add(a_tt, scalar, memory_config=a_sharded_config, use_legacy=False)


@pytest.mark.parametrize(
    "a_shape, b_shape, a_shard_size, b_shard_size, core_range",
    (
        [
            torch.Size([5, 6, 3200, 2]),
            torch.Size([5, 6, 3200, 1]),
            [3200, 32],
            [3200, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (4, 5))}),
        ],
        [
            torch.Size([5, 6, 3200, 33]),
            torch.Size([5, 6, 3200, 1]),
            [3200, 64],
            [3200, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (4, 5))}),
        ],
        [
            torch.Size([5, 6, 1, 1]),
            torch.Size([5, 6, 1, 1]),
            [32, 32],
            [32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (4, 5))}),
        ],
        [
            torch.Size([5, 6, 2, 2]),
            torch.Size([5, 6, 2, 1]),
            [32, 32],
            [32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (4, 5))}),
        ],
    ),
)
def test_binary_sharded_bcast_w_size(a_shape, b_shape, a_shard_size, b_shard_size, core_range, device):
    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16)(a_shape)
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16)(b_shape)

    a_shard_config = ttnn.create_sharded_memory_config(
        a_shard_size,
        core_grid=core_range,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_shard_config = ttnn.create_sharded_memory_config(
        b_shard_size,
        core_grid=core_range,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    a_tt = ttnn.from_torch(
        a_pt,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=a_shard_config,
    )
    b_tt = ttnn.from_torch(
        b_pt,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=b_shard_config,
    )

    out_pt = torch.add(a_pt, b_pt)
    out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=a_shard_config, use_legacy=False)
    out_tt_sharded = ttnn.to_torch(out_tt_sharded)
    torch.testing.assert_close(out_tt_sharded, out_pt)


@pytest.mark.parametrize(
    "a_shape, b_shape, a_strategy, b_strategy, a_shard_size, b_shard_size, a_core_range, b_core_range",
    (
        [
            torch.Size([5, 7, 2 * 32, 32]),
            torch.Size([5, 7, 2 * 32, 32]),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardStrategy.HEIGHT,
            [10 * 32, 32],
            [14 * 32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 4))}),
        ],
    ),
)
def test_binary_sharded_invalid_spec(
    a_shape, b_shape, a_strategy, b_strategy, a_shard_size, b_shard_size, a_core_range, b_core_range, device
):
    a_sharded_config = ttnn.create_sharded_memory_config(
        a_shard_size,
        core_grid=a_core_range,
        strategy=a_strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        b_shard_size,
        core_grid=b_core_range,
        strategy=b_strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    a_pt, a_tt = rand_bf16_gen(a_shape, device, memory_config=a_sharded_config)
    b_pt, b_tt = rand_bf16_gen(b_shape, device, memory_config=b_sharded_config)

    with pytest.raises(RuntimeError):
        _ = ttnn.add(a_tt, b_tt, memory_config=a_sharded_config, use_legacy=False)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
@pytest.mark.parametrize(
    "a_shape, b_shape, shard_type, shard_size, core_range",
    (
        [
            torch.Size([64, 33]),
            torch.Size([64, 33]),
            ttnn.ShardStrategy.HEIGHT,
            [32, 33],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 1))}),
        ],
        [
            torch.Size([64, 33]),
            torch.Size([64, 33]),
            ttnn.ShardStrategy.HEIGHT,
            [32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 1))}),
        ],
        [
            torch.Size([64, 4 * 32]),
            torch.Size([64, 4 * 32]),
            ttnn.ShardStrategy.WIDTH,
            [64, 4 * 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))}),
        ],
        [
            torch.Size([64, 4 * 32]),
            torch.Size([64, 4 * 32]),
            ttnn.ShardStrategy.BLOCK,
            [32, 4 * 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 1))}),
        ],
    ),
)
def test_binary_sharded_invalid_row_major_layout(
    a_shape, b_shape, shard_type, shard_size, core_range, dtype_pt, dtype_tt, device
):
    torch.manual_seed(0)
    a_sharded_config = ttnn.create_sharded_memory_config(
        shard_size,
        core_grid=core_range,
        strategy=shard_type,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        shard_size,
        core_grid=core_range,
        strategy=shard_type,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

    with pytest.raises(RuntimeError):
        a_tt = ttnn.from_torch(
            a_pt,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=a_sharded_config,
        )

        b_tt = ttnn.from_torch(
            b_pt,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=b_sharded_config,
        )

        _ = ttnn.add(a_tt, b_tt, memory_config=a_sharded_config, use_legacy=False)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    (
        [torch.bfloat16, ttnn.bfloat16],
        [torch.int32, ttnn.int32],
    ),
)
@pytest.mark.parametrize(
    "a_shape, b_shape, shard_type, shard_size, core_range",
    (
        [
            torch.Size([64, 64]),
            torch.Size([64, 64]),
            ttnn.ShardStrategy.HEIGHT,
            [32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 1))}),
        ],
        [
            torch.Size([64, 32]),
            torch.Size([64, 32]),
            ttnn.ShardStrategy.HEIGHT,
            [32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 1))}),
        ],
    ),
)
def test_binary_sharded_row_major_layout(
    dtype_pt, dtype_tt, a_shape, b_shape, shard_type, shard_size, core_range, device
):
    torch.manual_seed(0)
    a_sharded_config = ttnn.create_sharded_memory_config(
        shard_size,
        core_grid=core_range,
        strategy=shard_type,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        shard_size,
        core_grid=core_range,
        strategy=shard_type,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)
    a_tt = ttnn.from_torch(
        a_pt,
        dtype=dtype_tt,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=a_sharded_config,
    )

    b_tt = ttnn.from_torch(
        b_pt,
        dtype=dtype_tt,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=b_sharded_config,
    )
    out_pt = torch.add(a_pt, b_pt)
    out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=a_sharded_config, use_legacy=False)
    out_tt_sharded = ttnn.to_torch(out_tt_sharded)
    torch.testing.assert_close(out_tt_sharded, out_pt)


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        [[1, 1, 7, 7], [1, 1, 7, 7]],
        [[1, 1, 71, 71], [1, 1, 71, 71]],
        [[7, 71, 71, 7], [7, 71, 71, 7]],
        [[1, 1, 7, 7], [1, 71, 7, 7]],
        [[1, 71, 7, 7], [1, 1, 7, 7]],
        [[71, 1, 7, 7], [1, 1, 7, 7]],
        [[1, 1, 7, 7], [71, 1, 7, 7]],
        [[1, 1, 7, 7], [7, 71, 7, 7]],
        [[7, 71, 7, 7], [1, 1, 7, 7]],
        [[920, 1, 256], [256]],
        [[256], [920, 1, 256]],
    ],
)
def test_binary_subtile_no_bcast(a_shape, b_shape, device):
    torch.manual_seed(0)

    torch_input_tensor_a, input_tensor_a = rand_bf16_gen(a_shape, device)
    torch_input_tensor_b, input_tensor_b = rand_bf16_gen(b_shape, device)

    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=False)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        [[1, 1, 320, 320], [1, 1, 1, 320]],
        [[1, 1, 1, 320], [1, 1, 320, 320]],
        [[1, 4, 320, 320], [1, 1, 1, 320]],
        [[1, 1, 1, 320], [1, 4, 320, 320]],
        [[4, 1, 320, 320], [1, 1, 1, 320]],
        [[1, 1, 1, 320], [4, 1, 320, 320]],
        [[4, 4, 320, 320], [1, 1, 1, 320]],
        [[1, 1, 1, 320], [4, 4, 320, 320]],
        [[8192, 8192], [1, 8192]],
        [[1, 8192], [8192, 8192]],
    ],
)
def test_binary_subtile_row_bcast(a_shape, b_shape, device):
    torch.manual_seed(0)

    torch_input_tensor_a, input_tensor_a = rand_bf16_gen(a_shape, device)
    torch_input_tensor_b, input_tensor_b = rand_bf16_gen(b_shape, device)

    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=False)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988


profile_a_b_shape_pairs = [
    # [[8192, 8192], [8192, 8192]],
    # [[1, 8192], [8192, 8192]],
    # [[8192, 8192], [1, 8192]],
    # [[8192, 1], [8192, 8192]],
    # [[8192, 8192], [8192, 1]],
    # [[1, 8192], [8192, 1]],
    # [[8192, 1], [1, 8192]],
    # [[1, 1], [8192, 8192]],
    [[8192, 8192], [1, 1]],
]


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ((torch.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "memory_config_input",
    [ttnn.DRAM_MEMORY_CONFIG],
)
@pytest.mark.parametrize("a_and_b_shape", profile_a_b_shape_pairs)
def test_binary_bcast_profile(device, dtype_pt, dtype_tt, a_and_b_shape, memory_config_input):
    device.enable_program_cache()
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
        output = ttnn.add(input_tensor_a, input_tensor_b, memory_config=memory_config_input, use_legacy=False)
        output = ttnn.to_torch(output)

        assert (
            output.shape == torch_result.shape
        ), f"Output shape {output.shape} does not match torch shape {torch_result.shape}"

        torch.testing.assert_close(torch_result, output)
        ttnn.synchronize_device(device)


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        [[1, 1, 320, 1], [1, 1, 320, 320]],
        # a bcast, b no bcast
        [[1, 1, 320, 1], [1, 4, 320, 320]],
        [[1, 1, 320, 1], [4, 1, 320, 320]],
        [[1, 1, 320, 1], [4, 4, 320, 320]],
        [[4, 1, 320, 1], [1, 1, 320, 320]],
        [[1, 4, 320, 1], [1, 1, 320, 320]],
        [[4, 4, 320, 1], [1, 1, 320, 320]],
        [[1, 4, 320, 1], [4, 1, 320, 320]],
        [[4, 1, 320, 1], [1, 4, 320, 320]],
        [[4, 4, 320, 1], [4, 4, 320, 320]],
        # a no bcast, b bcast
        [[1, 1, 320, 320], [1, 4, 320, 1]],
        [[1, 1, 320, 320], [4, 1, 320, 1]],
        [[1, 1, 320, 320], [4, 4, 320, 1]],
        [[4, 1, 320, 320], [1, 1, 320, 1]],
        [[1, 4, 320, 320], [1, 1, 320, 1]],
        [[4, 4, 320, 320], [1, 1, 320, 1]],
        [[1, 4, 320, 320], [4, 1, 320, 1]],
        [[4, 1, 320, 320], [1, 4, 320, 1]],
        [[4, 4, 320, 320], [4, 4, 320, 1]],
    ],
)
def test_binary_subtile_col_bcast(a_shape, b_shape, device):
    torch.manual_seed(0)

    torch_input_tensor_a, input_tensor_a = rand_bf16_gen(a_shape, device)
    torch_input_tensor_b, input_tensor_b = rand_bf16_gen(b_shape, device)

    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=False)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        [[1, 1, 1, 1], [1, 1, 320, 320]],
        [[1, 1, 320, 320], [1, 1, 1, 1]],
        # a scalar, b no bcast
        [[1, 1, 1, 1], [1, 4, 320, 320]],
        [[1, 1, 1, 1], [4, 1, 320, 320]],
        [[1, 1, 1, 1], [4, 4, 320, 320]],
        [[1, 4, 1, 1], [1, 1, 320, 320]],
        [[4, 1, 1, 1], [1, 1, 320, 320]],
        [[4, 4, 1, 1], [1, 1, 320, 320]],
        # # a no bast, b scalar
        [[1, 1, 320, 320], [1, 4, 1, 1]],
        [[1, 1, 320, 320], [4, 1, 1, 1]],
        [[1, 1, 320, 320], [4, 4, 1, 1]],
        [[1, 4, 320, 320], [1, 1, 1, 1]],
        [[4, 1, 320, 320], [1, 1, 1, 1]],
        [[4, 4, 320, 320], [1, 1, 1, 1]],
    ],
)
def test_binary_subtile_scalar_bcast(a_shape, b_shape, device):
    torch.manual_seed(0)

    torch_input_tensor_a, input_tensor_a = rand_bf16_gen(a_shape, device)
    torch_input_tensor_b, input_tensor_b = rand_bf16_gen(b_shape, device)

    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=False)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        [[1, 1, 320, 1], [1, 1, 1, 320]],
        # a col, b row
        [[1, 1, 320, 1], [1, 4, 1, 320]],
        [[1, 1, 320, 1], [4, 1, 1, 320]],
        [[1, 1, 320, 1], [4, 4, 1, 320]],
        [[4, 1, 320, 1], [1, 1, 1, 320]],
        [[1, 4, 320, 1], [1, 1, 1, 320]],
        [[4, 4, 320, 1], [1, 1, 1, 320]],
        [[1, 4, 320, 1], [4, 1, 1, 320]],
        [[4, 1, 320, 1], [1, 4, 1, 320]],
        [[4, 4, 320, 1], [4, 4, 1, 320]],
        # a row, b col
        [[1, 1, 1, 320], [1, 4, 320, 1]],
        [[1, 1, 1, 320], [4, 1, 320, 1]],
        [[1, 1, 1, 320], [4, 4, 320, 1]],
        [[4, 1, 1, 320], [1, 1, 320, 1]],
        [[1, 4, 1, 320], [1, 1, 320, 1]],
        [[4, 4, 1, 320], [1, 1, 320, 1]],
        [[1, 4, 1, 320], [4, 1, 320, 1]],
        [[4, 1, 1, 320], [1, 4, 320, 1]],
        [[4, 4, 1, 320], [4, 4, 320, 1]],
    ],
)
def test_binary_subtile_row_b_col_a_bcast(a_shape, b_shape, device):
    torch.manual_seed(0)

    torch_input_tensor_a, input_tensor_a = rand_bf16_gen(a_shape, device)
    torch_input_tensor_b, input_tensor_b = rand_bf16_gen(b_shape, device)

    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=False)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988


@pytest.mark.parametrize(
    "input_shape_a",
    [
        torch.Size([32, 32]),
        torch.Size([64, 64]),
        torch.Size([1, 1, 32, 32]),
        torch.Size([1, 1, 320, 384]),
        torch.Size([1, 3, 320, 384]),
    ],
)
@pytest.mark.parametrize("bcast_dim", [ttnn.BcastOpDim.H, ttnn.BcastOpDim.W, ttnn.BcastOpDim.HW])
@pytest.mark.parametrize("math_op", [ttnn.BcastOpMath.ADD, ttnn.BcastOpMath.SUB, ttnn.BcastOpMath.MUL])
def test_bcast(input_shape_a, device, bcast_dim, math_op):
    input_shape_b = list(input_shape_a)

    if bcast_dim == ttnn.BcastOpDim.H or bcast_dim == ttnn.BcastOpDim.HW:
        input_shape_b[-2] = 1

    if bcast_dim == ttnn.BcastOpDim.W or bcast_dim == ttnn.BcastOpDim.HW:
        input_shape_b[-1] = 1
    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.bfloat16), ttnn.bfloat16)(
        input_shape_a
    )
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=-90, high=90, dtype=torch.bfloat16), ttnn.bfloat16)(
        input_shape_b
    )

    a_tt = ttnn.from_torch(
        a_pt,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    b_tt = ttnn.from_torch(
        b_pt,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    output_tensor = ttnn.bcast(a_tt, b_tt, math_op, bcast_dim)

    golden_function = ttnn.get_golden_function(ttnn.bcast)
    golden_tensor = golden_function(a_pt, b_pt, math_op, bcast_dim)

    comp_pass = compare_pcc([output_tensor], [golden_tensor], 0.9999)
    assert comp_pass


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
def test_binary_sharded_decoder_program_cache(dtype_pt, dtype_tt, device, use_program_cache):
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


def test_yolov8_add_small(device):
    tor_a = torch.tensor(
        [
            [
                [
                    [
                        1.843750000000000,
                        2.546875000000000,
                        2.968750000000000,
                        2.109375000000000,
                        2.156250000000000,
                        2.718750000000000,
                        2.000000000000000,
                        1.312500000000000,
                        1.976562500000000,
                        2.187500000000000,
                        1.468750000000000,
                        1.578125000000000,
                        1.851562500000000,
                        1.812500000000000,
                        1.648437500000000,
                        1.687500000000000,
                    ]
                ],
                [
                    [
                        1.484375000000000,
                        1.398437500000000,
                        1.570312500000000,
                        1.757812500000000,
                        2.343750000000000,
                        1.851562500000000,
                        2.359375000000000,
                        1.976562500000000,
                        1.679687500000000,
                        2.062500000000000,
                        1.000000000000000,
                        1.093750000000000,
                        0.910156250000000,
                        1.789062500000000,
                        1.882812500000000,
                        2.218750000000000,
                    ]
                ],
                [
                    [
                        36.750000000000000,
                        36.750000000000000,
                        39.250000000000000,
                        40.750000000000000,
                        41.000000000000000,
                        37.000000000000000,
                        37.250000000000000,
                        35.250000000000000,
                        36.750000000000000,
                        34.750000000000000,
                        37.500000000000000,
                        34.250000000000000,
                        37.500000000000000,
                        37.500000000000000,
                        36.500000000000000,
                        37.000000000000000,
                    ]
                ],
                [
                    [
                        0.656250000000000,
                        0.562500000000000,
                        0.476562500000000,
                        0.460937500000000,
                        0.730468750000000,
                        0.578125000000000,
                        0.679687500000000,
                        0.718750000000000,
                        0.988281250000000,
                        0.761718750000000,
                        0.765625000000000,
                        0.730468750000000,
                        0.574218750000000,
                        0.671875000000000,
                        0.777343750000000,
                        0.589843750000000,
                    ]
                ],
            ]
        ],
        dtype=torch.bfloat16,
    )

    tor_b = torch.tensor(
        [[[[-1.541992187500000]], [[-1.537109375000000]], [[-0.816894531250000]], [[-0.034881591796875]]]],
        dtype=torch.float32,
    )

    print("tor_a", tor_a.shape)
    print("tor_b", tor_b.shape)
    tor_res = torch.add(tor_a, tor_b)
    mem = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type=ttnn.BufferType.L1, shard_spec=None
    )

    tt_a = ttnn.from_torch(tor_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    tt_b = ttnn.from_torch(tor_b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    #  print("with typecast")
    #  tt_a = ttnn.typecast(tt_a, dtype=ttnn.float32)
    print("add")
    result = ttnn.add(tt_a, tt_b, use_legacy=False)

    tt_res = ttnn.to_torch(result)
    #  print("tt_res", tt_res)

    pcc = compare_pcc([result], [tor_res], 0.999)
    assert pcc


def rand_gen(shape, device, *, dtype, tt_dtype, min=0, max=1, memory_config):
    pt = torch.rand(shape, dtype=dtype) * (max - min) + min
    tt = ttnn.from_torch(pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, dtype=tt_dtype)
    return pt, tt


@pytest.mark.parametrize(
    "dtype_pt_a, dtype_tt_a, dtype_pt_b, dtype_tt_b",
    (
        [torch.bfloat16, ttnn.bfloat16, torch.float32, ttnn.float32],
        [torch.float32, ttnn.float32, torch.bfloat16, ttnn.bfloat16],
    ),
)
def test_binary_mixed_add(dtype_pt_a, dtype_tt_a, dtype_pt_b, dtype_tt_b, device):
    torch.manual_seed(0)
    a_shape = torch.Size([1, 4, 2, 160])
    b_shape = torch.Size([1, 4, 1, 160])
    mem = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type=ttnn.BufferType.L1, shard_spec=None
    )
    a_pt, a_tt = rand_gen(a_shape, device, dtype=dtype_pt_a, tt_dtype=dtype_tt_a, memory_config=mem)
    b_pt, b_tt = rand_gen(b_shape, device, dtype=dtype_pt_b, tt_dtype=dtype_tt_b, memory_config=mem)

    golden_fn = ttnn.get_golden_function(ttnn.add)

    out_tt = ttnn.add(a_tt, b_tt, use_legacy=False)
    out_pt = golden_fn(a_pt, b_pt)

    assert compare_pcc([out_tt], [out_pt])
