# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    compare_pcc,
)
from models.utility_functions import skip_for_grayskull, torch_random
from itertools import product as parameters
from functools import partial
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt


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
    "rsub",
    "mul",
    "div",
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
        parameters({"eq", "ne"}, {square_lhs, sin_rhs, exp_floor_lhs_exp_rhs, log_lhs_sqrt_abs_post}),
        parameters({"logaddexp", "logaddexp2"}, {floor_lhs_ceil_rhs_cos_post}),
        parameters({"gte", "lt", "lte"}, {exp_floor_lhs_exp_rhs, log_lhs_sqrt_abs_post}),
        parameters({"logical_and", "logical_or", "logical_xor", "bias_gelu"}, {log_lhs_sqrt_abs_post}),
        parameters({"div"}, {exp_post, tanh_post, exp2_post, expm1_post, i0_post, tan_post}),
        parameters({"sub"}, {log_post, log2_post, log10_post}),
        parameters({"ldexp"}, {erfinv_post, tan_post, floor_post, ceil_post}),
        parameters({"squared_difference"}, {erfinv_post, i0_post}),
        parameters({"add"}, {tan_post, tanh_post}),
        {("mul", log_lhs_sqrt_abs_post)},
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

    out_tt = ttnn_op(a_tt, b_tt, lhs_activations=lhs, rhs_activations=rhs, post_activations=post)

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
@pytest.mark.parametrize("ttnn_fn", ("add", "sub", "mul", "div"))
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
    ttnn_op = getattr(ttnn.experimental, ttnn_fn)
    post = [(getattr(ttnn.UnaryOpType, op), param) for op, param in post_activations]
    golden_post = ((lambda x: activation_with_param_fns[op](x, param)) for op, param in post_activations)
    # make 0 exclusive for rhs of div
    min, max = (1, 0) if ttnn_fn == "div" else (0, 1)

    a_pt, a_tt = rand_bf16_gen(a_shape, device)
    b_pt, b_tt = rand_bf16_gen(b_shape, device, min=min, max=max)

    out_tt = ttnn_op(a_tt, b_tt, post_activations=post)

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
    ttnn_c = ttnn.experimental.add(ttnn_a, ttnn_b)
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
        tt_c = ttnn.experimental.add(tt_a, tt_b)


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
        out_tt_interleaved = ttnn.experimental.add(a_tt, b_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out_tt_interleaved = ttnn.to_torch(out_tt_interleaved)
        assert ttnn.pearson_correlation_coefficient(out_tt_interleaved, out_pt) >= 0.99988

        out_tt_sharded = ttnn.experimental.add(a_tt, b_tt, memory_config=sharded_config)
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

    out_tt_interleaved = ttnn.experimental.add(a_tt, b_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out_tt_interleaved = ttnn.to_torch(out_tt_interleaved)
    assert ttnn.pearson_correlation_coefficient(out_tt_interleaved, out_pt) >= 0.99988

    out_tt_sharded = ttnn.experimental.add(a_tt, b_tt, memory_config=sharded_config)
    out_tt_sharded = ttnn.to_torch(out_tt_sharded)
    assert ttnn.pearson_correlation_coefficient(out_tt_sharded, out_pt) >= 0.99988


@skip_for_grayskull("Requires wormhole_b0 to run")
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
        ttnn.experimental.add,
        ttnn.experimental.sub,
        ttnn.experimental.mul,
        ttnn.experimental.div,
        ttnn.experimental.rsub,
        ttnn.experimental.eq,
        ttnn.experimental.ne,
        ttnn.experimental.gt,
        ttnn.experimental.gte,
        ttnn.experimental.lt,
        ttnn.experimental.lte,
        ttnn.experimental.logical_or,
        ttnn.experimental.logical_xor,
        ttnn.experimental.logical_and,
        ttnn.experimental.ldexp,
        ttnn.experimental.logaddexp,
        ttnn.experimental.logaddexp2,
        ttnn.experimental.squared_difference,
        ttnn.experimental.bias_gelu,
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
    out_tt = ttnn_fn(a_tt, b_tt, queue_id=cq_id)
    tt_out = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn_fn)
    out_pt = golden_fn(a_pt, b_pt)
    status = ttnn.pearson_correlation_coefficient(out_pt, tt_out)
    assert status >= 0.999


@skip_for_grayskull("Requires wormhole_b0 to run")
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
        ttnn.experimental.add,
        ttnn.experimental.sub,
        ttnn.experimental.mul,
        ttnn.experimental.div,
        ttnn.experimental.rsub,
        ttnn.experimental.eq,
        ttnn.experimental.ne,
        ttnn.experimental.gt,
        ttnn.experimental.gte,
        ttnn.experimental.lt,
        ttnn.experimental.lte,
        ttnn.experimental.logical_or,
        ttnn.experimental.logical_xor,
        ttnn.experimental.logical_and,
        ttnn.experimental.ldexp,
        ttnn.experimental.logaddexp,
        ttnn.experimental.logaddexp2,
        ttnn.experimental.squared_difference,
        ttnn.experimental.bias_gelu,
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
    ttnn_fn(a_tt, b_tt, queue_id=cq_id, output_tensor=out_tt)
    tt_out = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn_fn)
    out_pt = golden_fn(a_pt, b_pt)
    status = ttnn.pearson_correlation_coefficient(out_pt, tt_out)
    assert status >= 0.999


@skip_for_grayskull("Requires wormhole_b0 to run")
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
        ttnn.experimental.bitwise_and,
        ttnn.experimental.bitwise_or,
        ttnn.experimental.bitwise_xor,
        ttnn.experimental.bitwise_left_shift,
        ttnn.experimental.bitwise_right_shift,
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
    out_tt = ttnn_fn(a_tt, b_tt, queue_id=cq_id)
    tt_out = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn_fn)
    out_pt = golden_fn(a_pt, b_pt)

    status = ttnn.pearson_correlation_coefficient(out_pt, tt_out)
    assert status >= 0.999


@skip_for_grayskull("Requires wormhole_b0 to run")
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
        ttnn.experimental.bitwise_and,
        ttnn.experimental.bitwise_or,
        ttnn.experimental.bitwise_xor,
        ttnn.experimental.bitwise_left_shift,
        ttnn.experimental.bitwise_right_shift,
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
    ttnn_fn(a_tt, b_tt, queue_id=cq_id, output_tensor=out_tt)
    tt_out = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn_fn)
    out_pt = golden_fn(a_pt, b_pt)

    status = ttnn.pearson_correlation_coefficient(out_pt, tt_out)
    assert status >= 0.999


binary_inplace_fns = {
    "add_",
    "sub_",
    "mul_",
    "div_",
    "rsub_",
    "gt_",
    "lt_",
    "lte_",
    "gte_",
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
        parameters({"lt_", "gte_"}, {exp_floor_lhs_exp_rhs}),
        parameters({"lte_"}, {sin_rhs, log_lhs_sqrt_abs_post}),
        parameters({"bias_gelu_"}, {log_lhs_sqrt_abs_post}),
        parameters({"mul_"}, {log_lhs_sqrt_abs_post}),
    ),
)
@skip_for_grayskull("Possible accuracy issues with grayskull")
def test_inplace_binary_ops_with_tensor(a_shape, b_shape, ttnn_fn, activations, device):
    torch.manual_seed(0)

    ttnn_op = getattr(ttnn.experimental, ttnn_fn)
    lhs, rhs, post = ([getattr(ttnn.UnaryOpType, op) for op in ops] for ops in activations)
    golden_lhs, golden_rhs, golden_post = ((activation_fns[op] for op in ops) for ops in activations)
    min, max = (1, 0) if ttnn_fn == "div_" else (0, 1)

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
        lhs_activations=lhs,
        rhs_activations=rhs,
        post_activations=post,
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
            *parameters({"gt_", "lte_", "gte_", "lt_"}, {sin_rhs, square_lhs}),
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
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat4_b, ttnn.bfloat8_b])
@pytest.mark.parametrize("ttnn_fn", ["add_", "sub_", "mul_"])
@skip_for_grayskull("Possible accuracy issues with grayskull")
def test_inplace_bf4b_bf8b(a_shape, b_shape, input_dtype, ttnn_fn, device):
    torch.manual_seed(0)

    torch_input_tensor_a, input_tensor_a = rand_bf16_gen(a_shape, device, min=-1e3, max=1e3)
    torch_input_tensor_b, input_tensor_b = rand_bf16_gen(b_shape, device, min=-1e3, max=1e3)
    ttnn_op = getattr(ttnn.experimental, ttnn_fn)
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

    ttnn_op(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(input_tensor_a)
    assert output_tensor.shape == torch_output_tensor.shape

    def compare(output_tensor, torch_output_tensor, ttnn_fn, input_dtype):
        imprecise_cases = {
            "add_": {ttnn.bfloat4_b},
            "sub_": {ttnn.bfloat4_b},
            "mul_": {ttnn.bfloat4_b},
        }
        if ttnn_fn in imprecise_cases and input_dtype in imprecise_cases[ttnn_fn]:
            return ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.97
        else:
            return ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.999

    assert compare(output_tensor, torch_output_tensor, ttnn_fn, input_dtype)


@skip_for_grayskull("Requires wormhole_b0 to run")
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
    ttnn_op = getattr(ttnn.experimental, ttnn_fn)
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
    ttnn_op(input_tensor_a, input_tensor_b, queue_id=cq_id)
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
    ttnn_op = getattr(ttnn.experimental, ttnn_fn)

    _, input_tensor_a = rand_bf16_gen(a_shape, device)
    _, input_tensor_b = rand_bf16_gen(b_shape, device)

    with pytest.raises(RuntimeError):
        cq_id = 0
        ttnn_op(input_tensor_a, input_tensor_b, queue_id=cq_id)


@pytest.mark.parametrize(
    "ttnn_fn",
    [
        "add_",
        "sub_",
        "mul_",
        "div_",
        "rsub_",
        "gt_",
        "lt_",
        "lte_",
        "gte_",
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
@skip_for_grayskull("Possible accuracy issues with grayskull")
@pytest.mark.parametrize("scalar", [-0.25, -16.5, 0.0, 0.05, 1.7, 19.0])
def test_inplace_binary_with_scalar(a_shape, scalar, ttnn_fn, device):
    torch.manual_seed(0)

    ttnn_op = getattr(ttnn.experimental, ttnn_fn)
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

    ttnn_op(input_tensor_a, scalar)
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
    ttnn_op = getattr(ttnn.experimental, ttnn_fn)

    _, input_tensor_a = rand_bf16_gen(a_shape, device)
    _, input_tensor_b = rand_bf16_gen(b_shape, device)
    _, out_tt = rand_bf16_gen(out_shape, device)

    with pytest.raises(
        RuntimeError, match=r"Shape of Output tensor.+ provided does not match the broadcasted output shape .+"
    ):
        cq_id = 0
        ttnn_op(input_tensor_a, input_tensor_b, queue_id=cq_id, output_tensor=out_tt)


@skip_for_grayskull()
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
        out_tt_sharded = ttnn.experimental.add(a_tt, b_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        torch.testing.assert_close(out_tt_sharded, out_pt)

        out_tt_sharded = ttnn.experimental.add(a_tt, b_tt, memory_config=a_sharded_config)
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
        out_tt_sharded = ttnn.experimental.add(a_tt, b_tt, memory_config=a_sharded_config)


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
    out_tt_sharded = ttnn.experimental.add(a_tt, b_tt, memory_config=shard_config)
    out_tt_sharded = ttnn.to_torch(out_tt_sharded)
    assert ttnn.pearson_correlation_coefficient(out_tt_sharded, out_pt) >= 0.99988


@pytest.mark.parametrize(
    "ttnn_fn",
    [
        ttnn.experimental.add,
        ttnn.experimental.sub,
        ttnn.experimental.mul,
        # ttnn.experimental.div,
        # ttnn.experimental.rsub,
        ttnn.experimental.eq,
        ttnn.experimental.ne,
        ttnn.experimental.gt,
        ttnn.experimental.gte,
        ttnn.experimental.lt,
        # ttnn.experimental.lte,
        ttnn.experimental.logical_or,
        # ttnn.experimental.logical_xor,
        ttnn.experimental.logical_and,
        # ttnn.experimental.ldexp,
        # ttnn.experimental.logaddexp,
        # ttnn.experimental.logaddexp2,
        # ttnn.experimental.squared_difference,
        # ttnn.experimental.bias_gelu,
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

        out_tt_sharded = ttnn_fn(a_tt, b_tt, memory_config=shard_config)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert ttnn.pearson_correlation_coefficient(out_tt_sharded, out_pt) >= 0.99988

        out_tt_interleaved = ttnn_fn(a_tt, b_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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
    out_tt_sharded = ttnn.experimental.add(a_tt, b_tt, memory_config=shard_config)
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
    out_tt_sharded = ttnn.experimental.add(a_tt, b_tt, memory_config=shard_config)
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
        out_tt_sharded = ttnn.experimental.add(a_tt, b_tt, memory_config=shard_config)


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
    out_tt_sharded = ttnn.experimental.add(a_tt, scalar, memory_config=a_sharded_config)
    out_tt_sharded = ttnn.to_torch(out_tt_sharded)
    torch.testing.assert_close(out_tt_sharded, out_pt)

    out_tt_interleaved = ttnn.experimental.add(a_tt, scalar, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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
            torch.Size([32, 4 * 32]),
            ttnn.ShardStrategy.HEIGHT,
            [32, 4 * 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))}),
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

        tt_out = ttnn.experimental.add(a_tt, scalar, memory_config=a_sharded_config)


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

        tt_out = ttnn.experimental.add(a_tt, scalar, memory_config=a_sharded_config)


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
    out_tt_sharded = ttnn.experimental.add(a_tt, b_tt, memory_config=a_shard_config)
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
        _ = ttnn.experimental.add(a_tt, b_tt, memory_config=a_sharded_config)


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

        _ = ttnn.experimental.add(a_tt, b_tt, memory_config=a_sharded_config)


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
def test_binary_sharded_row_major_layout(a_shape, b_shape, shard_type, shard_size, core_range, device):
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
    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16)(a_shape)
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16)(b_shape)
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
    out_pt = torch.add(a_pt, b_pt)
    out_tt_sharded = ttnn.experimental.add(a_tt, b_tt, memory_config=a_sharded_config)
    out_tt_sharded = ttnn.to_torch(out_tt_sharded)
    torch.testing.assert_close(out_tt_sharded, out_pt)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    (
        [torch.int32, ttnn.int32],
        # currently float32 fails
        # [torch.float32, ttnn.float32],
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
def test_binary_sharded_invalid_row_major_dtype(
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
    with pytest.raises(RuntimeError):
        _ = ttnn.experimental.add(a_tt, b_tt, memory_config=a_sharded_config)
