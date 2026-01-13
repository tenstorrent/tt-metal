# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    compare_pcc,
)
from models.common.utility_functions import torch_random
from models.common.utility_functions import divup
from itertools import product as parameters
from functools import partial
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.utils_for_testing import assert_allclose, assert_with_ulp

pytestmark = pytest.mark.use_module_device


def rand_bf16_gen(shape, device, *, min=0, max=1, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    pt = torch.rand(shape, dtype=torch.bfloat16) * (max - min) + min
    tt = ttnn.from_torch(pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)
    return pt, tt


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
        [[2, 4, 12, 64, 64], [12, 1, 1]],
        [[12, 1, 1], [2, 4, 12, 64, 64]],
        [[2, 3, 3, 4, 32], [3, 3, 4, 32]],
        [[5, 2, 3, 3, 4, 32], [5, 1, 3, 3, 4, 32]],
    ],
)
def test_unequal_ranks(a_shape, b_shape, device):
    torch.manual_seed(0)

    torch_input_tensor_a, input_tensor_a = rand_bf16_gen(a_shape, device)
    torch_input_tensor_b, input_tensor_b = rand_bf16_gen(b_shape, device)

    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=None)
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
    ttnn_c = ttnn.add(ttnn_a, ttnn_b, use_legacy=None)
    c = ttnn.to_torch(ttnn_c).reshape((-1))

    assert c.tolist() == c_golden


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        [[3, 4, 8, 6, 32, 64], [1, 1, 8, 6, 32, 64]],
        [[1, 2, 3, 3, 4, 32], [5, 1, 3, 3, 4, 32]],
    ],
)
def test_binary_invalid_rank(device, a_shape, b_shape):
    torch.manual_seed(0)
    pt_a, tt_a = rand_bf16_gen(a_shape, device)
    pt_b, tt_b = rand_bf16_gen(b_shape, device)

    with pytest.raises(RuntimeError):
        tt_c = ttnn.add(tt_a, tt_b, use_legacy=None)


height_sharded_memory_config = ttnn.create_sharded_memory_config(
    # [320, 128], # 7 cores
    # core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
    [160, 128],  # 14 cores
    # [128, 160],
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
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

# width sharding is not good for large and tall (w is small) tensors
# because each core may ends up with a large tensor as well, then out of L1 space
width_sharded_memory_config = ttnn.create_sharded_memory_config(
    # [2240, 64],
    # core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 1))}),
    # [2240, 32],
    [32, 2240],
    # core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),
    # core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((1, 0), (1, 3))}),
    # core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 1)), ttnn.CoreRange((2, 2), (2, 3))}),
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((2, 2), (2, 3)), ttnn.CoreRange((0, 0), (0, 1))}),
    strategy=ttnn.ShardStrategy.WIDTH,
    orientation=ttnn.ShardOrientation.COL_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

block_sharded_memory_config = ttnn.create_sharded_memory_config(
    # [320, 64], # 128 / 64 = 2, core grid is 2x6
    # core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (1, 6))}),
    # following is better, more cores
    # [320, 32],  # 128 / 32 = 4, core grid is 4x7
    # core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 6))}),
    [32, 320],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (6, 3))}),
    # core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((1, 0), (4, 6))}),
    # [160, 32] will not work, because it needs core grid 4x14
    strategy=ttnn.ShardStrategy.BLOCK,
    orientation=ttnn.ShardOrientation.COL_MAJOR,
    use_height_and_width_as_shard_shape=True,
)


@pytest.mark.parametrize(
    "a_shape, b_shape",
    ((torch.Size([5, 7, 64, 128]), torch.Size([5, 7, 64, 128])),),
)
@pytest.mark.parametrize(
    "a_config, b_config, out_config",
    [
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, height_sharded_memory_config],
        [ttnn.DRAM_MEMORY_CONFIG, height_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG],
        [ttnn.DRAM_MEMORY_CONFIG, height_sharded_memory_config, height_sharded_memory_config],
        [height_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        [height_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG, height_sharded_memory_config],
        [height_sharded_memory_config, height_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG],
        [height_sharded_memory_config, height_sharded_memory_config, height_sharded_memory_config],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, width_sharded_memory_config],
        [ttnn.DRAM_MEMORY_CONFIG, width_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG],
        [ttnn.DRAM_MEMORY_CONFIG, width_sharded_memory_config, width_sharded_memory_config],
        [width_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        [width_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG, width_sharded_memory_config],
        [width_sharded_memory_config, width_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG],
        [width_sharded_memory_config, width_sharded_memory_config, width_sharded_memory_config],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, block_sharded_memory_config],
        [ttnn.DRAM_MEMORY_CONFIG, block_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG],
        [ttnn.DRAM_MEMORY_CONFIG, block_sharded_memory_config, block_sharded_memory_config],
        [block_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        [block_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG, block_sharded_memory_config],
        [block_sharded_memory_config, block_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG],
        [block_sharded_memory_config, block_sharded_memory_config, block_sharded_memory_config],
    ],
)
@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    (
        [torch.bfloat16, ttnn.bfloat16],
        # works, but time consuming
        # [torch.float32, ttnn.float32],
        # currently handled by legacy, and it does not work
        # [torch.bfloat16, ttnn.bfloat8_b],
        # [torch.bfloat16, ttnn.bfloat4_b],
    ),
)
def test_binary_sharded_bcast_no_identical(
    a_shape, b_shape, a_config, b_config, out_config, dtype_pt, dtype_tt, device
):
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
    if dtype_tt == ttnn.bfloat4_b:
        assert_with_pcc(ttnn.to_torch(out_tt), out_pt, 0.993)
    else:
        assert_with_pcc(ttnn.to_torch(out_tt), out_pt)

    # no memory config
    out_tt = ttnn.add(a_tt, b_tt, use_legacy=None)
    if dtype_tt == ttnn.bfloat4_b:
        assert_with_pcc(ttnn.to_torch(out_tt), out_pt, 0.993)
    else:
        assert_with_pcc(ttnn.to_torch(out_tt), out_pt)


@pytest.mark.parametrize(
    "a_shape, b_shape",
    (
        (torch.Size([5, 7, 64, 128]), torch.Size([5, 7, 64, 128])),
        (torch.Size([64 * 5 * 7, 127]), torch.Size([64 * 5 * 7, 127])),
    ),
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
def test_binary_sharded_row_major_layout(device, a_shape, b_shape, sharded_core_grid, memory_lay_out):
    torch.manual_seed(0)
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

    out_tt_interleaved = ttnn.add(a_tt, b_tt, use_legacy=None)
    out_tt_interleaved = ttnn.to_torch(out_tt_interleaved)
    assert ttnn.pearson_correlation_coefficient(out_tt_interleaved, out_pt) >= 0.99988

    out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=sharded_config, use_legacy=None)
    out_tt_sharded = ttnn.to_torch(out_tt_sharded)
    assert ttnn.pearson_correlation_coefficient(out_tt_sharded, out_pt) >= 0.99988


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
    "dtype_pt, dtype_tt",
    (
        [torch.bfloat16, ttnn.bfloat16],
        # does not work for binary_ng yet
        # [torch.bfloat16, ttnn.bfloat8_b],
    ),
)
def test_binary_sharded_bcast_w_height(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
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
        [2 * 5 * 32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, a_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, a_sharded_config),
    )

    for a_config, b_config, out_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

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
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=out_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_tt_sharded, out_pt)

        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_tt_sharded, out_pt)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_w_height_c(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([5, 7, 2 * 32, 4 * 32])
    b_shape = torch.Size([5, 1, 2 * 32, 1])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [7 * 2 * 32, 4 * 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 4))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 4))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, a_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, a_sharded_config),
    )

    for a_config, b_config, out_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

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
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=out_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_tt_sharded, out_pt)

        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_tt_sharded, out_pt)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_w_height_n(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([5, 7, 2 * 32, 4 * 32])
    b_shape = torch.Size([1, 7, 2 * 32, 1])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [10 * 32, 4 * 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, a_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, a_sharded_config),
    )

    for a_config, b_config, out_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

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
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=out_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_tt_sharded, out_pt)

        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_tt_sharded, out_pt)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_h_height(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 7, 32 * 2, 4 * 32])
    b_shape = torch.Size([1, 7, 1, 4 * 32])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 32 * 2, 4 * 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [1 * 32, 4 * 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, a_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, a_sharded_config),
    )

    for a_config, b_config, out_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

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
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=out_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_scalar_height(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 7, 2 * 32, 4 * 32])
    b_shape = torch.Size([1, 7, 1, 1])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 32 * 2, 4 * 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [1 * 32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, a_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, a_sharded_config),
    )

    for a_config, b_config, out_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

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
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=out_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_hw_mixed_height(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 7, 2 * 32, 1])
    b_shape = torch.Size([1, 7, 1, 4 * 32])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 32 * 2, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [1 * 32, 4 * 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
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
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_w_width(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([1, 2, 2 * 32, 40 * 32])
    b_shape = torch.Size([1, 1, 2 * 32, 1])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [32 * 2 * 2, 10 * 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [32 * 2 * 1, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, a_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, a_sharded_config),
    )

    for a_config, b_config, out_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

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
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=out_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_tt_sharded, out_pt)

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_h_width(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 1, 64, 7 * 32])
    b_shape = torch.Size([1, 1, 1, 7 * 32])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 1 * 64, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [1 * 32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, a_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, a_sharded_config),
    )

    for a_config, b_config, out_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

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
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=out_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_scalar_width(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 1, 64, 7 * 32])
    b_shape = torch.Size([1, 1, 1, 1])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 1 * 64, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [1 * 32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, a_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, a_sharded_config),
    )

    for a_config, b_config, out_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

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
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=out_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
@pytest.mark.parametrize(
    "sub_core_grids",
    (
        None,
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
            }
        ),
    ),
)
def test_binary_sharded_bcast_hw_mixed_width(device, dtype_pt, dtype_tt, sub_core_grids):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 1, 1, 7 * 32])
    b_shape = torch.Size([1, 1, 2 * 32, 1])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 1 * 32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))}),
        strategy=ttnn.ShardStrategy.WIDTH,
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
        out_tt_sharded = ttnn.add(
            a_tt, b_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, sub_core_grids=sub_core_grids, use_legacy=None
        )
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, sub_core_grids=sub_core_grids, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    (
        [torch.bfloat16, ttnn.bfloat16],
        [torch.float32, ttnn.float32],
    ),
)
@pytest.mark.parametrize(
    "nb, nc, nh, nw",
    (
        # binary shapes
        (1, 1, 32, 128 * 1024),
    ),
)
def test_binary_subcoregrid(dtype_pt, dtype_tt, nb, nc, nh, nw, device):
    """Test binary operations with sub_core_grids parameter"""
    torch.manual_seed(10)
    shape = [nb, nc, nh, nw]
    inp_a = torch.rand(*shape).to(dtype_pt)
    inp_b = torch.rand(*shape).to(dtype_pt)

    a = ttnn.Tensor(
        inp_a.flatten().tolist(),
        shape,
        dtype_tt,
        ttnn.TILE_LAYOUT,
        device,
    )

    b = ttnn.Tensor(
        inp_b.flatten().tolist(),
        shape,
        dtype_tt,
        ttnn.TILE_LAYOUT,
        device,
    )
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    out_tt = ttnn.gt(
        a,
        1,
        memory_config=out_mem_config,
        sub_core_grids=ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
            }
        ),
    )
    out = ttnn.to_torch(out_tt)
    expected = torch.gt(inp_a, 1)
    assert_with_pcc(out, expected)

    out_tt = ttnn.div(
        a,
        2,
        memory_config=out_mem_config,
        sub_core_grids=ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
            }
        ),
    )
    out = ttnn.to_torch(out_tt)
    expected = torch.div(inp_a, 2)
    assert_with_pcc(out, expected)

    out_tt = ttnn.multiply(
        a,
        b,
        memory_config=out_mem_config,
        sub_core_grids=ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
            }
        ),
    )
    out = ttnn.to_torch(out_tt)
    expected = torch.multiply(inp_a, inp_b)
    assert_with_pcc(out, expected)

    out_tt = ttnn.add(
        a,
        b,
        memory_config=out_mem_config,
        sub_core_grids=ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
            }
        ),
    )
    out = ttnn.to_torch(out_tt)
    expected = torch.add(inp_a, inp_b)
    assert_with_pcc(out, expected)


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
    torch.manual_seed(0)
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
    out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=shard_config, use_legacy=None)
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
    torch.manual_seed(0)
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

        out_tt_sharded = ttnn_fn(a_tt, b_tt, memory_config=shard_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert ttnn.pearson_correlation_coefficient(out_tt_sharded, out_pt) >= 0.99988

        out_tt_interleaved = ttnn_fn(a_tt, b_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=None)
        out_tt_interleaved = ttnn.to_torch(out_tt_interleaved)
        assert ttnn.pearson_correlation_coefficient(out_tt_interleaved, out_pt) >= 0.99988

        out_tt_interleaved = ttnn_fn(a_tt, b_tt, use_legacy=None)
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
    torch.manual_seed(0)
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
    out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=shard_config, use_legacy=None)
    out_tt_sharded = ttnn.to_torch(out_tt_sharded)
    assert ttnn.pearson_correlation_coefficient(out_tt_sharded, out_pt) >= 0.99988


@pytest.mark.parametrize(
    "a_shape, b_shape",
    ((torch.Size([5, 7, 32, 96]), torch.Size([5, 7, 32, 96])),),
)
@pytest.mark.parametrize(
    "shard_type, shard_size, core_range",
    (
        [ttnn.ShardStrategy.HEIGHT, [64, 96], ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (2, 5))})],
        [ttnn.ShardStrategy.WIDTH, [35 * 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 1))})],
        [ttnn.ShardStrategy.BLOCK, [32 * 7, 64], ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (1, 4))})],
        [ttnn.ShardStrategy.BLOCK, [32 * 8, 32], ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (2, 4))})],
        [ttnn.ShardStrategy.BLOCK, [32 * 8, 64], ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (1, 4))})],
    ),
)
def test_binary_sharded_bcast_no_identical_uneven(a_shape, b_shape, shard_type, shard_size, core_range, device):
    torch.manual_seed(0)
    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16)(a_shape)
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16)(b_shape)

    a_sharded_config = ttnn.create_sharded_memory_config(
        shard_size,
        core_grid=core_range,
        strategy=shard_type,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, a_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, a_sharded_config, a_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (a_sharded_config, a_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, a_sharded_config, a_sharded_config),
    )
    for a_config, b_config, dst_config in input_combinations:
        a_tt = ttnn.from_torch(
            a_pt,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=a_config,
        )
        b_tt = ttnn.from_torch(
            b_pt,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=b_config,
        )

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=dst_config, use_legacy=False)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert ttnn.pearson_correlation_coefficient(out_tt_sharded, out_pt) >= 0.99988

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=False)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert ttnn.pearson_correlation_coefficient(out_tt_sharded, out_pt) >= 0.99988


@pytest.mark.parametrize("scalar", [1.7, -0.25])
@pytest.mark.parametrize(
    "a_shape, shard_type, shard_size, core_range",
    (
        # W + HEIGHT
        [
            torch.Size([1, 40 * 32]),
            ttnn.ShardStrategy.HEIGHT,
            [32, 40 * 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))}),
        ],
        # W + WIDTH
        [
            torch.Size([1, 40 * 32]),
            ttnn.ShardStrategy.WIDTH,
            [32, 10 * 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),
        ],
        # W + BLOCK
        [
            torch.Size([1, 40 * 32]),
            ttnn.ShardStrategy.BLOCK,
            [32, 10 * 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 0))}),
        ],
        # H + HEIGHT
        [
            torch.Size([40 * 32, 1]),
            ttnn.ShardStrategy.HEIGHT,
            [10 * 32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),
        ],
        # H + WIDTH
        [
            torch.Size([4 * 32, 1]),
            ttnn.ShardStrategy.WIDTH,
            [4 * 32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))}),
        ],
        # H + BLOCK
        [
            torch.Size([40 * 32, 1]),
            ttnn.ShardStrategy.BLOCK,
            [10 * 32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),
        ],
        # C + HEIGHT
        [
            torch.Size([40, 1, 1]),
            ttnn.ShardStrategy.HEIGHT,
            [10 * 32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),
        ],
        # C + WIDTH
        [
            torch.Size([40, 1, 1]),
            ttnn.ShardStrategy.WIDTH,
            [40 * 32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))}),
        ],
        # C + BLOCK
        [
            torch.Size([40, 1, 1]),
            ttnn.ShardStrategy.BLOCK,
            [10 * 32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),
        ],
        # N + HEIGHT
        [
            torch.Size([40, 1, 1, 1]),
            ttnn.ShardStrategy.HEIGHT,
            [10 * 32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),
        ],
        # N + WIDTH
        [
            torch.Size([40, 1, 1, 1]),
            ttnn.ShardStrategy.WIDTH,
            [40 * 32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))}),
        ],
        # N + BLOCK
        [
            torch.Size([40, 1, 1, 1]),
            ttnn.ShardStrategy.BLOCK,
            [10 * 32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),
        ],
    ),
)
@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    (
        [torch.bfloat16, ttnn.bfloat16],
        # does not work fro binary_ng yet
        # [torch.bfloat16, ttnn.bfloat8_b],
    ),
)
def test_binary_sharded_bcast_scalar_value(
    dtype_pt, dtype_tt, scalar, a_shape, shard_type, shard_size, core_range, device
):
    torch.manual_seed(0)
    sharded_config = ttnn.create_sharded_memory_config(
        shard_size,
        core_grid=core_range,
        strategy=shard_type,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, sharded_config),
        (sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (sharded_config, sharded_config),
    )
    for a_config, dst_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        a_tt = ttnn.from_torch(
            a_pt,
            dtype=dtype_tt,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=a_config,
        )

        out_pt = torch.add(a_pt, scalar)
        out_tt_sharded = ttnn.add(a_tt, scalar, memory_config=dst_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_tt_sharded, out_pt)

        out_tt_interleaved = ttnn.add(a_tt, scalar, use_legacy=None)
        out_tt_interleaved = ttnn.to_torch(out_tt_interleaved)
        assert_with_pcc(out_tt_interleaved, out_pt)


@pytest.mark.parametrize("scalar", [1.7, -0.25])
@pytest.mark.parametrize(
    "a_shape, shard_type, shard_size, core_range",
    (
        # HEIGHT
        [
            torch.Size([5, 7, 32, 96]),
            ttnn.ShardStrategy.HEIGHT,
            [64, 96],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (2, 5))}),
        ],
        # WIDTH
        [
            torch.Size([1, 1, 32 * 25, 96]),
            ttnn.ShardStrategy.WIDTH,
            [32 * 25, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 1))}),
        ],
        # BLOCK
        [
            torch.Size([5, 7, 32, 96]),
            ttnn.ShardStrategy.BLOCK,
            [32 * 8, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (1, 4))}),
        ],
    ),
)
@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    (
        [torch.bfloat16, ttnn.bfloat16],
        # does not work fro binary_ng yet
        # [torch.bfloat16, ttnn.bfloat8_b],
    ),
)
def test_binary_sharded_bcast_scalar_value_uneven(
    dtype_pt, dtype_tt, scalar, a_shape, shard_type, shard_size, core_range, device
):
    torch.manual_seed(0)
    sharded_config = ttnn.create_sharded_memory_config(
        shard_size,
        core_grid=core_range,
        strategy=shard_type,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, sharded_config),
        (sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (sharded_config, sharded_config),
    )
    for a_config, dst_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        a_tt = ttnn.from_torch(
            a_pt,
            dtype=dtype_tt,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=a_config,
        )

        out_pt = torch.add(a_pt, scalar)
        out_tt_sharded = ttnn.add(a_tt, scalar, memory_config=dst_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_tt_sharded, out_pt)

        out_tt_interleaved = ttnn.add(a_tt, scalar, use_legacy=None)
        out_tt_interleaved = ttnn.to_torch(out_tt_interleaved)
        assert_with_pcc(out_tt_interleaved, out_pt)


@pytest.mark.parametrize("scalar", [-0.25])
@pytest.mark.parametrize(
    "a_shape, shard_type, shard_size, core_range",
    (
        [
            # TODO: shape cannot be uneven shard for now
            torch.Size([64, 32]),
            ttnn.ShardStrategy.HEIGHT,
            [32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 1))}),
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
    out_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16)(
        a_shape
    )

    with pytest.raises(RuntimeError) as e:
        a_tt = ttnn.from_torch(
            a_pt,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=a_sharded_config,
        )

        out_tt = ttnn.from_torch(
            out_pt,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=a_sharded_config,
        )

        tt_out = ttnn.add(a_tt, scalar, output_tensor=out_tt, use_legacy=None)
        tt_out = ttnn.to_torch(tt_out)
        assert_with_pcc(tt_out, out_pt)


@pytest.mark.parametrize("scalar", [-0.25])
@pytest.mark.parametrize(
    "a_shape, shard_type, shard_size, core_range",
    (
        [
            torch.Size([640, 32]),
            ttnn.ShardStrategy.HEIGHT,
            [320, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 1))}),
        ],
    ),
)
def test_binary_sharded_scalar_row_major(scalar, a_shape, shard_type, shard_size, core_range, device):
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
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=a_sharded_config,
    )
    tt_out = ttnn.add(a_tt, scalar, use_legacy=None)
    tt_out = ttnn.to_torch(tt_out)
    assert_with_pcc(tt_out, torch.add(a_pt, scalar))


@pytest.mark.parametrize(
    "a_shape, b_shape, a_shard_size, b_shard_size, core_range",
    (
        [
            torch.Size([5, 6, 320, 2]),
            torch.Size([5, 6, 320, 1]),
            [320, 32],
            [320, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (4, 5))}),
        ],
        [
            torch.Size([5, 6, 320, 33]),
            torch.Size([5, 6, 320, 1]),
            [320, 64],
            [320, 32],
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
    torch.manual_seed(0)
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
    out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=a_shard_config, use_legacy=None)
    out_tt_sharded = ttnn.to_torch(out_tt_sharded)
    assert_with_pcc(out_tt_sharded, out_pt)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
@pytest.mark.parametrize(
    "a_shape, b_shape, shard_type, shard_size, core_range",
    (
        # for row major layout, shard shape must be divisible by tile height and width
        [
            torch.Size([64, 33]),
            torch.Size([64, 33]),
            ttnn.ShardStrategy.HEIGHT,
            [32, 33],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 1))}),
        ],
        # for row major layout, width sharding is not supported
        [
            torch.Size([64, 4 * 32]),
            torch.Size([64, 4 * 32]),
            ttnn.ShardStrategy.WIDTH,
            [64, 4 * 32],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))}),
        ],
        # for row major layout, block sharding is not supported
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

        _ = ttnn.add(a_tt, b_tt, memory_config=a_sharded_config, use_legacy=None)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
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
def test_binary_sharded_row_major_layout_mixed(
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
        layout=ttnn.TILE_LAYOUT,
        memory_config=b_sharded_config,
    )
    out_pt = torch.add(a_pt, b_pt)
    out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=a_sharded_config, use_legacy=None)
    out_tt_sharded = ttnn.to_torch(out_tt_sharded)
    assert_with_pcc(out_tt_sharded, out_pt)

    out_pt = torch.add(a_pt, b_pt)
    out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
    out_tt_sharded = ttnn.to_torch(out_tt_sharded)
    assert_with_pcc(out_tt_sharded, out_pt)


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

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=None)
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

    torch_output_tensor = torch_input_tensor_a - torch_input_tensor_b

    output_tensor = ttnn.subtract(
        input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=None
    )
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988


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

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=None)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
    assert_with_pcc(torch_output_tensor, output_tensor)


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

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=None)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        [[1, 1, 9600, 1], [1, 1, 1, 9600]],
        [[1, 1, 1, 9600], [1, 1, 9600, 1]],
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

    torch_output_tensor = torch_input_tensor_a - torch_input_tensor_b
    # add is non associative, so we use subtract to test the bcast because it is associative
    output_tensor = ttnn.subtract(
        input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=None
    )
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
    torch.manual_seed(0)
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
    result = ttnn.add(tt_a, tt_b, use_legacy=None)

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

    out_tt = ttnn.add(a_tt, b_tt, use_legacy=None)
    out_pt = golden_fn(a_pt, b_pt)

    assert compare_pcc([out_tt], [out_pt])


def test_add_1m(device):
    torch.manual_seed(0)
    a = torch.ones(1, 1) * 1_000_000
    b = torch.ones(32, 32)
    c = a + b

    ta = ttnn.from_torch(a, device=device, layout=ttnn.TILE_LAYOUT)
    tb = ttnn.from_torch(b, device=device, layout=ttnn.TILE_LAYOUT)
    tc = ttnn.add(ta, tb, use_legacy=None)

    # torch.allclose() will fail when op internally uses TF32 format
    # assert torch.allclose(c, ttnn.to_torch(tc)), f"{c} != {ttnn.to_torch(tc)}"
    assert_with_pcc(c, ttnn.to_torch(tc))


def test_add_i32(device):
    torch.manual_seed(2024)
    a = torch.cat([torch.zeros(128, dtype=torch.int32), torch.ones(128, dtype=torch.int32)])
    a = torch.reshape(a, (1, 1, 1, 256))
    b = torch.zeros((1, 1, 256, 256), dtype=torch.int32)

    torch_add = a + b

    input_tensor_a = ttnn.from_torch(a, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(b, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, use_legacy=None)

    output_tensor = ttnn.to_torch(output_tensor)
    assert torch.equal(torch_add, output_tensor)


def test_add_error(device):
    pytest.skip("Test is skipped because half mem config feature not supported yet")
    # Create input tensors with specified shapes
    input_shape = [1, 1, 1, 39576]
    bias_shape = [1, 39576]

    # Create random tensors
    torch_input = torch.randn(*input_shape, dtype=torch.bfloat16)
    torch_bias = torch.randn(*bias_shape, dtype=torch.bfloat16)

    # Convert to TTNN tensors with tile layout
    ttnn_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_bias = ttnn.from_torch(torch_bias, device=device, layout=ttnn.TILE_LAYOUT)

    # Perform the add operation with the specified memory config
    ttnn_result = ttnn.add(ttnn_input, ttnn_bias, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, use_legacy=None)


@pytest.mark.parametrize(
    "shapes",
    [
        [[8, 16, 32], [1, 1, 1]],
        [[8, 16, 32], [1, 1, 32]],
    ],
)
def test_sub_implicit_broadcast(device, shapes):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shapes[0], dtype=torch.float32)
    torch_input_tensor_b = torch.rand(shapes[1], dtype=torch.float32)
    torch_output_tensor = torch.sub(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.sub(
        input_tensor_a, input_tensor_b, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=None
    )
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


def test_small_fp32_multiply(device):
    # Scaling with 0.01 to get realistic values that appear during training.
    a = ttnn.from_torch(0.01 * torch.randn((1, 1, 2048), dtype=torch.float32), layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(0.01 * torch.randn((4, 32, 2048), dtype=torch.float32), layout=ttnn.TILE_LAYOUT, device=device)

    out_torch = torch.multiply(ttnn.to_torch(a), ttnn.to_torch(b))
    # print("Torch multiply result:")
    # print(out_torch)

    out = ttnn.multiply(a, b)
    # print("TTNN multiply result:")
    # print(out)

    # assert_allclose(out_torch, ttnn.to_torch(out))
    assert_with_pcc(out_torch, ttnn.to_torch(out))


block_sharded_memory_config = ttnn.create_sharded_memory_config(
    [32, 32],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 1))}),
    strategy=ttnn.ShardStrategy.BLOCK,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_w_block(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 7, 32 * 2, 4 * 32])
    b_shape = torch.Size([1, 7, 32 * 2, 1])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 32 * 2, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 6))}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [32 * 2, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, a_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, a_sharded_config),
    )

    for a_config, b_config, out_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

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
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=out_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        # swap a and b
        out_pt = torch.add(b_pt, a_pt)
        out_tt_sharded = ttnn.add(b_tt, a_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_h_block(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 7, 32 * 2, 4 * 32])
    b_shape = torch.Size([1, 7, 1, 4 * 32])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 32 * 2, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 6))}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 6))}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, a_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, a_sharded_config),
    )

    for a_config, b_config, out_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

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
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=out_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_scalar_block(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 7, 32 * 2, 4 * 32])
    b_shape = torch.Size([1, 7, 1, 1])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 32 * 2, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 6))}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, a_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, a_sharded_config),
    )

    for a_config, b_config, out_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

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
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=out_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_hw_mixed_block(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 7, 32 * 2, 1])
    b_shape = torch.Size([1, 7, 1, 4 * 32])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 32 * 2, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [32, 4 * 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config),
    )

    for a_config, b_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

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
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(b_pt, a_pt)
        out_tt_sharded = ttnn.add(b_tt, a_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


height_sharded_memory_config = ttnn.create_sharded_memory_config(
    [256, 32],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (7, 7))}),
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)


@pytest.mark.parametrize(
    "a_shape, b_shape",
    ((torch.Size([1, 1, 16384, 4]), torch.Size([])),),
)
@pytest.mark.parametrize(
    "a_config, b_config, out_config",
    [
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, height_sharded_memory_config],
        [height_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        [height_sharded_memory_config, ttnn.DRAM_MEMORY_CONFIG, height_sharded_memory_config],
    ],
)
@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_scalar_zero_dim(
    a_shape, b_shape, a_config, b_config, out_config, dtype_pt, dtype_tt, device
):
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

    out_pt = torch.mul(a_pt, b_pt)
    out_tt = ttnn.mul(a_tt, b_tt, memory_config=out_config, use_legacy=None)
    assert_with_pcc(ttnn.to_torch(out_tt), out_pt)

    out_pt = torch.mul(b_pt, a_pt)
    out_tt = ttnn.mul(b_tt, a_tt, memory_config=out_config, use_legacy=None)
    assert_with_pcc(ttnn.to_torch(out_tt), out_pt)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_shardspec_mixed_buffer_type(dtype_pt, dtype_tt, device):
    torch.manual_seed(0)
    dram_grid_size = device.dram_grid_size()
    input_shape = (1, 1, dram_grid_size.x * dram_grid_size.y * 32, 32)

    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=dtype_pt), dtype_tt)(input_shape)
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=dtype_pt), dtype_tt)(input_shape)

    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (2, 3))})
    N, C, H, W = input_shape
    n_cores = 12
    import math

    shard_spec = ttnn.ShardSpec(
        shard_grid, [math.ceil((N * C * H) / n_cores / 32) * 32, W], ttnn.ShardOrientation.ROW_MAJOR
    )
    a_config = ttnn.MemoryConfig(ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    print("dram_grid_size:", dram_grid_size.x * dram_grid_size.y)
    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
        ),
        [
            divup(N * C * H, (dram_grid_size.x * dram_grid_size.y)),
            W,
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    b_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec)

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

    out_pt = torch.mul(a_pt, b_pt)
    out_tt = ttnn.mul(a_tt, b_tt, use_legacy=None)
    assert_with_pcc(ttnn.to_torch(out_tt), out_pt)

    out_pt = torch.mul(b_pt, a_pt)
    out_tt = ttnn.mul(b_tt, a_tt, use_legacy=None)
    assert_with_pcc(ttnn.to_torch(out_tt), out_pt)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_shardspec_dram(dtype_pt, dtype_tt, device):
    torch.manual_seed(0)
    dram_grid_size = device.dram_grid_size()
    input_shape = (1, 1, dram_grid_size.x * dram_grid_size.y * 32, 32)
    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=dtype_pt), dtype_tt)(input_shape)
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=dtype_pt), dtype_tt)(input_shape)

    N, C, H, W = input_shape
    print("N, C, H, W:", N, C, H, W)
    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
        ),
        [divup(N * C * H, (dram_grid_size.x * dram_grid_size.y)), W],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    a_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec)
    b_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec)

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
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_pt = torch.mul(a_pt, b_pt)
    out_tt = ttnn.mul(a_tt, b_tt, use_legacy=None)
    assert_with_pcc(ttnn.to_torch(out_tt), out_pt)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    (
        [torch.bfloat16, ttnn.bfloat16],
        # does not work for binary_ng yet
        # [torch.bfloat16, ttnn.bfloat8_b],
    ),
)
def test_binary_sharded_bcast_w_height_uneven(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([5, 7, 2 * 32, 4 * 32])
    b_shape = torch.Size([5, 7, 2 * 32, 1])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [11 * 32, 4 * 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [11 * 32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, a_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, a_sharded_config),
    )

    for a_config, b_config, out_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

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
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=out_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_tt_sharded, out_pt)

        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_tt_sharded, out_pt)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_w_width_uneven(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([1, 2, 2 * 32, 40 * 32])
    b_shape = torch.Size([1, 1, 2 * 32, 1])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [32 * 2 * 2, 11 * 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [32 * 2 * 1, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, a_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, a_sharded_config),
    )

    for a_config, b_config, out_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

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
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=out_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_tt_sharded, out_pt)

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_w_block_uneven(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 7, 32 * 2, 3 * 32])
    b_shape = torch.Size([1, 7, 32 * 2, 1])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [3 * 32 * 2, 2 * 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (1, 4))}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [32 * 2, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, a_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, a_sharded_config),
    )

    for a_config, b_config, out_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

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
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=out_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        # swap a and b
        out_pt = torch.add(b_pt, a_pt)
        out_tt_sharded = ttnn.add(b_tt, a_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_h_height_uneven(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 7, 32 * 2, 4 * 32])
    b_shape = torch.Size([1, 7, 1, 4 * 32])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [3 * 32 * 2, 4 * 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 4))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [1 * 32, 4 * 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, a_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, a_sharded_config),
    )

    for a_config, b_config, out_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

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
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=out_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_h_width_uneven(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 1, 64, 7 * 32])
    b_shape = torch.Size([1, 1, 1, 7 * 32])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 1 * 64, 64],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [1 * 32, 64],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, a_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, a_sharded_config),
    )

    for a_config, b_config, out_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

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
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=out_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_h_block_uneven(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 7, 32 * 2, 5 * 32])
    b_shape = torch.Size([1, 7, 1, 5 * 32])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [3 * 32 * 2, 2 * 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (2, 4))}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [32 * 2, 2 * 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 4))}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, a_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, a_sharded_config),
    )

    for a_config, b_config, out_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

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
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=out_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_scalar_height_uneven(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 7, 2 * 32, 4 * 32])
    b_shape = torch.Size([1, 7, 1, 1])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [3 * 32 * 2, 4 * 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 4))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, a_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, a_sharded_config),
    )

    for a_config, b_config, out_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

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
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=out_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_scalar_width_uneven(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 1, 64, 7 * 32])
    b_shape = torch.Size([1, 1, 1, 1])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 1 * 64, 2 * 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [1 * 32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, a_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, a_sharded_config),
    )

    for a_config, b_config, out_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

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
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=out_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_scalar_block_uneven(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 7, 32 * 2, 5 * 32])
    b_shape = torch.Size([1, 7, 1, 1])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [3 * 32 * 2, 2 * 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (2, 4))}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, a_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, a_sharded_config),
    )

    for a_config, b_config, out_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

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
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=out_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


def rand_bf16_gen_dtype(shape, device, *, min=0, max=1, dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    torch_dtype = getattr(torch, dtype)
    pt = torch.rand(shape, dtype=torch_dtype) * (max - min) + min
    tt = ttnn.from_torch(pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)
    return pt, tt


@pytest.mark.parametrize(
    "a_shape, b_shape",
    (
        # row bcast
        (torch.Size([5, 10, 640, 128]), torch.Size([5, 10, 1, 128])),
        (torch.Size([5, 10, 1, 128]), torch.Size([5, 10, 640, 128])),
        # row col mixed bcast
        (torch.Size([5, 10, 640, 1]), torch.Size([5, 10, 1, 128])),
        (torch.Size([5, 10, 1, 128]), torch.Size([5, 10, 640, 1])),
    ),
)
def test_binary_sfpu_row_bcast(a_shape, b_shape, device):
    torch.manual_seed(0)
    # make 0 exclusive for rhs of div
    min, max = (1, 0)

    ttnn_fn = ttnn.pow
    dtype = "bfloat16"
    a_pt, a_tt = rand_bf16_gen_dtype(a_shape, device, dtype=dtype)
    b_pt, b_tt = rand_bf16_gen_dtype(b_shape, device, min=min, max=max, dtype=dtype)

    out_tt = ttnn_fn(
        a_tt,
        b_tt,
        use_legacy=None,
    )
    golden_fn = ttnn.get_golden_function(ttnn_fn)
    golden = golden_fn(a_pt, b_pt)

    calculated = ttnn.to_torch(out_tt)
    assert_with_pcc(calculated, golden, 0.999)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_hw_mixed_output_mixed_shard_strategy_mixed_uneven(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 7, 32 * 2, 1])
    b_shape = torch.Size([1, 7, 1, 4 * 32])
    out_shape = torch.Size([2, 7, 32 * 2, 4 * 32])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [3 * 32 * 2, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 4))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [7 * 32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    out_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 2 * 32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 6))}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, out_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, out_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, out_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, out_sharded_config),
    )

    for a_config, b_config, dst_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)
        out_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(out_shape)

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
        out_tt = ttnn.from_torch(
            out_pt,
            dtype=dtype_tt,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=dst_config,
        )

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, output_tensor=out_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        # swap a and b
        out_pt = torch.add(b_pt, a_pt)
        out_tt_sharded = ttnn.add(b_tt, a_tt, output_tensor=out_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(b_pt, a_pt)
        out_tt_sharded = ttnn.add(b_tt, a_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_hw_mixed_output_mixed_shard_strategy_mixed_uneven_unaligned_shape(
    device, dtype_pt, dtype_tt
):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 7, 30 * 2, 1])
    b_shape = torch.Size([1, 7, 1, 4 * 30])
    out_shape = torch.Size([2, 7, 30 * 2, 4 * 30])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [3 * 32 * 2, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((1, 0), (1, 2)), ttnn.CoreRange((2, 0), (2, 1))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [7 * 32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((1, 0), (1, 1)), ttnn.CoreRange((2, 0), (2, 1))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    out_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 2 * 32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 6))}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, out_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, out_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, out_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, out_sharded_config),
    )

    for a_config, b_config, dst_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)
        out_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(out_shape)

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
        out_tt = ttnn.from_torch(
            out_pt,
            dtype=dtype_tt,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=dst_config,
        )

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, output_tensor=out_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        # swap a and b
        out_pt = torch.add(b_pt, a_pt)
        out_tt_sharded = ttnn.add(b_tt, a_tt, output_tensor=out_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(b_pt, a_pt)
        out_tt_sharded = ttnn.add(b_tt, a_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


@pytest.mark.parametrize("input_shape", [(1, 4096, 640)])
@pytest.mark.parametrize("is_legacy", [True, False])
def test_add_sharded(device, input_shape, is_legacy):
    torch_input_tensor_a = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(512, 96),
        core_grid=ttnn.CoreGrid(y=8, x=7),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded_mem_config
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded_mem_config
    )
    output = ttnn.add(input_tensor_a, input_tensor_b, use_legacy=is_legacy)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_hw_mixed_orientation_output(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 7, 31 * 2, 1])
    b_shape = torch.Size([1, 7, 1, 4 * 31])
    out_shape = torch.Size([2, 7, 31 * 2, 4 * 31])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 32 * 2, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((1, 0), (1, 6))}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [4 * 32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (6, 0))}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.COL_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    out_sharded_config = ttnn.create_sharded_memory_config(
        [32, 2 * 2 * 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (6, 3))}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.COL_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, out_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, out_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, out_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, out_sharded_config),
    )

    for a_config, b_config, dst_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)
        out_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(out_shape)

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
        out_tt = ttnn.from_torch(
            out_pt,
            dtype=dtype_tt,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=dst_config,
        )

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, output_tensor=out_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        # swap a and b
        out_pt = torch.add(b_pt, a_pt)
        out_tt_sharded = ttnn.add(b_tt, a_tt, output_tensor=out_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(b_pt, a_pt)
        out_tt_sharded = ttnn.add(b_tt, a_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_h_mixed_strategy_mixed_L1(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 7, 32 * 2, 4 * 32])
    b_shape = torch.Size([1, 7, 1, 4 * 32])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 32 * 2, 4 * 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [7 * 32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    import itertools

    input_combinations = itertools.product(
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, a_sharded_config],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, b_sharded_config],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, a_sharded_config],
    )
    for a_config, b_config, dst_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

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
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=dst_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        # swap a and b
        out_pt = torch.add(b_pt, a_pt)
        out_tt_sharded = ttnn.add(b_tt, a_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(b_pt, a_pt)
        out_tt_sharded = ttnn.add(b_tt, a_tt, memory_config=dst_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_identical_mixed_strategy(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 7, 32 * 2, 4 * 32])
    b_shape = torch.Size([2, 7, 32 * 2, 4 * 32])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 32 * 2, 4 * 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 7 * 32 * 2, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, a_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, a_sharded_config),
    )

    for a_config, b_config, dst_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

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
        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(a_pt, b_pt)
        out_tt_sharded = ttnn.add(a_tt, b_tt, memory_config=dst_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        # swap a and b
        out_pt = torch.add(b_pt, a_pt)
        out_tt_sharded = ttnn.add(b_tt, a_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(b_pt, a_pt)
        out_tt_sharded = ttnn.add(b_tt, a_tt, memory_config=dst_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
@pytest.mark.parametrize("scalar", [1.7, -0.25])
def test_binary_sharded_bcast_scalar_value_mixed_shard_uneven(device, dtype_pt, dtype_tt, scalar):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 7, 32 * 2, 4 * 32])
    out_shape = torch.Size([2, 7, 32 * 2, 4 * 32])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [3 * 32 * 2, 4 * 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 4))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    out_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 2 * 32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 6))}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, out_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, out_sharded_config),
    )

    for a_config, dst_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        out_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(out_shape)

        a_tt = ttnn.from_torch(
            a_pt,
            dtype=dtype_tt,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=a_config,
        )
        out_tt = ttnn.from_torch(
            out_pt,
            dtype=dtype_tt,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=out_sharded_config,
        )

        out_pt = torch.add(a_pt, scalar)
        out_tt_sharded = ttnn.add(a_tt, scalar, output_tensor=out_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(a_pt, scalar)
        out_tt_sharded = ttnn.add(a_tt, scalar, memory_config=dst_config, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)

        out_pt = torch.add(a_pt, scalar)
        out_tt_sharded = ttnn.add(a_tt, scalar, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    (
        [torch.bfloat16, ttnn.bfloat16],
        [torch.float32, ttnn.float32],
    ),
)
@pytest.mark.parametrize(
    "nb, nc, nh, nw",
    (
        # binary shapes
        (1, 1, 32, 32 * 1024),
    ),
)
@pytest.mark.parametrize(
    "ttnn_fn",
    [
        ttnn.eq_,
        ttnn.gt_,
        ttnn.lt_,
        ttnn.le_,
        ttnn.ge_,
        ttnn.ne_,
        ttnn.logical_and_,
        ttnn.logical_or_,
        ttnn.logical_xor_,
    ],
)
def test_binary_inplace_ops_with_subcore_grids(dtype_pt, dtype_tt, nb, nc, nh, nw, device, ttnn_fn):
    torch.manual_seed(10)
    shape = [nb, nc, nh, nw]
    inp_a = torch.rand(*shape).to(dtype_pt)
    inp_b = torch.rand(*shape).to(dtype_pt)

    a = ttnn.Tensor(
        inp_a.flatten().tolist(),
        shape,
        dtype_tt,
        ttnn.TILE_LAYOUT,
        device,
    )

    b = ttnn.Tensor(
        inp_b.flatten().tolist(),
        shape,
        dtype_tt,
        ttnn.TILE_LAYOUT,
        device,
    )

    out_tt = ttnn_fn(
        a,
        b,
        sub_core_grids=ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
            }
        ),
    )
    out = ttnn.to_torch(out_tt)
    golden_fn = ttnn.get_golden_function(ttnn_fn)
    expected = golden_fn(inp_a, inp_b)
    assert torch.equal(out, expected)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    (
        [torch.bfloat16, ttnn.bfloat16],
        [torch.float32, ttnn.float32],
    ),
)
@pytest.mark.parametrize(
    "nb, nc, nh, nw",
    (
        # binary shapes
        (1, 1, 32, 32 * 1024),
    ),
)
@pytest.mark.parametrize(
    "rounding_mode",
    [
        "trunc",
        "floor",
        None,
    ],
)
def test_div_composite_ops_with_subcore_grids(dtype_pt, dtype_tt, nb, nc, nh, nw, rounding_mode, device):
    torch.manual_seed(10)
    shape = [nb, nc, nh, nw]
    inp_a = torch.rand(*shape).to(dtype_pt)
    inp_b = torch.rand(*shape).to(dtype_pt)

    a = ttnn.Tensor(
        inp_a.flatten().tolist(),
        shape,
        dtype_tt,
        ttnn.TILE_LAYOUT,
        device,
    )

    b = ttnn.Tensor(
        inp_b.flatten().tolist(),
        shape,
        dtype_tt,
        ttnn.TILE_LAYOUT,
        device,
    )

    out_tt = ttnn.div(
        a,
        b,
        rounding_mode=rounding_mode,
        sub_core_grids=ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
            }
        ),
    )
    out = ttnn.to_torch(out_tt)
    golden_fn = ttnn.get_golden_function(ttnn.div)
    expected = golden_fn(inp_a, inp_b, rounding_mode=rounding_mode)
    assert_with_pcc(out, expected)

    out_tt = ttnn.div(
        a,
        2.0,
        rounding_mode=rounding_mode,
        sub_core_grids=ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
            }
        ),
    )
    out = ttnn.to_torch(out_tt)
    golden_fn = ttnn.get_golden_function(ttnn.div)
    expected = golden_fn(inp_a, 2.0, rounding_mode=rounding_mode)
    assert_with_pcc(out, expected)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    (
        [torch.bfloat16, ttnn.bfloat16],
        [torch.float32, ttnn.float32],
    ),
)
@pytest.mark.parametrize(
    "nb, nc, nh, nw",
    (
        # binary shapes
        (1, 1, 32, 32 * 1024),
    ),
)
def test_remainder_composite_ops_with_subcore_grids(dtype_pt, dtype_tt, nb, nc, nh, nw, device):
    torch.manual_seed(10)
    shape = [nb, nc, nh, nw]
    inp_a = torch.rand(*shape).to(dtype_pt)
    inp_b = torch.rand(*shape).to(dtype_pt)

    a = ttnn.Tensor(
        inp_a.flatten().tolist(),
        shape,
        dtype_tt,
        ttnn.TILE_LAYOUT,
        device,
    )

    b = ttnn.Tensor(
        inp_b.flatten().tolist(),
        shape,
        dtype_tt,
        ttnn.TILE_LAYOUT,
        device,
    )

    out_tt = ttnn.remainder(
        a,
        b,
        sub_core_grids=ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
            }
        ),
    )
    out = ttnn.to_torch(out_tt)
    golden_fn = ttnn.get_golden_function(ttnn.remainder)
    expected = golden_fn(inp_a, inp_b, device=device)
    assert_with_pcc(out, expected)

    out_tt = ttnn.remainder(
        a,
        2.0,
        sub_core_grids=ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
            }
        ),
    )
    out = ttnn.to_torch(out_tt)
    golden_fn = ttnn.get_golden_function(ttnn.remainder)
    expected = golden_fn(inp_a, 2.0, device=device)
    assert_with_pcc(out, expected)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_binary_sharded_bcast_identical_sdxl(device, dtype_pt, dtype_tt):
    torch.manual_seed(0)
    a_shape = torch.Size([1, 1, 4096, 640])
    b_shape = torch.Size([1, 1, 4096, 640])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [512, 128],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (4, 7))}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = ((ttnn.L1_MEMORY_CONFIG, a_sharded_config),)

    for a_config, b_config in input_combinations:
        a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(a_shape)
        b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(b_shape)

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
        out_tt_sharded = ttnn.add(a_tt, b_tt, use_legacy=None)
        out_tt_sharded = ttnn.to_torch(out_tt_sharded)
        assert_with_pcc(out_pt, out_tt_sharded)


def test_binary_reshard(device):
    torch.manual_seed(0)
    # Create input tensors (32x8192 = 1x256 tiles)
    torch_a = torch.randn(32, 8192, dtype=torch.bfloat16)
    torch_b = torch.randn(32, 8192, dtype=torch.bfloat16)

    # Convert to TTNN tensors on device (DRAM interleaved)
    a = ttnn.from_torch(torch_a, device=device, layout=ttnn.TILE_LAYOUT)
    b = ttnn.from_torch(torch_b, device=device, layout=ttnn.TILE_LAYOUT)

    # Shard both inputs to 64 cores (8x8 grid)
    # 256 tiles / 64 cores = 4 tiles per shard
    shard_spec_64 = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
        (32, 128),  # shard shape: 32 rows x 128 cols (4 tiles width)
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config_64 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec_64)

    # Output shard spec: 32 cores (first 4 columns of 8x8 grid)
    # 256 tiles / 32 cores = 8 tiles per shard
    shard_spec_32 = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        (32, 256),  # shard shape: 32 rows x 256 cols (8 tiles width)
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config_32 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec_32)

    expected = torch_a + torch_b
    a_sharded = ttnn.to_memory_config(a, mem_config_64)
    b_sharded = ttnn.to_memory_config(b, mem_config_64)
    result = ttnn.add(a_sharded, b_sharded, memory_config=mem_config_32, use_legacy=None)
    result = ttnn.to_torch(result)
    assert_with_pcc(expected, result)

    a_sharded = ttnn.to_memory_config(a, mem_config_32)
    b_sharded = ttnn.to_memory_config(b, mem_config_32)
    result = ttnn.add(a_sharded, b_sharded, memory_config=mem_config_64, use_legacy=None)
    result = ttnn.to_torch(result)
    assert_with_pcc(expected, result)


@pytest.mark.parametrize(
    "output_memory_config",
    [
        ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
    ],
)
@pytest.mark.parametrize("input_shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
def test_binary_sharded_half_mem_config(device, input_shard_orientation, output_memory_config):
    """Test binary operations with generic sharded memory configs that inherit shard spec from inputs"""
    torch.manual_seed(0)
    torch_input_a = torch.rand((32, 32, 64), dtype=torch.bfloat16)
    torch_input_b = torch.rand((32, 32, 64), dtype=torch.bfloat16)
    torch_output = torch_input_a + torch_input_b

    # Create sharded input configuration
    if output_memory_config.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        shard_config = ttnn.create_sharded_memory_config(
            shape=(32, 64),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=input_shard_orientation,
            use_height_and_width_as_shard_shape=True,
        )
    else:  # WIDTH_SHARDED or BLOCK_SHARDED
        # Use a simpler config for WIDTH_SHARDED and BLOCK_SHARDED
        shard_config = ttnn.create_sharded_memory_config(
            shape=(1024, 64),
            core_grid=ttnn.CoreGrid(y=1, x=1),
            strategy=ttnn.ShardStrategy.WIDTH
            if output_memory_config.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
            else ttnn.ShardStrategy.BLOCK,
            orientation=input_shard_orientation,
            use_height_and_width_as_shard_shape=True,
        )

    input_a = ttnn.from_torch(torch_input_a, layout=ttnn.TILE_LAYOUT, memory_config=shard_config, device=device)
    input_b = ttnn.from_torch(torch_input_b, layout=ttnn.TILE_LAYOUT, memory_config=shard_config, device=device)

    # Use generic sharded memory config without explicit shard spec - should inherit from inputs
    output = ttnn.add(input_a, input_b, memory_config=output_memory_config)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output)


@pytest.mark.parametrize(
    "output_memory_config",
    [
        ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    ],
)
def test_binary_sharded_half_mem_config_scalar(device, output_memory_config, scalar=3.0):
    """Test binary scalar operations with generic sharded memory configs that inherit shard spec from input"""
    torch.manual_seed(0)
    torch_input = torch.rand((32, 32, 64), dtype=torch.bfloat16)
    torch_output = scalar * torch_input

    # Create sharded input configuration with ROW_MAJOR orientation
    if output_memory_config.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        shard_config = ttnn.create_sharded_memory_config(
            shape=(32, 64),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    else:  # WIDTH_SHARDED
        shard_config = ttnn.create_sharded_memory_config(
            shape=(1024, 32),
            core_grid=ttnn.CoreGrid(y=1, x=2),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, memory_config=shard_config, device=device)

    # Use generic sharded memory config without explicit shard spec - should inherit from input
    output = ttnn.mul(input_tensor, scalar, memory_config=output_memory_config)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output)


def test_binary_bcast_sharded_output_half_mem_config(device):
    pytest.skip("Skipping test due to incomplete implementation of sharded broadcast output")
    """Test binary broadcast with generic sharded memory config inheriting from sharded input"""
    torch.manual_seed(0)
    torch_input_a = torch.rand((2, 7, 64, 128), dtype=torch.bfloat16)
    torch_input_b = torch.rand((64, 128), dtype=torch.bfloat16)
    torch_output = torch_input_a + torch_input_b

    # Create height sharded config for input B (2 rows x 1 col grid)
    b_shard_config = ttnn.create_sharded_memory_config(
        shape=(32, 128),
        core_grid=ttnn.CoreGrid(y=2, x=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_a = ttnn.from_torch(torch_input_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_b = ttnn.from_torch(torch_input_b, layout=ttnn.TILE_LAYOUT, memory_config=b_shard_config, device=device)

    # Use generic height sharded memory config - should inherit from input B
    output = ttnn.add(input_a, input_b, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output)
