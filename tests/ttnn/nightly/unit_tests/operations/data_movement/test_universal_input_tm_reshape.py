# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


# ---------------------------------------------------------------------------
# Helper: build a shard memory config for a given shape, strategy and layout
# ---------------------------------------------------------------------------


def _divisible_grid_1d(total_dim, max_cores, step):
    """Find largest n <= max_cores such that total_dim is divisible by n*step.

    Mirrors the grid search in recompute_shard_spec_for_output (reshape.cpp).
    """
    for n in range(max_cores, 0, -1):
        if total_dim % (n * step) == 0:
            return n
    return 1


# Element size (bytes) for each ttnn dtype we exercise in this file.
# BFP8_B is only valid with TILE layout, so its RM step is irrelevant.
_DTYPE_ELEM_SIZE = {
    ttnn.bfloat16: 2,
    ttnn.float32: 4,
    ttnn.uint32: 4,
    ttnn.int32: 4,
    ttnn.uint16: 2,
    ttnn.uint8: 1,
    ttnn.bfloat8_b: 1,
}


def make_sharded_memory_config(device, shape, strategy, layout, dtype=ttnn.bfloat16):
    """Create a valid sharded MemoryConfig for `shape` on `device`.

    For TILE layout the shard dims must be tile-aligned (multiples of 32).
    For ROW_MAJOR, shard_width * element_size must be a multiple of the
    recommended L1 alignment (64 bytes today), so the shard-width step is
    derived from the `dtype` argument.
    """
    grid = device.compute_with_storage_grid_size()
    max_x, max_y = grid.x, grid.y
    tile_h, tile_w = 32, 32

    # TILE layout: pad last two dims to tile-multiples before collapsing so
    # total_h/total_w match the physical layout and create_sharded_memory_config
    # gets tile-aligned shards.
    shape_for_memcfg = list(shape)
    if layout == ttnn.TILE_LAYOUT and len(shape) >= 2:
        padded_h_dim = ((shape[-2] + tile_h - 1) // tile_h) * tile_h
        padded_w_dim = ((shape[-1] + tile_w - 1) // tile_w) * tile_w
        total_h = padded_h_dim
        for d in shape[:-2]:
            total_h *= d
        total_w = padded_w_dim
        shape_for_memcfg[-2] = padded_h_dim
        shape_for_memcfg[-1] = padded_w_dim
    else:
        total_h = 1
        for d in shape[:-1]:
            total_h *= d
        total_w = shape[-1]

    step_h = tile_h if layout == ttnn.TILE_LAYOUT else 1
    recommended_alignment_bytes = 64
    element_size = _DTYPE_ELEM_SIZE.get(dtype, 2)
    rm_step_w = max(1, recommended_alignment_bytes // element_size)
    step_w = tile_w if layout == ttnn.TILE_LAYOUT else rm_step_w

    if strategy == ttnn.ShardStrategy.HEIGHT:
        ny = _divisible_grid_1d(total_h, max_y, step_h)
        core_grid = ttnn.CoreGrid(y=ny, x=1)
    elif strategy == ttnn.ShardStrategy.WIDTH:
        nx = _divisible_grid_1d(total_w, max_x, step_w)
        core_grid = ttnn.CoreGrid(y=1, x=nx)
    else:  # BLOCK
        ny = _divisible_grid_1d(total_h, max_y, step_h)
        nx = _divisible_grid_1d(total_w, max_x, step_w)
        core_grid = ttnn.CoreGrid(y=ny, x=nx)

    return ttnn.create_sharded_memory_config(
        shape=shape_for_memcfg,
        core_grid=core_grid,
        strategy=strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


# ---------------------------------------------------------------------------
# Test cases: (input_shape, output_shape, test_id)
#   - All dims are tile-aligned (multiples of 32) so both TILE and RM work
#   - Volume is conserved
# ---------------------------------------------------------------------------

RESHAPE_CASES = [
    # merge_ch: PerformView fast path; rest: distinct s2i→reshape→i2s bugs.
    ([1, 4, 256, 128], [1, 1, 1024, 128], "merge_ch"),
    ([1, 4, 256, 128], [1, 4, 128, 256], "swap_hw"),
    ([1, 4, 128, 256], [1, 4, 256, 128], "halve_w_double_h"),
    ([1, 4, 256, 128], [1, 4, 512, 64], "double_h_halve_w"),
]

LAYOUTS = [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]


# ---------------------------------------------------------------------------
# Parametrized test: input with various memory configs, output is interleaved
# ---------------------------------------------------------------------------


def _run_reshape(device, input_shape, output_shape, input_memory_config, layout):
    """Core test runner: reshape input_shape -> output_shape, check PCC."""
    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output = torch_input.reshape(output_shape)

    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=input_memory_config,
    )

    tt_output = ttnn.reshape(tt_input, output_shape)
    actual = ttnn.to_torch(tt_output)

    assert list(actual.shape) == list(
        torch_output.shape
    ), f"Shape mismatch: got {list(actual.shape)}, expected {list(torch_output.shape)}"
    assert torch.equal(torch_output, actual), "Data mismatch: reshape should preserve values exactly"


@pytest.mark.parametrize(
    "input_shape,output_shape,case_id",
    [(inp, out, cid) for inp, out, cid in RESHAPE_CASES],
    ids=[cid for _, _, cid in RESHAPE_CASES],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["TILE", "ROW_MAJOR"])
def test_reshape_dram_input(device, input_shape, output_shape, case_id, layout):
    """Baseline: DRAM interleaved input — should always pass."""
    _run_reshape(device, input_shape, output_shape, ttnn.DRAM_MEMORY_CONFIG, layout)


@pytest.mark.parametrize(
    "input_shape,output_shape,case_id",
    [(inp, out, cid) for inp, out, cid in RESHAPE_CASES],
    ids=[cid for _, _, cid in RESHAPE_CASES],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["TILE", "ROW_MAJOR"])
def test_reshape_l1_input(device, input_shape, output_shape, case_id, layout):
    """L1 interleaved input."""
    _run_reshape(device, input_shape, output_shape, ttnn.L1_MEMORY_CONFIG, layout)


@pytest.mark.parametrize(
    "input_shape,output_shape,case_id",
    [(inp, out, cid) for inp, out, cid in RESHAPE_CASES],
    ids=[cid for _, _, cid in RESHAPE_CASES],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["TILE", "ROW_MAJOR"])
def test_reshape_height_sharded_input(device, input_shape, output_shape, case_id, layout):
    """HEIGHT sharded input."""
    mem_cfg = make_sharded_memory_config(device, input_shape, ttnn.ShardStrategy.HEIGHT, layout)
    _run_reshape(device, input_shape, output_shape, mem_cfg, layout)


@pytest.mark.parametrize(
    "input_shape,output_shape,case_id",
    [(inp, out, cid) for inp, out, cid in RESHAPE_CASES],
    ids=[cid for _, _, cid in RESHAPE_CASES],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["TILE", "ROW_MAJOR"])
def test_reshape_width_sharded_input(device, input_shape, output_shape, case_id, layout):
    """WIDTH sharded input."""
    mem_cfg = make_sharded_memory_config(device, input_shape, ttnn.ShardStrategy.WIDTH, layout)
    _run_reshape(device, input_shape, output_shape, mem_cfg, layout)


@pytest.mark.parametrize(
    "input_shape,output_shape,case_id",
    [(inp, out, cid) for inp, out, cid in RESHAPE_CASES],
    ids=[cid for _, _, cid in RESHAPE_CASES],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["TILE", "ROW_MAJOR"])
def test_reshape_block_sharded_input(device, input_shape, output_shape, case_id, layout):
    """BLOCK sharded input."""
    mem_cfg = make_sharded_memory_config(device, input_shape, ttnn.ShardStrategy.BLOCK, layout)
    _run_reshape(device, input_shape, output_shape, mem_cfg, layout)


# ---------------------------------------------------------------------------
# Sharded output tests: interleaved input, sharded output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_shape,output_shape,case_id",
    [(inp, out, cid) for inp, out, cid in RESHAPE_CASES],
    ids=[cid for _, _, cid in RESHAPE_CASES],
)
@pytest.mark.parametrize(
    "strategy",
    [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK],
    ids=["height", "width", "block"],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["TILE", "ROW_MAJOR"])
def test_reshape_sharded_output(device, input_shape, output_shape, case_id, strategy, layout):
    """DRAM interleaved input, sharded output (reshape into a sharded tensor)."""
    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output = torch_input.reshape(output_shape)

    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_mem_cfg = make_sharded_memory_config(device, output_shape, strategy, layout)
    tt_output = ttnn.reshape(tt_input, output_shape, memory_config=output_mem_cfg)
    actual = ttnn.to_torch(tt_output)

    assert list(actual.shape) == list(
        torch_output.shape
    ), f"Shape mismatch: got {list(actual.shape)}, expected {list(torch_output.shape)}"
    assert torch.equal(torch_output, actual), "Data mismatch: reshape should preserve values exactly"


# ---------------------------------------------------------------------------
# Sharded-to-sharded: same strategy, input and output are both sharded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_shape,output_shape,case_id",
    [
        ([1, 4, 256, 128], [1, 1, 1024, 128], "merge_ch"),
        ([1, 4, 256, 128], [1, 4, 128, 256], "swap_hw"),
        ([1, 4, 128, 256], [1, 4, 256, 128], "halve_w_double_h"),
        ([1, 4, 256, 128], [1, 4, 512, 64], "double_h_halve_w"),
    ],
    ids=["merge_ch", "swap_hw", "halve_w_double_h", "double_h_halve_w"],
)
@pytest.mark.parametrize(
    "strategy",
    [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK],
    ids=["height", "width", "block"],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["TILE", "ROW_MAJOR"])
def test_reshape_sharded_to_sharded(device, input_shape, output_shape, case_id, strategy, layout):
    """Both input and output are sharded with the same strategy."""
    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output = torch_input.reshape(output_shape)

    input_mem_cfg = make_sharded_memory_config(device, input_shape, strategy, layout)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=input_mem_cfg,
    )

    output_mem_cfg = make_sharded_memory_config(device, output_shape, strategy, layout)
    tt_output = ttnn.reshape(tt_input, output_shape, memory_config=output_mem_cfg)
    actual = ttnn.to_torch(tt_output)

    assert list(actual.shape) == list(
        torch_output.shape
    ), f"Shape mismatch: got {list(actual.shape)}, expected {list(torch_output.shape)}"
    assert torch.equal(torch_output, actual), "Data mismatch: reshape should preserve values exactly"


# ---------------------------------------------------------------------------
# Edge cases: non-4D shapes (rank 2 and rank 3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_shape,output_shape",
    [
        ([256, 128], [128, 256]),  # 2D swap
        ([256, 128], [1, 256, 128]),  # 2D -> 3D, zero-cost view
        ([256, 128], [256, 2, 64]),  # 2D -> 3D with dim change (non-zero-cost)
    ],
    ids=["2d_swap", "2d_to_3d_view", "2d_to_3d_dim_change"],
)
@pytest.mark.parametrize(
    "strategy",
    [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK],
    ids=["height", "width", "block"],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["TILE", "ROW_MAJOR"])
def test_reshape_non_4d_sharded_input(device, input_shape, output_shape, strategy, layout):
    """Non-4D tensor with sharded input; 2D->3D case forces real data movement."""
    mem_cfg = make_sharded_memory_config(device, input_shape, strategy, layout)
    _run_reshape(device, input_shape, output_shape, mem_cfg, layout)


# ---------------------------------------------------------------------------
# Irregular (non-tile-aligned) shapes with interleaved I/O: validates the
# reshape scheme can change dimensionality outside the zero-cost view path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_shape,output_shape",
    [
        ([50, 100], [50, 2, 50]),  # 2D -> 3D, non-tile-aligned dims
        ([50, 100], [100, 50]),  # 2D swap, non-tile-aligned
        ([50, 2, 50], [50, 100]),  # 3D -> 2D, non-tile-aligned
    ],
    ids=["irregular_2d_to_3d", "irregular_2d_swap", "irregular_3d_to_2d"],
)
def test_reshape_irregular_interleaved(device, input_shape, output_shape):
    """Irregular (non-tile-aligned) shapes with DRAM interleaved I/O."""
    _run_reshape(device, input_shape, output_shape, ttnn.DRAM_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT)


# ---------------------------------------------------------------------------
# Cross-strategy sharded-to-sharded: different shard strategy for input/output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "in_strategy,out_strategy",
    [
        (ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH),
        (ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.BLOCK),
        (ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.HEIGHT),
        (ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK),
        (ttnn.ShardStrategy.BLOCK, ttnn.ShardStrategy.HEIGHT),
        (ttnn.ShardStrategy.BLOCK, ttnn.ShardStrategy.WIDTH),
    ],
    ids=["h_to_w", "h_to_b", "w_to_h", "w_to_b", "b_to_h", "b_to_w"],
)
@pytest.mark.parametrize(
    "input_shape,output_shape,case_id",
    [
        ([1, 4, 256, 128], [1, 1, 1024, 128], "view_like"),
        ([1, 4, 256, 128], [1, 4, 128, 256], "data_movement"),
        # Irregular aligned: odd multipliers (3, 9) with tile-aligned dims.
        ([1, 3, 96, 96], [1, 9, 32, 96], "irregular_aligned"),
    ],
    ids=["view_like", "data_movement", "irregular_aligned"],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["TILE", "ROW_MAJOR"])
def test_reshape_cross_strategy_sharded(device, in_strategy, out_strategy, input_shape, output_shape, case_id, layout):
    """Input and output use different sharding strategies."""
    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output = torch_input.reshape(output_shape)

    input_mem_cfg = make_sharded_memory_config(device, input_shape, in_strategy, layout)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=input_mem_cfg,
    )

    output_mem_cfg = make_sharded_memory_config(device, output_shape, out_strategy, layout)
    tt_output = ttnn.reshape(tt_input, output_shape, memory_config=output_mem_cfg)
    actual = ttnn.to_torch(tt_output)

    assert list(actual.shape) == list(
        torch_output.shape
    ), f"Shape mismatch: got {list(actual.shape)}, expected {list(torch_output.shape)}"
    assert torch.equal(torch_output, actual), "Data mismatch: reshape should preserve values exactly"


# ---------------------------------------------------------------------------
# Default output derivation: sharded input, no explicit output memory_config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy",
    [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK],
    ids=["height", "width", "block"],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["TILE", "ROW_MAJOR"])
def test_reshape_sharded_default_output(device, strategy, layout):
    """Sharded input without explicit output memory_config; system derives it."""
    input_shape = [1, 4, 256, 128]
    output_shape = [1, 1, 1024, 128]

    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output = torch_input.reshape(output_shape)

    input_mem_cfg = make_sharded_memory_config(device, input_shape, strategy, layout)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=input_mem_cfg,
    )

    tt_output = ttnn.reshape(tt_input, output_shape)
    actual = ttnn.to_torch(tt_output)

    assert list(actual.shape) == list(
        torch_output.shape
    ), f"Shape mismatch: got {list(actual.shape)}, expected {list(torch_output.shape)}"
    assert torch.equal(torch_output, actual), "Data mismatch: reshape should preserve values exactly"


# ---------------------------------------------------------------------------
# BFLOAT8_B dtype: the BF8 path retains s2i/i2s — verify correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_shape,output_shape,case_id",
    [
        ([1, 4, 256, 128], [1, 1, 1024, 128], "merge_ch"),
        ([1, 4, 256, 128], [1, 4, 128, 256], "swap_hw"),
        # Irregular aligned: odd multipliers (3, 9) with tile-aligned dims.
        ([1, 3, 96, 96], [1, 1, 288, 96], "irregular_aligned"),
    ],
    ids=["merge_ch", "swap_hw", "irregular_aligned"],
)
@pytest.mark.parametrize(
    "strategy",
    [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK],
    ids=["height", "width", "block"],
)
def test_reshape_bfloat8_b(device, input_shape, output_shape, case_id, strategy):
    """BFLOAT8_B uses the s2i/i2s path; verify it produces correct results."""
    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output = torch_input.reshape(output_shape)

    input_mem_cfg = make_sharded_memory_config(device, input_shape, strategy, ttnn.TILE_LAYOUT)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        device=device,
        memory_config=input_mem_cfg,
    )

    tt_output = ttnn.reshape(tt_input, output_shape)
    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)

    assert list(actual.shape) == list(
        torch_output.shape
    ), f"Shape mismatch: got {list(actual.shape)}, expected {list(torch_output.shape)}"
    assert_with_pcc(torch_output, actual, 0.99)


# ---------------------------------------------------------------------------
# Grid-reduction path: output shape forces recompute_shard_spec_for_output to
# shrink the grid; exercises the INTERLEAVED-fallback guard in reshape.cpp.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_shape,output_shape",
    [
        ([1, 8, 32, 256], [1, 1, 256, 256]),
    ],
    ids=["grid_reduction"],
)
@pytest.mark.parametrize(
    "strategy",
    [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK],
    ids=["height", "width", "block"],
)
def test_reshape_grid_reduction(device, input_shape, output_shape, strategy):
    """Input grid is large relative to output dims, forcing grid reduction."""
    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output = torch_input.reshape(output_shape)

    input_mem_cfg = make_sharded_memory_config(device, input_shape, strategy, ttnn.TILE_LAYOUT)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=input_mem_cfg,
    )

    output_mem_cfg = make_sharded_memory_config(device, output_shape, strategy, ttnn.TILE_LAYOUT)
    tt_output = ttnn.reshape(tt_input, output_shape, memory_config=output_mem_cfg)
    actual = ttnn.to_torch(tt_output)

    assert list(actual.shape) == list(
        torch_output.shape
    ), f"Shape mismatch: got {list(actual.shape)}, expected {list(torch_output.shape)}"
    assert torch.equal(torch_output, actual), "Data mismatch: reshape should preserve values exactly"


# ---------------------------------------------------------------------------
# Irregular (non-tile-aligned) shapes on the TILE path, interleaved I/O.
# TILE companion to test_reshape_irregular_interleaved (RM).
# ---------------------------------------------------------------------------


IRREGULAR_RESHAPE_CASES = [
    ([50, 100], [50, 2, 50]),
    ([50, 100], [100, 50]),
    ([50, 2, 50], [50, 100]),
    ([1, 3, 50, 50], [1, 3, 25, 100]),
    ([1, 2, 96, 50], [1, 2, 50, 96]),
]
IRREGULAR_RESHAPE_IDS = [
    "2d_to_3d",
    "2d_swap",
    "3d_to_2d",
    "4d_halve_h_double_w",
    "4d_swap_hw",
]


@pytest.mark.parametrize(
    "input_shape,output_shape",
    IRREGULAR_RESHAPE_CASES,
    ids=IRREGULAR_RESHAPE_IDS,
)
def test_reshape_irregular_interleaved_tile(device, input_shape, output_shape):
    """Irregular shapes, TILE layout, DRAM interleaved I/O."""
    _run_reshape(device, input_shape, output_shape, ttnn.DRAM_MEMORY_CONFIG, ttnn.TILE_LAYOUT)


# ---------------------------------------------------------------------------
# Irregular shapes with sharded TILE input (shard geometry uses padded shape).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_shape,output_shape",
    IRREGULAR_RESHAPE_CASES,
    ids=IRREGULAR_RESHAPE_IDS,
)
@pytest.mark.parametrize(
    "strategy",
    [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK],
    ids=["height", "width", "block"],
)
def test_reshape_irregular_sharded_input_tile(device, input_shape, output_shape, strategy):
    """Irregular (non-tile-aligned) shapes, TILE layout, sharded input."""
    mem_cfg = make_sharded_memory_config(device, input_shape, strategy, ttnn.TILE_LAYOUT)
    _run_reshape(device, input_shape, output_shape, mem_cfg, ttnn.TILE_LAYOUT)


# ---------------------------------------------------------------------------
# Multi-dtype coverage (bf16 / fp32 / bfp8) on top of test_reshape_bfloat8_b.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_shape,output_shape,case_id",
    [
        ([1, 4, 256, 128], [1, 1, 1024, 128], "merge_ch"),
        ([1, 4, 256, 128], [1, 4, 128, 256], "swap_hw"),
    ],
    ids=["merge_ch", "swap_hw"],
)
@pytest.mark.parametrize(
    "strategy",
    [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK],
    ids=["height", "width", "block"],
)
@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b],
    ids=["bf16", "fp32", "bfp8"],
)
def test_reshape_multi_dtype(device, input_shape, output_shape, case_id, strategy, dtype):
    """Reshape across multiple dtypes with sharded TILE input."""
    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output = torch_input.reshape(output_shape)

    input_mem_cfg = make_sharded_memory_config(device, input_shape, strategy, ttnn.TILE_LAYOUT, dtype=dtype)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
        memory_config=input_mem_cfg,
    )

    tt_output = ttnn.reshape(tt_input, output_shape)
    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)

    assert list(actual.shape) == list(
        torch_output.shape
    ), f"Shape mismatch: got {list(actual.shape)}, expected {list(torch_output.shape)}"
    if dtype == ttnn.bfloat8_b:
        # BFP8 is lossy; use PCC.
        assert_with_pcc(torch_output, actual, 0.99)
    else:
        assert torch.equal(torch_output, actual), "Data mismatch: reshape should preserve values exactly"
