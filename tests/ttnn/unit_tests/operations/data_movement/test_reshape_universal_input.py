# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


# ---------------------------------------------------------------------------
# Helper: build a shard memory config for a given shape, strategy and layout
# ---------------------------------------------------------------------------


def _divisible_grid_1d(total_dim, max_cores, step):
    """Find largest n <= max_cores such that total_dim is divisible by n*step."""
    for n in range(max_cores, 0, -1):
        if total_dim % (n * step) == 0:
            return n
    return 1


def make_sharded_memory_config(device, shape, strategy, layout):
    """Create a valid sharded MemoryConfig for `shape` on `device`.

    For TILE layout the shard dims must be tile-aligned (multiples of 32).
    For ROW_MAJOR the shard width × element_size must satisfy the device L1
    alignment; for bfloat16 (2 bytes/element):
      - Wormhole: L1 alignment = 128 bytes → shard width multiple of 64
      - Blackhole: L1 alignment = 64 bytes  → shard width multiple of 32
    """
    grid = device.compute_with_storage_grid_size()
    max_x, max_y = grid.x, grid.y
    tile_h, tile_w = 32, 32

    # Flatten all dims except the last two into a single H for shard purposes
    total_h = 1
    for d in shape[:-1]:
        total_h *= d
    total_w = shape[-1]

    step_h = tile_h if layout == ttnn.TILE_LAYOUT else 1
    # L1 alignment varies by arch (128 bytes on WH, 64 bytes on BH).
    # For bfloat16 (2 bytes/elem): shard_width must be a multiple of
    # alignment_bytes / element_size so that shard_width * 2 % alignment == 0.
    l1_alignment = 128 if ttnn.device.is_wormhole_b0(device) else 64
    element_size = 2  # bfloat16
    rm_step_w = l1_alignment // element_size  # 64 on WH, 32 on BH
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
        shape=shape,
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
    # (input_shape, output_shape, test_id)
    # view-like: one case covers the PerformView fast path (last dim unchanged)
    ([1, 4, 256, 128], [1, 1, 1024, 128], "merge_ch"),
    # data-movement (last dim changes): each case targets a distinct s2i→reshape→i2s bug
    #   swap_hw: width shards exceed grid columns
    #   halve_w_double_h: height shards exceed grid rows
    #   double_h_halve_w: BLOCK output grid recomputation
    ([1, 4, 256, 128], [1, 4, 128, 256], "swap_hw"),
    ([1, 4, 128, 256], [1, 4, 256, 128], "halve_w_double_h"),
    ([1, 4, 256, 128], [1, 4, 512, 64], "double_h_halve_w"),
]

SHARDING_STRATEGIES = [
    ttnn.ShardStrategy.HEIGHT,
    ttnn.ShardStrategy.WIDTH,
    ttnn.ShardStrategy.BLOCK,
]

MEMORY_CONFIGS = ["DRAM", "L1", "HEIGHT_SHARDED", "WIDTH_SHARDED", "BLOCK_SHARDED"]

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
    "strategy,strat_id",
    [
        (ttnn.ShardStrategy.HEIGHT, "height"),
        (ttnn.ShardStrategy.WIDTH, "width"),
        (ttnn.ShardStrategy.BLOCK, "block"),
    ],
    ids=["height", "width", "block"],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["TILE", "ROW_MAJOR"])
def test_reshape_sharded_output(device, input_shape, output_shape, case_id, strategy, strat_id, layout):
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
        # Includes both view-like (PerformView fast path) and data-movement (s2i→reshape→i2s) cases.
        ([1, 4, 256, 128], [1, 1, 1024, 128], "merge_ch"),
        ([1, 4, 256, 128], [1, 4, 128, 256], "swap_hw"),
        ([1, 4, 256, 128], [1, 4, 512, 64], "double_h_halve_w"),
    ],
    ids=["merge_ch", "swap_hw", "double_h_halve_w"],
)
@pytest.mark.parametrize(
    "strategy,strat_id",
    [
        (ttnn.ShardStrategy.HEIGHT, "height"),
        (ttnn.ShardStrategy.WIDTH, "width"),
        (ttnn.ShardStrategy.BLOCK, "block"),
    ],
    ids=["height", "width", "block"],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["TILE", "ROW_MAJOR"])
def test_reshape_sharded_to_sharded(device, input_shape, output_shape, case_id, strategy, strat_id, layout):
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
        ([256, 128], [1, 256, 128]),  # 2D -> 3D
    ],
    ids=["2d_swap", "2d_to_3d"],
)
@pytest.mark.parametrize(
    "strategy",
    [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK],
    ids=["height", "width", "block"],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["TILE", "ROW_MAJOR"])
def test_reshape_non_4d_sharded_input(device, input_shape, output_shape, strategy, layout):
    """Non-4D tensor with sharded input."""
    mem_cfg = make_sharded_memory_config(device, input_shape, strategy, layout)
    _run_reshape(device, input_shape, output_shape, mem_cfg, layout)
