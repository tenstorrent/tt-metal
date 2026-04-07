# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Universal input support tests for ttnn.pad (issue #40407).

Tests all 5 memory configs (DRAM, L1, HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED)
and both layouts (TILE, ROW_MAJOR) as inputs — mirroring the thoroughness of
test_reshape_universal_input.py.

Goal: ttnn.pad should accept any input memory config and produce correct results.
No xfail/skip decorators — failures indicate real gaps to be fixed.

Padding scenarios (all with base input shape [1,1,64,128]):
  - pad_h_small:    pad H by 32              [1,1,64,128]  -> [1,1, 96,128]
  - pad_h_large:    pad H by 64              [1,1,64,128]  -> [1,1,128,128]
  - pad_w_small:    pad W by 64              [1,1,64,128]  -> [1,1, 64,192]
  - pad_w_large:    pad W by 128             [1,1,64,128]  -> [1,1, 64,256]
  - pad_both_small: pad H by 32 and W by 64  [1,1,64,128]  -> [1,1, 96,192]
  - pad_both_large: pad H by 64 and W by 128 [1,1,64,128]  -> [1,1,128,256]
  - pad_c_only:     pad channel by 1         [1,1,64,128]  -> [1,2, 64,128]

Code paths exercised:
  - pad_h_*     : height-only padding (PadRmShardedHeightOnlyProgramFactory for RM HEIGHT_SHARDED)
  - pad_w_*     : width-only  padding (PadRmShardedWidthOnlyProgramFactory  for RM HEIGHT_SHARDED)
  - pad_both_*  : H+W combined → 2-pass decomposition in pad.cpp
  - pad_c_only  : upper-dim only (no H/W change)

All padding amounts are tile-aligned (multiples of 32).
All output widths are multiples of 64 elements (L1-aligned for bfloat16 on Wormhole)
so sharded output configs are valid for both TILE and ROW_MAJOR layouts.

Five test categories (matching test_reshape_universal_input.py structure):
  1. Interleaved inputs → DRAM output         (baseline, should always pass)
  2. Sharded inputs → DRAM output             (isolates reading from sharded input)
  3. DRAM input → sharded output              (isolates writing to sharded output)
  4. Sharded input → sharded output           (full sharded path, production use-case)
  5. Non-4D sharded inputs (rank-2, rank-3)
"""

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


# ---------------------------------------------------------------------------
# Helper: build a shard memory config for a given shape, strategy and layout
# (identical to the helper in test_reshape_universal_input.py)
# ---------------------------------------------------------------------------


def _divisible_grid_1d(total_dim, max_cores, step):
    """Find largest n <= max_cores such that total_dim is divisible by n*step."""
    for n in range(max_cores, 0, -1):
        if total_dim % (n * step) == 0:
            return n
    return 1


def make_sharded_memory_config(device, shape, strategy, layout):
    """Create a valid sharded MemoryConfig for `shape` on `device`.

    For TILE layout shard dims must be tile-aligned (multiples of 32).
    For ROW_MAJOR the shard width × element_size must satisfy the device L1
    alignment (128 bytes on Wormhole); for bfloat16 that means shard width
    must be a multiple of 64 elements.
    """
    grid = device.compute_with_storage_grid_size()
    max_x, max_y = grid.x, grid.y
    tile_h, tile_w = 32, 32

    total_h = 1
    for d in shape[:-1]:
        total_h *= d
    total_w = shape[-1]

    step_h = tile_h if layout == ttnn.TILE_LAYOUT else 1
    step_w = tile_w if layout == ttnn.TILE_LAYOUT else 64

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
# Pad scenarios: (padding_spec, input_shape, output_shape, test_id)
#
# All amounts are multiples of 32 (tile-aligned).
# All output widths are multiples of 64 (L1-aligned for bfloat16 on Wormhole).
# All output heights are multiples of 32.
# ---------------------------------------------------------------------------

PAD_CASES = [
    # (padding_spec, input_shape, output_shape, test_id)
    # --- height-only padding ---
    ([(0, 0), (0, 0), (0, 32), (0, 0)], [1, 1, 64, 128], [1, 1, 96, 128], "pad_h_small"),
    ([(0, 0), (0, 0), (0, 64), (0, 0)], [1, 1, 64, 128], [1, 1, 128, 128], "pad_h_large"),
    # --- width-only padding ---
    ([(0, 0), (0, 0), (0, 0), (0, 64)], [1, 1, 64, 128], [1, 1, 64, 192], "pad_w_small"),
    ([(0, 0), (0, 0), (0, 0), (0, 128)], [1, 1, 64, 128], [1, 1, 64, 256], "pad_w_large"),
    # --- both H and W (triggers 2-pass decomposition in pad.cpp) ---
    ([(0, 0), (0, 0), (0, 32), (0, 64)], [1, 1, 64, 128], [1, 1, 96, 192], "pad_both_small"),
    ([(0, 0), (0, 0), (0, 64), (0, 128)], [1, 1, 64, 128], [1, 1, 128, 256], "pad_both_large"),
    # --- upper-dim only (no H/W change, channel grows) ---
    ([(0, 0), (0, 1), (0, 0), (0, 0)], [1, 1, 64, 128], [1, 2, 64, 128], "pad_c_only"),
]

# Non-4D pad cases: rank-2 and rank-3 inputs
NON_4D_PAD_CASES = [
    # (padding_spec, input_shape, output_shape, test_id)
    ([(0, 32), (0, 0)], [64, 128], [96, 128], "2d_pad_h"),
    ([(0, 0), (0, 64)], [64, 128], [64, 192], "2d_pad_w"),
    ([(0, 0), (0, 32), (0, 64)], [1, 64, 128], [1, 96, 192], "3d_pad_both"),
]

SHARDING_STRATEGIES = [
    ttnn.ShardStrategy.HEIGHT,
    ttnn.ShardStrategy.WIDTH,
    ttnn.ShardStrategy.BLOCK,
]

LAYOUTS = [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]


# ---------------------------------------------------------------------------
# Core test runner
# ---------------------------------------------------------------------------


def _run_pad(
    device,
    input_shape,
    padding_spec,
    expected_output_shape,
    input_memory_config,
    layout,
    output_memory_config=None,
):
    """Run ttnn.pad and verify output against torch reference.

    output_memory_config=None → DRAM interleaved output.
    Pass a sharded MemoryConfig to test the sharded-output path.
    """
    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    # torch.nn.functional.pad takes padding in reverse-dim order (innermost first)
    flat_pad = []
    for before, after in reversed(padding_spec):
        flat_pad.extend([before, after])
    torch_output = torch.nn.functional.pad(torch_input, flat_pad, value=0.0)

    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=input_memory_config,
    )

    out_cfg = output_memory_config if output_memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    tt_output = ttnn.pad(tt_input, padding_spec, value=0.0, memory_config=out_cfg)
    actual = ttnn.to_torch(tt_output)

    assert list(actual.shape) == list(
        expected_output_shape
    ), f"Shape mismatch: got {list(actual.shape)}, expected {list(expected_output_shape)}"
    assert_with_pcc(torch_output, actual, pcc=0.9999)


# ---------------------------------------------------------------------------
# Category 1: Interleaved inputs → DRAM output  (baseline, must always pass)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "padding_spec,input_shape,output_shape,case_id",
    PAD_CASES,
    ids=[c[3] for c in PAD_CASES],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["TILE", "ROW_MAJOR"])
def test_pad_dram_input(device, padding_spec, input_shape, output_shape, case_id, layout):
    """Baseline: DRAM interleaved input — should always pass."""
    _run_pad(device, input_shape, padding_spec, output_shape, ttnn.DRAM_MEMORY_CONFIG, layout)


@pytest.mark.parametrize(
    "padding_spec,input_shape,output_shape,case_id",
    PAD_CASES,
    ids=[c[3] for c in PAD_CASES],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["TILE", "ROW_MAJOR"])
def test_pad_l1_input(device, padding_spec, input_shape, output_shape, case_id, layout):
    """L1 interleaved input."""
    _run_pad(device, input_shape, padding_spec, output_shape, ttnn.L1_MEMORY_CONFIG, layout)


# ---------------------------------------------------------------------------
# Category 2: Sharded inputs → DRAM output
#   Verifies that the op can correctly READ from each sharding type.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "padding_spec,input_shape,output_shape,case_id",
    PAD_CASES,
    ids=[c[3] for c in PAD_CASES],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["TILE", "ROW_MAJOR"])
def test_pad_height_sharded_input(device, padding_spec, input_shape, output_shape, case_id, layout):
    """HEIGHT_SHARDED input → DRAM output."""
    mem_cfg = make_sharded_memory_config(device, input_shape, ttnn.ShardStrategy.HEIGHT, layout)
    _run_pad(device, input_shape, padding_spec, output_shape, mem_cfg, layout)


@pytest.mark.parametrize(
    "padding_spec,input_shape,output_shape,case_id",
    PAD_CASES,
    ids=[c[3] for c in PAD_CASES],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["TILE", "ROW_MAJOR"])
def test_pad_width_sharded_input(device, padding_spec, input_shape, output_shape, case_id, layout):
    """WIDTH_SHARDED input → DRAM output."""
    mem_cfg = make_sharded_memory_config(device, input_shape, ttnn.ShardStrategy.WIDTH, layout)
    _run_pad(device, input_shape, padding_spec, output_shape, mem_cfg, layout)


@pytest.mark.parametrize(
    "padding_spec,input_shape,output_shape,case_id",
    PAD_CASES,
    ids=[c[3] for c in PAD_CASES],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["TILE", "ROW_MAJOR"])
def test_pad_block_sharded_input(device, padding_spec, input_shape, output_shape, case_id, layout):
    """BLOCK_SHARDED input → DRAM output."""
    mem_cfg = make_sharded_memory_config(device, input_shape, ttnn.ShardStrategy.BLOCK, layout)
    _run_pad(device, input_shape, padding_spec, output_shape, mem_cfg, layout)


# ---------------------------------------------------------------------------
# Category 3: DRAM input → sharded output
#   Verifies that the op can correctly WRITE to each sharding type.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "padding_spec,input_shape,output_shape,case_id",
    PAD_CASES,
    ids=[c[3] for c in PAD_CASES],
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
def test_pad_sharded_output(device, padding_spec, input_shape, output_shape, case_id, strategy, strat_id, layout):
    """DRAM interleaved input → sharded output."""
    out_cfg = make_sharded_memory_config(device, output_shape, strategy, layout)
    _run_pad(
        device, input_shape, padding_spec, output_shape, ttnn.DRAM_MEMORY_CONFIG, layout, output_memory_config=out_cfg
    )


# ---------------------------------------------------------------------------
# Category 4: Sharded input → sharded output (same strategy, recomputed spec)
#   Full sharded-to-sharded pad — primary production use-case.
#   Source of the H/W/N gaps in unary.md.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "padding_spec,input_shape,output_shape,case_id",
    PAD_CASES,
    ids=[c[3] for c in PAD_CASES],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["TILE", "ROW_MAJOR"])
def test_pad_height_sharded_to_sharded(device, padding_spec, input_shape, output_shape, case_id, layout):
    """HEIGHT_SHARDED input → HEIGHT_SHARDED output."""
    in_cfg = make_sharded_memory_config(device, input_shape, ttnn.ShardStrategy.HEIGHT, layout)
    out_cfg = make_sharded_memory_config(device, output_shape, ttnn.ShardStrategy.HEIGHT, layout)
    _run_pad(device, input_shape, padding_spec, output_shape, in_cfg, layout, output_memory_config=out_cfg)


@pytest.mark.parametrize(
    "padding_spec,input_shape,output_shape,case_id",
    PAD_CASES,
    ids=[c[3] for c in PAD_CASES],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["TILE", "ROW_MAJOR"])
def test_pad_width_sharded_to_sharded(device, padding_spec, input_shape, output_shape, case_id, layout):
    """WIDTH_SHARDED input → WIDTH_SHARDED output."""
    in_cfg = make_sharded_memory_config(device, input_shape, ttnn.ShardStrategy.WIDTH, layout)
    out_cfg = make_sharded_memory_config(device, output_shape, ttnn.ShardStrategy.WIDTH, layout)
    _run_pad(device, input_shape, padding_spec, output_shape, in_cfg, layout, output_memory_config=out_cfg)


@pytest.mark.parametrize(
    "padding_spec,input_shape,output_shape,case_id",
    PAD_CASES,
    ids=[c[3] for c in PAD_CASES],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["TILE", "ROW_MAJOR"])
def test_pad_block_sharded_to_sharded(device, padding_spec, input_shape, output_shape, case_id, layout):
    """BLOCK_SHARDED input → BLOCK_SHARDED output."""
    in_cfg = make_sharded_memory_config(device, input_shape, ttnn.ShardStrategy.BLOCK, layout)
    out_cfg = make_sharded_memory_config(device, output_shape, ttnn.ShardStrategy.BLOCK, layout)
    _run_pad(device, input_shape, padding_spec, output_shape, in_cfg, layout, output_memory_config=out_cfg)


# ---------------------------------------------------------------------------
# Category 5: Non-4D sharded inputs (rank-2 and rank-3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "padding_spec,input_shape,output_shape,case_id",
    NON_4D_PAD_CASES,
    ids=[c[3] for c in NON_4D_PAD_CASES],
)
@pytest.mark.parametrize(
    "strategy",
    SHARDING_STRATEGIES,
    ids=["height", "width", "block"],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["TILE", "ROW_MAJOR"])
def test_pad_non_4d_sharded_input(device, padding_spec, input_shape, output_shape, case_id, strategy, layout):
    """Non-4D tensor (rank-2 or rank-3) with sharded input → DRAM output."""
    mem_cfg = make_sharded_memory_config(device, input_shape, strategy, layout)
    _run_pad(device, input_shape, padding_spec, output_shape, mem_cfg, layout)
