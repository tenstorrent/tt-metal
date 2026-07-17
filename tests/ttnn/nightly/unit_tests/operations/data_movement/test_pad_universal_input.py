# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Universal input support tests for ttnn.pad (issue #40407).

Tests all 5 memory configs (DRAM, L1, HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED)
and both layouts (TILE, ROW_MAJOR) as inputs

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

Ten test categories (matching test_reshape_universal_input.py structure):
  1. Interleaved inputs -> DRAM output         (baseline, should always pass)
  2. Sharded inputs -> DRAM output             (isolates reading from sharded input)
  3. DRAM input -> sharded output              (isolates writing to sharded output)
  4. Sharded input -> sharded output           (full sharded path, production use-case)
  5. Non-4D sharded inputs (rank-2, rank-3)
  6. RM W/B sharded width front-padding
  7. RM HEIGHT_SHARDED width front-padding (PadRmShardedWidthOnlyProgramFactory)
  8. DRAM W/B sharded input (to_memory_config composite fallback)
  9. Alt shape + float32 W/B sharded input     (accessor_page_size / shard_width coverage)
  10. Cross-strategy sharded -> sharded         (input/output shard geometry differ)
"""

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, make_sharded_memory_config


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
    # --- width-only padding (trailing) ---
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

_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
}

# Non-square base shape: different shard_width from [1,1,64,128] grid division.
ALT_SHAPE_PAD_CASES = [
    ([(0, 0), (0, 0), (0, 0), (0, 64)], [1, 1, 32, 256], [1, 1, 32, 320], "pad_w_small"),
    ([(0, 0), (0, 0), (0, 32), (0, 0)], [1, 1, 32, 256], [1, 1, 64, 256], "pad_h_small"),
]

# Single pad case for cross-strategy sharded output (exercises mismatched accessor_page_size).
CROSS_STRATEGY_PAD_CASE = (
    [(0, 0), (0, 0), (0, 0), (0, 64)],
    [1, 1, 64, 128],
    [1, 1, 64, 192],
    "pad_w_small",
)


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
    dtype=ttnn.bfloat16,
):
    """Run ttnn.pad and verify output against torch reference.

    output_memory_config=None -> DRAM interleaved output.
    Pass a sharded MemoryConfig to test the sharded-output path.
    """
    torch_dtype = _TORCH_DTYPE[dtype]
    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch_dtype)

    # torch.nn.functional.pad takes padding in reverse-dim order (innermost first)
    flat_pad = []
    for before, after in reversed(padding_spec):
        flat_pad.extend([before, after])
    torch_output = torch.nn.functional.pad(torch_input, flat_pad, value=0.0)

    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        dtype=dtype,
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
# Category 4: Sharded input -> sharded output (same strategy, recomputed spec)
#   Full sharded-to-sharded pad - primary production use-case.
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


# ---------------------------------------------------------------------------
# Category 6: RM WIDTH/BLOCK sharded — width front-padding
#
# Width front-padding (non-zero left pad on W axis) is only supported by the
# RM pad kernel path.  TILE layout has a pre-existing TT_FATAL that rejects
# front padding.
# ---------------------------------------------------------------------------

FRONT_PAD_CASES = [
    # (padding_spec, input_shape, output_shape, test_id)
    ([(0, 0), (0, 0), (0, 0), (64, 0)], [1, 1, 64, 128], [1, 1, 64, 192], "front_pad_w"),
]


@pytest.mark.parametrize(
    "padding_spec,input_shape,output_shape,case_id",
    FRONT_PAD_CASES,
    ids=[c[3] for c in FRONT_PAD_CASES],
)
@pytest.mark.parametrize(
    "strategy,strat_id",
    [
        (ttnn.ShardStrategy.WIDTH, "width"),
        (ttnn.ShardStrategy.BLOCK, "block"),
    ],
    ids=["width", "block"],
)
def test_pad_rm_wb_sharded_front_pad_input(
    device, padding_spec, input_shape, output_shape, case_id, strategy, strat_id
):
    """RM WIDTH/BLOCK sharded input with width front-padding → DRAM output."""
    in_cfg = make_sharded_memory_config(device, input_shape, strategy, ttnn.ROW_MAJOR_LAYOUT)
    _run_pad(device, input_shape, padding_spec, output_shape, in_cfg, ttnn.ROW_MAJOR_LAYOUT)


@pytest.mark.parametrize(
    "padding_spec,input_shape,output_shape,case_id",
    FRONT_PAD_CASES,
    ids=[c[3] for c in FRONT_PAD_CASES],
)
@pytest.mark.parametrize(
    "strategy,strat_id",
    [
        (ttnn.ShardStrategy.WIDTH, "width"),
        (ttnn.ShardStrategy.BLOCK, "block"),
    ],
    ids=["width", "block"],
)
def test_pad_rm_wb_sharded_front_pad_to_sharded(
    device, padding_spec, input_shape, output_shape, case_id, strategy, strat_id
):
    """RM WIDTH/BLOCK sharded input with width front-padding → same sharded output."""
    in_cfg = make_sharded_memory_config(device, input_shape, strategy, ttnn.ROW_MAJOR_LAYOUT)
    out_cfg = make_sharded_memory_config(device, output_shape, strategy, ttnn.ROW_MAJOR_LAYOUT)
    _run_pad(
        device, input_shape, padding_spec, output_shape, in_cfg, ttnn.ROW_MAJOR_LAYOUT, output_memory_config=out_cfg
    )


# ---------------------------------------------------------------------------
# Category 7: RM HEIGHT_SHARDED — width front-padding
#
# front_pad_w duplicates FRONT_PAD_CASES shapes but uses HEIGHT sharding, so it
# routes through PadRmShardedWidthOnlyProgramFactory instead of the W/B RM path.
# ---------------------------------------------------------------------------

HEIGHT_FRONT_PAD_CASES = [
    # (padding_spec, input_shape, output_shape, test_id)
    ([(0, 0), (0, 0), (0, 0), (64, 0)], [1, 1, 64, 128], [1, 1, 64, 192], "front_pad_w"),
    ([(0, 0), (0, 0), (0, 0), (32, 32)], [1, 1, 32, 64], [1, 1, 32, 128], "front_pad_w_both"),
]


@pytest.mark.parametrize(
    "padding_spec,input_shape,output_shape,case_id",
    HEIGHT_FRONT_PAD_CASES,
    ids=[c[3] for c in HEIGHT_FRONT_PAD_CASES],
)
def test_pad_height_sharded_front_pad_to_sharded(device, padding_spec, input_shape, output_shape, case_id):
    """HEIGHT_SHARDED RM input with width front-padding → HEIGHT_SHARDED output."""
    in_cfg = make_sharded_memory_config(device, input_shape, ttnn.ShardStrategy.HEIGHT, ttnn.ROW_MAJOR_LAYOUT)
    out_cfg = make_sharded_memory_config(device, output_shape, ttnn.ShardStrategy.HEIGHT, ttnn.ROW_MAJOR_LAYOUT)
    _run_pad(
        device, input_shape, padding_spec, output_shape, in_cfg, ttnn.ROW_MAJOR_LAYOUT, output_memory_config=out_cfg
    )


@pytest.mark.parametrize(
    "padding_spec,input_shape,output_shape,case_id",
    HEIGHT_FRONT_PAD_CASES,
    ids=[c[3] for c in HEIGHT_FRONT_PAD_CASES],
)
def test_pad_height_sharded_front_pad_to_dram(device, padding_spec, input_shape, output_shape, case_id):
    """HEIGHT_SHARDED RM input with width front-padding → DRAM output (default factory path)."""
    in_cfg = make_sharded_memory_config(device, input_shape, ttnn.ShardStrategy.HEIGHT, ttnn.ROW_MAJOR_LAYOUT)
    _run_pad(device, input_shape, padding_spec, output_shape, in_cfg, ttnn.ROW_MAJOR_LAYOUT)


@pytest.mark.parametrize(
    "padding_spec,input_shape,output_shape,case_id",
    [HEIGHT_FRONT_PAD_CASES[0]],  # front_pad_w only
    ids=["front_pad_w"],
)
def test_pad_float32_height_sharded_front_pad_to_sharded(device, padding_spec, input_shape, output_shape, case_id):
    """float32 HEIGHT_SHARDED RM front-pad → sharded (element_size=4 W front-pad bytes)."""
    in_cfg = make_sharded_memory_config(
        device, input_shape, ttnn.ShardStrategy.HEIGHT, ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32
    )
    out_cfg = make_sharded_memory_config(
        device, output_shape, ttnn.ShardStrategy.HEIGHT, ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32
    )
    _run_pad(
        device,
        input_shape,
        padding_spec,
        output_shape,
        in_cfg,
        ttnn.ROW_MAJOR_LAYOUT,
        output_memory_config=out_cfg,
        dtype=ttnn.float32,
    )


# ---------------------------------------------------------------------------
# Category 8: DRAM WIDTH/BLOCK sharded input (to_memory_config composite fallback)
#
# needs_pad_composite_fallback routes DRAM-sharded W/B inputs through
# to_memory_config -> pad. L1 sharded tests above exercise the native path.
# WIDTH sharded DRAM is used here; BLOCK DRAM tensor creation fails because
# make_sharded_memory_config maps compute cores to DRAM banks (see test below).
# TODO(#41610): add BLOCK DRAM end-to-end test once DRAM block shard grids are
# supported by the test helper / host validation.
# ---------------------------------------------------------------------------

DRAM_WB_FALLBACK_CASES = [
    ([(0, 0), (0, 0), (0, 0), (0, 64)], [1, 1, 64, 128], [1, 1, 64, 192], "pad_w_small"),
]


@pytest.mark.parametrize(
    "padding_spec,input_shape,output_shape,case_id",
    DRAM_WB_FALLBACK_CASES,
    ids=[c[3] for c in DRAM_WB_FALLBACK_CASES],
)
def test_pad_dram_wb_sharded_input_composite_fallback(device, padding_spec, input_shape, output_shape, case_id):
    """DRAM WIDTH sharded input -> DRAM output via to_memory_config composite fallback."""
    in_cfg = make_sharded_memory_config(
        device,
        input_shape,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )
    _run_pad(device, input_shape, padding_spec, output_shape, in_cfg, ttnn.ROW_MAJOR_LAYOUT)


def test_pad_dram_block_sharded_composite_fallback_unsupported(device, expect_error):
    """BLOCK DRAM also hits needs_pad_composite_fallback but cannot be host-uploaded today.

    make_sharded_memory_config builds a compute-core grid; DRAM block sharding needs
    DRAM bank coordinates, so from_torch fails before pad runs.
    """
    in_cfg = make_sharded_memory_config(
        device,
        [1, 1, 64, 128],
        ttnn.ShardStrategy.BLOCK,
        ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )
    torch_input = torch.randn([1, 1, 64, 128], dtype=torch.bfloat16)
    with expect_error(RuntimeError, "Logical DRAM core"):
        ttnn.from_torch(
            torch_input,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            device=device,
            memory_config=in_cfg,
        )


# ---------------------------------------------------------------------------
# Category 9: Alt shape + float32 W/B sharded input
#
# accessor_page_size = shard_width * element_size in the RM factory. Exercises
# non-default shard_width (alt aspect ratio) and element_size=4 (float32).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "padding_spec,input_shape,output_shape,case_id",
    ALT_SHAPE_PAD_CASES,
    ids=[c[3] for c in ALT_SHAPE_PAD_CASES],
)
@pytest.mark.parametrize(
    "strategy,strat_id",
    [
        (ttnn.ShardStrategy.WIDTH, "width"),
        (ttnn.ShardStrategy.BLOCK, "block"),
    ],
    ids=["width", "block"],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["TILE", "ROW_MAJOR"])
def test_pad_alt_shape_wb_sharded_input(
    device, padding_spec, input_shape, output_shape, case_id, strategy, strat_id, layout
):
    """Non-square [1,1,32,256] W/B sharded input -> DRAM output."""
    mem_cfg = make_sharded_memory_config(device, input_shape, strategy, layout)
    _run_pad(device, input_shape, padding_spec, output_shape, mem_cfg, layout)


@pytest.mark.parametrize(
    "padding_spec,input_shape,output_shape,case_id",
    [PAD_CASES[2]],  # pad_w_small only
    ids=["pad_w_small"],
)
@pytest.mark.parametrize(
    "strategy,strat_id",
    [
        (ttnn.ShardStrategy.WIDTH, "width"),
        (ttnn.ShardStrategy.BLOCK, "block"),
    ],
    ids=["width", "block"],
)
def test_pad_float32_wb_sharded_input(device, padding_spec, input_shape, output_shape, case_id, strategy, strat_id):
    """float32 RM W/B sharded input -> DRAM output (element_size=4 accessor_page_size)."""
    mem_cfg = make_sharded_memory_config(device, input_shape, strategy, ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
    _run_pad(
        device,
        input_shape,
        padding_spec,
        output_shape,
        mem_cfg,
        ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.float32,
    )


# ---------------------------------------------------------------------------
# Category 10: Cross-strategy sharded input -> sharded output
#
# Production configs may pad between different shard strategies; reader and writer
# must use independent input/output accessor_page_size values.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "in_strategy,out_strategy,cross_id",
    [
        (ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH, "height_to_width"),
        (ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.HEIGHT, "width_to_height"),
    ],
    ids=["height_to_width", "width_to_height"],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["TILE", "ROW_MAJOR"])
def test_pad_cross_strategy_sharded_to_sharded(device, in_strategy, out_strategy, cross_id, layout):
    """Cross-strategy sharded input -> sharded output (mismatched shard geometries)."""
    padding_spec, input_shape, output_shape, _case_id = CROSS_STRATEGY_PAD_CASE
    in_cfg = make_sharded_memory_config(device, input_shape, in_strategy, layout)
    out_cfg = make_sharded_memory_config(device, output_shape, out_strategy, layout)
    _run_pad(device, input_shape, padding_spec, output_shape, in_cfg, layout, output_memory_config=out_cfg)
