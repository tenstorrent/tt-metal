# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, make_sharded_memory_config
from tests.ttnn.utils_for_testing import assert_reshape as _assert_reshape


# ---------------------------------------------------------------------------
# Memory-config scenario builders.
# Each scenario is (label, in_builder, out_builder); builders take
# (device, shape, layout, dtype) and return a MemoryConfig (or None = default).
# ---------------------------------------------------------------------------


def _mc_interleaved(mc):
    return lambda device, shape, layout, dtype: mc


def _mc_sharded(strategy):
    return lambda device, shape, layout, dtype: make_sharded_memory_config(device, shape, strategy, layout, dtype=dtype)


def _mc_none():
    return lambda device, shape, layout, dtype: None


HEIGHT = ttnn.ShardStrategy.HEIGHT
WIDTH = ttnn.ShardStrategy.WIDTH
BLOCK = ttnn.ShardStrategy.BLOCK

SCENARIOS = [
    # interleaved input, default output (system derives)
    ("dram_default", _mc_interleaved(ttnn.DRAM_MEMORY_CONFIG), _mc_none()),
    ("l1_default", _mc_interleaved(ttnn.L1_MEMORY_CONFIG), _mc_none()),
    # sharded input, default output
    ("height_default", _mc_sharded(HEIGHT), _mc_none()),
    ("width_default", _mc_sharded(WIDTH), _mc_none()),
    ("block_default", _mc_sharded(BLOCK), _mc_none()),
    # DRAM input, sharded output
    ("dram_height", _mc_interleaved(ttnn.DRAM_MEMORY_CONFIG), _mc_sharded(HEIGHT)),
    ("dram_width", _mc_interleaved(ttnn.DRAM_MEMORY_CONFIG), _mc_sharded(WIDTH)),
    ("dram_block", _mc_interleaved(ttnn.DRAM_MEMORY_CONFIG), _mc_sharded(BLOCK)),
    # L1 input, sharded output (#46161: close L1→sharded test gap)
    ("l1_height", _mc_interleaved(ttnn.L1_MEMORY_CONFIG), _mc_sharded(HEIGHT)),
    ("l1_width", _mc_interleaved(ttnn.L1_MEMORY_CONFIG), _mc_sharded(WIDTH)),
    ("l1_block", _mc_interleaved(ttnn.L1_MEMORY_CONFIG), _mc_sharded(BLOCK)),
    # sharded-to-sharded, same strategy
    ("height_height", _mc_sharded(HEIGHT), _mc_sharded(HEIGHT)),
    ("width_width", _mc_sharded(WIDTH), _mc_sharded(WIDTH)),
    ("block_block", _mc_sharded(BLOCK), _mc_sharded(BLOCK)),
]
SCENARIO_IDS = [s[0] for s in SCENARIOS]


# ---------------------------------------------------------------------------
# Shape cases. Each entry: (input_shape, output_shape, case_id).
# Covers: tile-aligned 4D, non-4D rank-2/3, irregular (non-tile-aligned),
# an aligned-odd-multiplier case, and a grid-reduction case.
# ---------------------------------------------------------------------------


SHAPE_CASES = [
    # tile-aligned 4D
    ([1, 4, 256, 128], [1, 1, 1024, 128], "merge_ch"),
    ([1, 4, 256, 128], [1, 4, 128, 256], "swap_hw"),
    ([1, 4, 128, 256], [1, 4, 256, 128], "halve_w_double_h"),
    ([1, 4, 256, 128], [1, 4, 512, 64], "double_h_halve_w"),
    # non-4D
    ([256, 128], [128, 256], "2d_swap"),
    ([256, 128], [1, 256, 128], "2d_to_3d_view"),
    ([256, 128], [256, 2, 64], "2d_to_3d_dim_change"),
    # irregular (non-tile-aligned last two dims)
    ([50, 100], [50, 2, 50], "irreg_2d_to_3d"),
    ([50, 100], [100, 50], "irreg_2d_swap"),
    ([50, 2, 50], [50, 100], "irreg_3d_to_2d"),
    ([1, 3, 50, 50], [1, 3, 25, 100], "irreg_4d"),
    ([1, 2, 96, 50], [1, 2, 50, 96], "irreg_4d_swap"),
    # aligned odd multiplier
    ([1, 3, 96, 96], [1, 9, 32, 96], "irreg_aligned"),
    # grid reduction: forces recompute_shard_spec_for_output to shrink the grid
    ([1, 8, 32, 256], [1, 1, 256, 256], "grid_reduction"),
    # odd batch sizes (#46161: inner dims are tile-aligned and L1-aligned)
    ([3, 1, 64, 64], [1, 3, 64, 64], "odd_batch_3"),
    ([7, 32, 32], [1, 7, 32, 32], "odd_batch_7"),
    ([2, 3, 64, 64], [6, 1, 64, 64], "odd_batch_2x3"),
    ([3, 64, 128], [3, 128, 64], "odd_batch_swap_hw"),
]
SHAPE_IDS = [s[2] for s in SHAPE_CASES]

_IRREGULAR_CASE_IDS = {
    "irreg_2d_to_3d",
    "irreg_2d_swap",
    "irreg_3d_to_2d",
    "irreg_4d",
    "irreg_4d_swap",
}
_GRID_REDUCTION_CASE_ID = "grid_reduction"

LAYOUTS = [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]
LAYOUT_IDS = ["TILE", "RM"]

DTYPES = [
    (ttnn.bfloat16, "bf16"),
    (ttnn.float32, "fp32"),
    (ttnn.bfloat8_b, "bfp8"),
    (ttnn.bfloat4_b, "bfp4"),
]
DTYPE_IDS = [d[1] for d in DTYPES]

# Per-shape scenario allowlist — matches the semantic coverage of the
# pre-consolidation test file (i.e. the union of what the 16 separate
# test_reshape_* functions exercised, minus redundant duplicates),
# plus #46161 expansions for L1→sharded, irregular+sharded output,
# and odd batch sizes.
_ALL_SCENARIOS = set(SCENARIO_IDS)
_SHARDED_IN_DEFAULT_OUT = {"height_default", "width_default", "block_default"}
_SAME_STRATEGY_S2S = {"height_height", "width_width", "block_block"}
_INTERLEAVED_TO_SHARDED = {
    "dram_height",
    "dram_width",
    "dram_block",
    "l1_height",
    "l1_width",
    "l1_block",
}

_SHAPE_SCENARIO_ALLOWLIST = {
    # Tile-aligned 4D: full scenario set (includes L1→sharded from #46161).
    "merge_ch": _ALL_SCENARIOS,
    "swap_hw": _ALL_SCENARIOS,
    "halve_w_double_h": _ALL_SCENARIOS,
    "double_h_halve_w": _ALL_SCENARIOS,
    # Non-4D: only sharded_in × default_out (as in test_reshape_non_4d_sharded_input).
    "2d_swap": _SHARDED_IN_DEFAULT_OUT,
    "2d_to_3d_view": _SHARDED_IN_DEFAULT_OUT,
    "2d_to_3d_dim_change": _SHARDED_IN_DEFAULT_OUT,
    # Irregular: expanded to include TILE sharded-output scenarios (#46161).
    # RM remains blocked for irregular shapes by the _is_valid filter (non-L1-aligned widths).
    "irreg_2d_to_3d": {"dram_default"} | _SHARDED_IN_DEFAULT_OUT | _INTERLEAVED_TO_SHARDED | _SAME_STRATEGY_S2S,
    "irreg_2d_swap": {"dram_default"} | _SHARDED_IN_DEFAULT_OUT | _INTERLEAVED_TO_SHARDED | _SAME_STRATEGY_S2S,
    "irreg_3d_to_2d": {"dram_default"} | _SHARDED_IN_DEFAULT_OUT | _INTERLEAVED_TO_SHARDED | _SAME_STRATEGY_S2S,
    "irreg_4d": {"dram_default"} | _SHARDED_IN_DEFAULT_OUT | _INTERLEAVED_TO_SHARDED | _SAME_STRATEGY_S2S,
    "irreg_4d_swap": {"dram_default"} | _SHARDED_IN_DEFAULT_OUT | _INTERLEAVED_TO_SHARDED | _SAME_STRATEGY_S2S,
    # irreg_aligned in the old file was only exercised via cross_strategy
    # (separate test) and via test_reshape_bfloat8_b. Restrict main-test
    # coverage to the sharded-in-default scenarios for the bfp8 dtype only
    # (see dtype filter below).
    "irreg_aligned": _SHARDED_IN_DEFAULT_OUT,
    # grid_reduction: same-strategy sharded-to-sharded only.
    "grid_reduction": _SAME_STRATEGY_S2S,
    # Odd batch sizes (#46161): inner dims are tile-aligned and L1-aligned,
    # so all scenarios work for both TILE and RM.
    "odd_batch_3": _ALL_SCENARIOS,
    "odd_batch_swap_hw": _ALL_SCENARIOS,
    "odd_batch_7": _SHARDED_IN_DEFAULT_OUT | {"dram_default", "l1_default"},
    "odd_batch_2x3": _SHARDED_IN_DEFAULT_OUT | {"dram_default", "l1_default"},
}

# dtype axis restrictions matching the pre-consolidation test_reshape_bfloat8_b
# and test_reshape_multi_dtype coverage (sharded_in × default_out, TILE only).
_FP32_CASE_IDS = {"merge_ch", "swap_hw"}
_BFP8_CASE_IDS = {"merge_ch", "swap_hw", "irreg_aligned"}
_BFP4_CASE_IDS = _BFP8_CASE_IDS  # bf4 mirrors bf8 scope exactly


def _is_valid(scenario_label, case_id, layout, dtype):
    """Return True if (scenario, shape, layout, dtype) is in-scope.

    Allowlist based on the pre-consolidation coverage plus #46161
    expansions (L1→sharded, irregular+sharded output, odd batch).
    """
    # Per-shape scenario allowlist
    if scenario_label not in _SHAPE_SCENARIO_ALLOWLIST.get(case_id, set()):
        return False
    is_irregular = case_id in _IRREGULAR_CASE_IDS
    # Layout constraints
    if case_id == _GRID_REDUCTION_CASE_ID and layout != ttnn.TILE_LAYOUT:
        return False
    if case_id == "irreg_aligned" and layout != ttnn.TILE_LAYOUT:
        return False
    # Irregular RM only existed for the first three cases (dram interleaved).
    if is_irregular and layout == ttnn.ROW_MAJOR_LAYOUT:
        if scenario_label != "dram_default":
            return False
        if case_id in ("irreg_4d", "irreg_4d_swap"):
            return False
    # dtype constraints — matches old bfp8 + multi_dtype scope.
    if dtype == ttnn.float32:
        if case_id not in _FP32_CASE_IDS:
            return False
        if scenario_label not in _SHARDED_IN_DEFAULT_OUT:
            return False
        if layout != ttnn.TILE_LAYOUT:
            return False
    if dtype == ttnn.bfloat8_b:
        if case_id not in _BFP8_CASE_IDS:
            return False
        if scenario_label not in _SHARDED_IN_DEFAULT_OUT:
            return False
        if layout != ttnn.TILE_LAYOUT:
            return False
    if dtype == ttnn.bfloat4_b:
        if case_id not in _BFP4_CASE_IDS:
            return False
        if scenario_label not in _SHARDED_IN_DEFAULT_OUT:
            return False
        if layout != ttnn.TILE_LAYOUT:
            return False
    # bf16 does NOT run on irreg_aligned in the main test (only cross_strategy + bfp8).
    if dtype == ttnn.bfloat16 and case_id == "irreg_aligned":
        return False
    return True


def _enumerate_cases():
    """Build explicit pytest.params for every valid (shape, scenario, layout, dtype) tuple."""
    for input_shape, output_shape, case_id in SHAPE_CASES:
        for scenario in SCENARIOS:
            label = scenario[0]
            for layout, layout_id in zip(LAYOUTS, LAYOUT_IDS):
                for dtype, dtype_id in DTYPES:
                    if not _is_valid(label, case_id, layout, dtype):
                        continue
                    yield pytest.param(
                        input_shape,
                        output_shape,
                        case_id,
                        scenario,
                        layout,
                        (dtype, dtype_id),
                        id=f"{dtype_id}-{layout_id}-{label}-{case_id}",
                    )


_TEST_RESHAPE_CASES = list(_enumerate_cases())


# ---------------------------------------------------------------------------
# Main test: interleaved and single-strategy sharded combinations.
# Parametrize axes:
#   shape × scenario × layout × dtype, with invalid combos skipped dynamically.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_shape,output_shape,case_id,scenario,layout,dtype_spec",
    _TEST_RESHAPE_CASES,
)
def test_reshape(device, input_shape, output_shape, case_id, scenario, layout, dtype_spec):
    """Reshape across all supported input/output memory configs, layouts, and dtypes.

    Covers interleaved I/O (DRAM/L1), sharded input (HEIGHT/WIDTH/BLOCK) with
    default or explicit sharded output, sharded-to-sharded with the same
    strategy, non-4D and irregular shapes, and the grid-reduction edge case.
    """
    _, in_builder, out_builder = scenario
    dtype, _ = dtype_spec

    in_memcfg = in_builder(device, input_shape, layout, dtype)
    out_memcfg = out_builder(device, output_shape, layout, dtype)

    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output = torch_input.reshape(output_shape)

    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        dtype=dtype,
        device=device,
        memory_config=in_memcfg,
    )

    if out_memcfg is None:
        tt_output = ttnn.reshape(tt_input, output_shape)
    else:
        tt_output = ttnn.reshape(tt_input, output_shape, memory_config=out_memcfg)

    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)
    _assert_reshape(torch_output, actual, dtype)


# ---------------------------------------------------------------------------
# Cross-strategy sharded-to-sharded: input and output use different strategies.
# Separate function because it has a different parameter signature
# (two strategies vs one scenario object).
# ---------------------------------------------------------------------------


CROSS_STRATEGY_PAIRS = [
    (HEIGHT, WIDTH, "h_to_w"),
    (HEIGHT, BLOCK, "h_to_b"),
    (WIDTH, HEIGHT, "w_to_h"),
    (WIDTH, BLOCK, "w_to_b"),
    (BLOCK, HEIGHT, "b_to_h"),
    (BLOCK, WIDTH, "b_to_w"),
]

CROSS_STRATEGY_SHAPES = [
    ([1, 4, 256, 128], [1, 1, 1024, 128], "view_like"),
    ([1, 4, 256, 128], [1, 4, 128, 256], "data_movement"),
    ([1, 3, 96, 96], [1, 9, 32, 96], "irregular_aligned"),
]


@pytest.mark.parametrize(
    "input_shape,output_shape,case_id",
    CROSS_STRATEGY_SHAPES,
    ids=[c[2] for c in CROSS_STRATEGY_SHAPES],
)
@pytest.mark.parametrize(
    "in_strategy,out_strategy,pair_id",
    CROSS_STRATEGY_PAIRS,
    ids=[p[2] for p in CROSS_STRATEGY_PAIRS],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_reshape_cross_strategy(device, input_shape, output_shape, case_id, in_strategy, out_strategy, pair_id, layout):
    """Sharded input and output with different shard strategies."""
    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output = torch_input.reshape(output_shape)

    in_memcfg = make_sharded_memory_config(device, input_shape, in_strategy, layout)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=in_memcfg,
    )

    out_memcfg = make_sharded_memory_config(device, output_shape, out_strategy, layout)
    tt_output = ttnn.reshape(tt_input, output_shape, memory_config=out_memcfg)
    actual = ttnn.to_torch(tt_output)
    _assert_reshape(torch_output, actual, ttnn.bfloat16)


# Reshape on a sharded TILE input must keep the input shard grid. The per-core
# shape may round up to tile alignment, but the grid itself must never be
# silently shrunk.


@pytest.mark.parametrize(
    "input_shape, output_shape, expected_out_shard_shape",
    [
        # 64 cores, phys_h=320 -> per-core 5 rows padded up to 32 (6.4x waste).
        ((640, 32), (320, 64), [32, 64]),
        # Same setup with phys_h=160 -> per-core 3 rows padded to 32 (10x waste).
        ((640, 32), (160, 128), [32, 128]),
    ],
    ids=["overpad_320", "deeper_overpad_160"],
)
def test_reshape_height_sharded_preserves_input_grid_when_alignment_wastes(
    device, input_shape, output_shape, expected_out_shard_shape
):
    """Reshape on a HEIGHT_SHARDED TILE input must preserve the input grid and
    round each per-core shape up to tile alignment, even when phys_h/num_cores
    is not already tile-aligned. Data must match torch bit-for-bit on bf16.
    """
    core_grid_size = device.compute_with_storage_grid_size()
    if core_grid_size.x < 8 or core_grid_size.y < 8:
        pytest.skip("requires at least 8x8 core grid")

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})

    # 64 cores, phys_h=input_shape[0] tile-aligned -> shard_h = 32 per core.
    input_shard_shape = [32, input_shape[1]]
    in_shard_spec = ttnn.ShardSpec(grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in_memcfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttnn.BufferType.L1, shard_spec=in_shard_spec
    )

    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output = torch_input.reshape(output_shape)

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_memcfg
    )

    tt_output = ttnn.reshape(tt_input, output_shape)

    out_memcfg = tt_output.memory_config()
    assert (
        out_memcfg.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    ), f"reshape changed output memory layout to {out_memcfg.memory_layout} (expected HEIGHT_SHARDED)"
    assert (
        out_memcfg.shard_spec.grid == grid
    ), f"reshape silently changed the output shard grid ({grid} -> {out_memcfg.shard_spec.grid})"
    assert (
        list(out_memcfg.shard_spec.shape) == expected_out_shard_shape
    ), f"output per-core shard shape {list(out_memcfg.shard_spec.shape)} != expected {expected_out_shard_shape}"

    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)
    _assert_reshape(torch_output, actual, ttnn.bfloat16)


def test_reshape_width_sharded_preserves_input_grid_when_alignment_wastes(device):
    """Symmetric WIDTH_SHARDED case: the input grid must be preserved and the
    per-core width rounded up to tile alignment.
    """
    core_grid_size = device.compute_with_storage_grid_size()
    if core_grid_size.x < 8 or core_grid_size.y < 8:
        pytest.skip("requires at least 8x8 core grid")

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})

    # 8 cores, phys (32, 256) -> shard_w = 32 per core (clean).
    input_shape = (32, 256)
    input_shard_shape = [32, 32]
    in_shard_spec = ttnn.ShardSpec(grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in_memcfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, buffer_type=ttnn.BufferType.L1, shard_spec=in_shard_spec
    )

    # phys_w=32 on 8 cores -> per-core 4 cols padded to 32 (8x waste).
    output_shape = (256, 32)
    expected_out_shard_shape = [256, 32]

    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output = torch_input.reshape(output_shape)

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_memcfg
    )

    tt_output = ttnn.reshape(tt_input, output_shape)

    out_memcfg = tt_output.memory_config()
    assert (
        out_memcfg.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    ), f"reshape changed output memory layout to {out_memcfg.memory_layout} (expected WIDTH_SHARDED)"
    assert (
        out_memcfg.shard_spec.grid == grid
    ), f"reshape silently changed the output shard grid ({grid} -> {out_memcfg.shard_spec.grid})"
    assert (
        list(out_memcfg.shard_spec.shape) == expected_out_shard_shape
    ), f"output per-core shard shape {list(out_memcfg.shard_spec.shape)} != expected {expected_out_shard_shape}"

    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)
    _assert_reshape(torch_output, actual, ttnn.bfloat16)


# Auto-derive shard_spec: caller pins the output layout but leaves shard_spec
# unset. reshape should reuse the input's shard_spec as the seed grid rather
# than raising "Shard spec has no value".


@pytest.mark.parametrize(
    "layout_name, in_shard_shape, in_grid_xy, input_shape, output_shape, expected_out_shard_shape",
    [
        # HEIGHT: phys_h=3584 on 56 cores -> ceil(3584/56)=64, rounded to tile -> 64 rows per core.
        ("HEIGHT_SHARDED", [32, 64], (7, 6), (1, 1, 1792, 64), (1, 1, 3584, 32), [64, 32]),
        # WIDTH: phys_w=128 on 8 cores -> ceil(128/8)=16, rounded to tile -> 32 cols per core.
        ("WIDTH_SHARDED", [32, 32], (7, 0), (1, 1, 32, 256), (1, 1, 64, 128), [64, 32]),
        # BLOCK: phys (128,512) on 4x4 -> find_best_n_1d keeps full 4x4 -> 32x128 per core.
        ("BLOCK_SHARDED", [64, 64], (3, 3), (1, 1, 256, 256), (1, 1, 128, 512), [32, 128]),
    ],
    ids=["height", "width", "block"],
)
def test_reshape_sharded_memory_config_without_shard_spec_autoderives_from_input(
    device, layout_name, in_shard_shape, in_grid_xy, input_shape, output_shape, expected_out_shard_shape
):
    """Public API: ttnn.reshape(t, shape, memory_config=MemoryConfig(layout, L1))
    (no shard_spec) must succeed by seeding from the input's shard_spec and
    derive a valid output shard_spec. Data must match torch bit-for-bit.
    """
    core_grid_size = device.compute_with_storage_grid_size()
    if core_grid_size.x <= in_grid_xy[0] or core_grid_size.y <= in_grid_xy[1]:
        pytest.skip(f"requires at least {in_grid_xy[0] + 1}x{in_grid_xy[1] + 1} core grid")

    layout = getattr(ttnn.TensorMemoryLayout, layout_name)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(*in_grid_xy))})
    in_shard_spec = ttnn.ShardSpec(grid, in_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in_memcfg = ttnn.MemoryConfig(layout, buffer_type=ttnn.BufferType.L1, shard_spec=in_shard_spec)

    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output = torch_input.reshape(output_shape)

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_memcfg
    )

    out_memcfg_no_spec = ttnn.MemoryConfig(layout, buffer_type=ttnn.BufferType.L1)
    assert out_memcfg_no_spec.shard_spec is None, "test setup: out_memcfg should not carry a shard_spec"

    tt_output = ttnn.reshape(tt_input, output_shape, memory_config=out_memcfg_no_spec)

    out_memcfg = tt_output.memory_config()
    assert out_memcfg.is_sharded(), "auto-derived output should remain sharded"
    assert out_memcfg.memory_layout == layout, f"layout changed to {out_memcfg.memory_layout}"
    assert out_memcfg.shard_spec is not None, "auto-derived output must have a shard_spec"
    assert out_memcfg.shard_spec.grid == grid, "auto-derived output should reuse the input grid"
    assert (
        list(out_memcfg.shard_spec.shape) == expected_out_shard_shape
    ), f"derived shard shape {list(out_memcfg.shard_spec.shape)} != expected {expected_out_shard_shape}"

    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)
    _assert_reshape(torch_output, actual, ttnn.bfloat16)


def test_reshape_layout_only_sharded_output_without_input_shard_spec_fails(device, expect_error):
    """Interleaved input + layout-only sharded output memory_config must TT_FATAL
    with an actionable message when no input shard_spec is available to seed from.
    """
    torch_input = torch.randn(1, 1, 256, 256, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttnn.BufferType.L1)
    assert out_mc.shard_spec is None

    with expect_error(RuntimeError, "no input_shard_spec is available"):
        ttnn.reshape(tt_input, (1, 1, 512, 128), memory_config=out_mc)


# Input from an ND shard spec that normalizes to a 2D layout keeps the higher-rank nd_shard_spec
# attached; a rank-lowering reshape used to abort in BufferDistributionSpec on the shard-rank check.


@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
@pytest.mark.parametrize(
    "nd_shard_shape, strategy_name",
    [
        ([1, 1, 32, 128], "height"),  # splits the flattened height -> HEIGHT_SHARDED
        ([1, 1, 64, 64], "width"),  # splits the width -> WIDTH_SHARDED
    ],
    ids=["nd_height", "nd_width"],
)
def test_reshape_nd_shard_spec_normalized_to_2d_input(device, layout, nd_shard_shape, strategy_name):
    """An input whose MemoryConfig was built from a rank-4 NdShardSpec that
    normalizes to a 2D layout must reshape to a lower-rank shape without tripping
    the shard-rank check, and preserve values exactly."""
    input_shape = [1, 1, 64, 128]
    output_shape = [1, 128, 64]  # rank 3 < the retained rank-4 nd_shard_spec

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape(nd_shard_shape), grid=grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    in_memcfg = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output = torch_input.reshape(output_shape)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=in_memcfg)

    # Precondition the bug depends on: a config built from a rank-4 NdShardSpec normalizes to a 2D
    # HEIGHT/WIDTH-sharded layout while still carrying the rank-4 nd_shard_spec. Assert it here so
    # that if TensorSpec normalization ever changes, this test flags that it no longer reproduces
    # the reported failure instead of silently passing.
    in_mc = tt_input.memory_config()
    expected_layout = (
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED if strategy_name == "height" else ttnn.TensorMemoryLayout.WIDTH_SHARDED
    )
    assert in_mc.memory_layout == expected_layout, f"expected {expected_layout}, got {in_mc.memory_layout}"
    assert in_mc.nd_shard_spec is not None, "input should retain the higher-rank nd_shard_spec"
    assert len(in_mc.nd_shard_spec.shard_shape) == 4, "retained nd_shard_spec should be rank 4"

    tt_output = ttnn.reshape(tt_input, output_shape, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)
    _assert_reshape(torch_output, actual, ttnn.bfloat16)


# ND-sharded output config: the internal sharded paths only handle a 2D shard_spec, so an ND output
# used to abort. Covered for both a spec that normalizes to 2D and one that stays genuinely ND.


@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
@pytest.mark.parametrize(
    "input_shape, output_shape, nd_shard_shape, is_genuine_nd",
    [
        # normalizes to a 2D layout once applied to the output tensor
        ([1, 1, 64, 128], [1, 1, 128, 64], [1, 1, 64, 64], False),
        # genuinely ND (batch split) — stays ND_SHARDED
        ([2, 2, 64, 64], [2, 2, 32, 128], [1, 1, 64, 64], True),
        # same-shape (no-op) reshape must still apply the ND config: it runs before the generic
        # no-op early return, which otherwise returns the unchanged interleaved input.
        ([1, 1, 64, 128], [1, 1, 64, 128], [1, 1, 64, 64], False),
    ],
    ids=["normalizes_2d", "genuine_nd", "same_shape"],
)
def test_reshape_nd_sharded_output(device, layout, input_shape, output_shape, nd_shard_shape, is_genuine_nd):
    """Reshape with an ND-sharded output memory_config must succeed and preserve
    values, whether the ND spec normalizes to a 2D layout, stays ND, or leaves the
    shape unchanged (no-op that must still honor the requested ND config)."""
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape(nd_shard_shape), grid=grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    out_memcfg = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output = torch_input.reshape(output_shape)

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_output = ttnn.reshape(tt_input, output_shape, memory_config=out_memcfg)

    out_mc = tt_output.memory_config()
    assert out_mc.is_sharded()
    if is_genuine_nd:
        # Guard against the silent fallback to a 2D HEIGHT/WIDTH-sharded output with a derived 2D
        # spec: an explicitly-requested ND distribution must come back ND_SHARDED with the exact
        # nd_shard_spec that was asked for.
        assert (
            out_mc.memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED
        ), f"expected ND_SHARDED, got {out_mc.memory_layout}"
        assert out_mc.nd_shard_spec is not None
        assert list(out_mc.nd_shard_spec.shard_shape) == list(
            out_memcfg.nd_shard_spec.shard_shape
        ), f"output nd shard shape {list(out_mc.nd_shard_spec.shard_shape)} != requested {list(out_memcfg.nd_shard_spec.shard_shape)}"
    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)
    _assert_reshape(torch_output, actual, ttnn.bfloat16)


@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_reshape_nd_sharded_input_rank_change_inherited(device, layout):
    """A genuinely ND-sharded input reshaped to a different rank with an inherited (not
    explicit) output config must adapt the shard shape to the new rank instead of handing
    BufferDistributionSpec a stale shard_shape it would abort on."""
    input_shape = [2, 2, 64, 64]
    output_shape = [4, 64, 64]  # rank 4 -> rank 3

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape([1, 1, 64, 64]), grid=grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    in_memcfg = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output = torch_input.reshape(output_shape)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=in_memcfg)
    assert tt_input.memory_config().memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED

    tt_output = ttnn.reshape(tt_input, output_shape)

    out_mc = tt_output.memory_config()
    assert (
        out_mc.memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED
    ), f"expected ND_SHARDED output, got {out_mc.memory_layout}"
    assert out_mc.nd_shard_spec is not None
    assert out_mc.nd_shard_spec.shard_shape.rank == len(output_shape)

    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)
    _assert_reshape(torch_output, actual, ttnn.bfloat16)


@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_reshape_nd_low_rank_shard_spec_inherited(device, layout):
    """A lower-rank ND shard spec (shard rank < tensor rank, which is legal) reshaped with an
    inherited config to a rank below the shard rank must re-derive an output-rank shard spec
    instead of carrying the stale shard rank onto the output and re-tripping the rank abort."""
    input_shape = [2, 2, 64, 64]
    output_shape = [16384]  # rank 4 -> rank 1, below the rank-2 shard

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape([64, 64]), grid=grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    in_memcfg = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output = torch_input.reshape(output_shape)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=in_memcfg)
    assert tt_input.memory_config().memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED
    assert len(tt_input.memory_config().nd_shard_spec.shard_shape) == 2

    tt_output = ttnn.reshape(tt_input, output_shape)

    assert tt_output.memory_config().is_sharded()
    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)
    _assert_reshape(torch_output, actual, ttnn.bfloat16)


@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
@pytest.mark.parametrize("out_kind", ["interleaved", "height_sharded_2d"], ids=["out_interleaved", "out_2d_sharded"])
def test_reshape_nd_input_explicit_non_nd_output(device, layout, out_kind):
    """A genuinely ND-sharded input with an explicit non-ND output config (interleaved or
    2D-sharded) must stage through interleaved like the ND-output case, not abort in the
    view_device 'View is not supported for ND sharded tensors' path."""
    input_shape = [2, 2, 64, 64]
    output_shape = [2, 2, 32, 128]

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape([1, 1, 64, 64]), grid=grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    in_memcfg = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    if out_kind == "interleaved":
        out_memcfg = ttnn.DRAM_MEMORY_CONFIG
    else:
        # HEIGHT_SHARDED: total height 2*2*32=128 over 2 cores -> [64, 128].
        out_memcfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [64, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )

    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output = torch_input.reshape(output_shape)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=in_memcfg)
    assert tt_input.memory_config().memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED

    tt_output = ttnn.reshape(tt_input, output_shape, memory_config=out_memcfg)

    assert tt_output.memory_config().memory_layout != ttnn.TensorMemoryLayout.ND_SHARDED
    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)
    _assert_reshape(torch_output, actual, ttnn.bfloat16)


@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
@pytest.mark.parametrize("explicit_output", [False, True], ids=["default_output", "explicit_output"])
def test_reshape_nd_sharded_input(device, layout, explicit_output):
    """A genuinely ND-sharded input must reshape and preserve values whether the output
    memory_config is inherited from the input (default) or passed explicitly."""
    input_shape = [2, 2, 64, 64]
    output_shape = [2, 2, 32, 128]

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape([1, 1, 64, 64]), grid=grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    in_memcfg = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output = torch_input.reshape(output_shape)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=in_memcfg)
    assert tt_input.memory_config().memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED

    if explicit_output:
        tt_output = ttnn.reshape(tt_input, output_shape, memory_config=in_memcfg)
    else:
        tt_output = ttnn.reshape(tt_input, output_shape)

    out_mc = tt_output.memory_config()
    assert (
        out_mc.memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED
    ), f"expected ND_SHARDED output, got {out_mc.memory_layout}"
    assert out_mc.nd_shard_spec is not None
    if explicit_output:
        assert list(out_mc.nd_shard_spec.shard_shape) == list(
            in_memcfg.nd_shard_spec.shard_shape
        ), f"explicit output shard shape {list(out_mc.nd_shard_spec.shard_shape)} != requested {list(in_memcfg.nd_shard_spec.shard_shape)}"
    else:
        assert list(out_mc.nd_shard_spec.shard_shape) == [
            1,
            1,
            32,
            128,
        ], f"derived output shard shape {list(out_mc.nd_shard_spec.shard_shape)} != expected [1, 1, 32, 128]"

    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)
    _assert_reshape(torch_output, actual, ttnn.bfloat16)


# A normalized ND config passed as an *explicit* output config skips the ND branch
# (is_nd_sharded_memory_config is false once a 2D shard_spec is present) and reaches the 2D
# explicit-override path, which returned it verbatim - stale nd_shard_spec included.
@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_reshape_explicit_normalized_nd_memory_config(device, layout):
    """A tensor-derived config that normalized from an NdShardSpec to BLOCK_SHARDED carries both
    specs. Handing it to a rank-changing reshape as an explicit memory_config must not carry the
    higher-rank nd_shard_spec into the output's spec, which aborted buffer allocation with
    "Tensor shape rank (3) can't be less than shard shape rank (4)!"."""
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
    # Default ROUND_ROBIN_1D on purpose: GRID_2D rejects a rank-4 shard shape outright
    # ("2D grid distribution is only supported for 2D sharding!"), so the donor tensor below
    # would not allocate at all and the test would never reach the path it covers.
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape([1, 1, 32, 32]), grid=grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    donor = ttnn.from_torch(
        torch.zeros([1, 1, 64, 64], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=layout,
        device=device,
        memory_config=ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec),
    )
    explicit_mc = donor.memory_config()

    # Preconditions the bug depends on: normalization produced a 2D layout that still carries the
    # rank-4 nd_shard_spec. If normalization changes, fail loudly rather than pass silently.
    assert explicit_mc.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED, explicit_mc.memory_layout
    assert explicit_mc.shard_spec is not None, "normalized config should carry a 2D shard_spec"
    assert explicit_mc.nd_shard_spec is not None, "normalized config should retain the nd_shard_spec"
    assert len(explicit_mc.nd_shard_spec.shard_shape) == 4, "retained nd_shard_spec should be rank 4"

    input_shape = [1, 1, 64, 64]
    output_shape = [1, 2, 32, 64]  # reshape_tiled squeezes this to rank 3, below the nd rank

    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output = torch_input.reshape(output_shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    tt_output = ttnn.reshape(tt_input, output_shape, memory_config=explicit_mc)

    out_mc = tt_output.memory_config()
    assert out_mc.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED, out_mc.memory_layout
    # The stale higher-rank spec must be gone; an auto-derived rank-<=2 one is fine. Bound at 2, not
    # at the output rank: the output is rank 4 here, so `<= len(tt_output.shape)` would be satisfied
    # by the exact rank-4 spec this test exists to rule out.
    assert (
        out_mc.nd_shard_spec is None or len(out_mc.nd_shard_spec.shard_shape) <= 2
    ), f"stale nd_shard_spec survived: rank {len(out_mc.nd_shard_spec.shard_shape)}"

    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)
    _assert_reshape(torch_output, actual, ttnn.bfloat16)


# MemoryConfig::operator== is provenance-sensitive, so a same-shape reshape whose requested config
# describes the input's own distribution used to fall through to a full reshard.
@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_reshape_same_shape_functionally_identical_config_is_noop(device, layout):
    """A same-shape reshape asked for the input's own distribution, expressed as a plain 2D config
    rather than the input's normalized-from-ND one, must stay a no-op instead of resharding into a
    byte-identical layout."""
    shape = [1, 1, 64, 128]
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape([1, 1, 32, 128]), grid=grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )

    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=layout,
        device=device,
        memory_config=ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec),
    )

    in_mc = tt_input.memory_config()
    assert in_mc.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED, in_mc.memory_layout
    assert in_mc.nd_shard_spec is not None, "normalized config should retain the nd_shard_spec"

    # Same distribution, different provenance: no nd_shard_spec, so operator== reports a difference.
    equivalent_mc = ttnn.MemoryConfig(in_mc.memory_layout, ttnn.BufferType.L1, in_mc.shard_spec)
    tt_output = ttnn.reshape(tt_input, shape, memory_config=equivalent_mc)

    assert (
        tt_output.buffer_address() == tt_input.buffer_address()
    ), "a functionally identical config must not allocate a new buffer"

    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)
    _assert_reshape(torch_input.reshape(shape), actual, ttnn.bfloat16)


# view_device keeps a logical-only update on an ND tensor zero-copy; the ND staging branch must not
# intercept it and turn it into an L1 -> DRAM -> L1 round trip.
def test_reshape_nd_input_logical_only_update_is_view(device):
    """A genuinely ND-sharded input reshaped to a different logical shape with the same padded shape
    stays a metadata-only view (same buffer), not two device copies."""
    # Batch-split shard: splitting the leading dims (not the inner two) is what keeps the config
    # genuinely ND. A shard that only divides the height, e.g. [1, 1, 32, 64] on [1, 1, 64, 64],
    # flattens cleanly and TensorSpec normalizes it to HEIGHT_SHARDED with a 2D shard_spec, which
    # would not exercise the ND staging branch at all.
    padded_shape = [2, 2, 64, 64]
    logical_shape = [2, 2, 60, 64]  # same padded tiles, fewer logical rows

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape([1, 1, 64, 64]), grid=grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )

    torch.manual_seed(0)
    torch_input = torch.randn(padded_shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec),
    )

    # Precondition: the input is genuinely ND (no 2D shard_spec), so it takes the ND staging branch.
    in_mc = tt_input.memory_config()
    assert (
        in_mc.memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED
    ), f"input should stay ND_SHARDED, got {in_mc.memory_layout}"
    assert in_mc.nd_shard_spec is not None, "input should be ND-sharded"
    assert in_mc.shard_spec is None, "a genuinely ND config should have no 2D shard_spec"

    tt_output = ttnn.reshape(tt_input, logical_shape, padded_shape)

    assert list(tt_output.shape) == logical_shape, f"got {list(tt_output.shape)}"
    assert (
        tt_output.buffer_address() == tt_input.buffer_address()
    ), "a logical-only update on an ND tensor must stay a view, not round-trip through DRAM"

    # Buffer identity alone would also hold if the gate returned a view for a reshape that must
    # move data, so check the values too. The padded shape and last dim are unchanged and only the
    # logical row count shrinks, so this is a crop of the leading rows.
    expected = torch_input[:, :, : logical_shape[-2], :]
    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)
    _assert_reshape(expected, actual, ttnn.bfloat16)


# The zero-copy gate above must not fire when the last dim changes: an equal padded shape alone
# does not make the reshape a reinterpretation, and returning a view there is silently wrong data.
def test_reshape_nd_input_last_dim_change_is_not_a_view(device):
    """Same padded shape but a changed logical last dim must go through the staging path and obey
    torch-reshape semantics, not return the input's bytes in place."""
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})  # 1 core
    # Shard must be tile-aligned for TILE layout; [1, 32, 32] over [2, 62, 32] gives 4 shards on
    # 1 core, so normalization is rejected and the config stays genuinely ND_SHARDED.
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape([1, 32, 32]), grid=grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )

    torch.manual_seed(0)
    # [2, 62, 32] and [2, 64, 31] both pad to [2, 64, 32] and have volume 3968, so the padded shape
    # and volume match while the last dim changes 32 -> 31.
    torch_input = torch.randn([2, 62, 32], dtype=torch.bfloat16)  # padded to [2, 64, 32] by TILE
    torch_output = torch_input.reshape([2, 64, 31])

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec),
    )
    in_mc = tt_input.memory_config()
    assert (
        in_mc.memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED
    ), f"input should stay ND_SHARDED, got {in_mc.memory_layout}"
    assert list(tt_input.padded_shape) == [2, 64, 32], f"got {list(tt_input.padded_shape)}"

    # Padded shapes match (both [2, 64, 32]) and the volume is unchanged (3968), so only the
    # last-dim check keeps this off the view path.
    tt_output = ttnn.reshape(tt_input, [2, 64, 31], [2, 64, 32])

    assert list(tt_output.shape) == [2, 64, 31], f"got {list(tt_output.shape)}"
    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)
    _assert_reshape(torch_output, actual, ttnn.bfloat16)
