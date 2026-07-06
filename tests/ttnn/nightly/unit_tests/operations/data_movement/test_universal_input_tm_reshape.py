# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Element size (bytes) for each ttnn dtype we exercise in this file.
_DTYPE_ELEM_SIZE = {
    ttnn.bfloat16: 2,
    ttnn.float32: 4,
    ttnn.uint32: 4,
    ttnn.int32: 4,
    ttnn.uint16: 2,
    ttnn.uint8: 1,
    ttnn.bfloat8_b: 1,
}


def _divisible_grid_1d(total_dim, max_cores, step):
    """Return the largest n <= max_cores such that total_dim % (n * step) == 0.

    Raises:
        ValueError: If no valid n exists (i.e. total_dim is not a multiple of step).
    """
    for n in range(max_cores, 0, -1):
        if total_dim % (n * step) == 0:
            return n
    raise ValueError(f"No valid 1D grid size for total_dim={total_dim}, max_cores={max_cores}, step={step}")


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


def _assert_reshape(torch_output, actual, dtype):
    assert list(actual.shape) == list(
        torch_output.shape
    ), f"Shape mismatch: got {list(actual.shape)}, expected {list(torch_output.shape)}"
    if dtype == ttnn.bfloat8_b:
        assert_with_pcc(torch_output, actual, 0.99)
    else:
        assert torch.equal(torch_output, actual), "Data mismatch: reshape should preserve values exactly"


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


def test_reshape_layout_only_sharded_output_without_input_shard_spec_fails(device):
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

    with pytest.raises(RuntimeError, match="no input_shard_spec is available"):
        ttnn.reshape(tt_input, (1, 1, 512, 128), memory_config=out_mc)
