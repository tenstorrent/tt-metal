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
# test_reshape_* functions exercised, minus redundant duplicates).
_ALL_SCENARIOS = set(SCENARIO_IDS)
_SHARDED_IN_DEFAULT_OUT = {"height_default", "width_default", "block_default"}
_SAME_STRATEGY_S2S = {"height_height", "width_width", "block_block"}

_SHAPE_SCENARIO_ALLOWLIST = {
    # Tile-aligned 4D: full scenario set (as in the original dram/l1/
    # height/width/block_sharded_input + sharded_output + sharded_to_sharded tests).
    "merge_ch": _ALL_SCENARIOS,
    "swap_hw": _ALL_SCENARIOS,
    "halve_w_double_h": _ALL_SCENARIOS,
    "double_h_halve_w": _ALL_SCENARIOS,
    # Non-4D: only sharded_in × default_out (as in test_reshape_non_4d_sharded_input).
    "2d_swap": _SHARDED_IN_DEFAULT_OUT,
    "2d_to_3d_view": _SHARDED_IN_DEFAULT_OUT,
    "2d_to_3d_dim_change": _SHARDED_IN_DEFAULT_OUT,
    # Irregular: dram interleaved + sharded_in × default_out
    # (as in test_reshape_irregular_interleaved{,_tile} + irregular_sharded_input_tile).
    "irreg_2d_to_3d": {"dram_default"} | _SHARDED_IN_DEFAULT_OUT,
    "irreg_2d_swap": {"dram_default"} | _SHARDED_IN_DEFAULT_OUT,
    "irreg_3d_to_2d": {"dram_default"} | _SHARDED_IN_DEFAULT_OUT,
    # The last two irregular shapes were TILE-only originally (no RM case).
    "irreg_4d": {"dram_default"} | _SHARDED_IN_DEFAULT_OUT,
    "irreg_4d_swap": {"dram_default"} | _SHARDED_IN_DEFAULT_OUT,
    # irreg_aligned in the old file was only exercised via cross_strategy
    # (separate test) and via test_reshape_bfloat8_b. Restrict main-test
    # coverage to the sharded-in-default scenarios for the bfp8 dtype only
    # (see dtype filter below).
    "irreg_aligned": _SHARDED_IN_DEFAULT_OUT,
    # grid_reduction: same-strategy sharded-to-sharded only.
    "grid_reduction": _SAME_STRATEGY_S2S,
}

# dtype axis restrictions matching the pre-consolidation test_reshape_bfloat8_b
# and test_reshape_multi_dtype coverage (sharded_in × default_out, TILE only).
_FP32_CASE_IDS = {"merge_ch", "swap_hw"}
_BFP8_CASE_IDS = {"merge_ch", "swap_hw", "irreg_aligned"}


def _is_valid(scenario_label, case_id, layout, dtype):
    """Return True if (scenario, shape, layout, dtype) is in-scope.

    Strict allowlist that reproduces the pre-consolidation coverage —
    we do not expand the cross product beyond what the original 16
    test_reshape_* functions already exercised.
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
