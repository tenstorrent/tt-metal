# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_to_memory_config (the spec the implementation is held to; do not edit).

Cases are production cases only (BB-22): shapes and both placements are copied verbatim from
eval/golden_tests/g6_to_memory_config/feature_spec.py, every shard spec row-major as written there.
On top of them come the cells the design's rules require: a regime pin for every built regime
(push_bank and bounce have no bfloat16 production case), program-cache reuse and determinism
(BB-06, BB-07), the support refusals of the registry contract, and the host validation errors.

Graded exactly: helpers.TOLERANCES is ("exact", None) for every dtype, so values are compared with
torch.equal against the identity reference, and the result's memory configuration is compared with
the requested one, shard grid range order included.
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue
from ttnn.operations.g6_to_memory_config import g6_to_memory_config, plan_of

SEED = 42
# The refusal and validation messages are the implementation's own wording; only the type is the contract.
ANY_MESSAGE = ""

# "<model> <case id>": (shape, input placement, target placement)
CASES = {
    "Qwen 2058": (
        [1, 16, 1, 256],
        {"buffer": "dram", "scheme": "interleaved"},
        {
            "buffer": "l1",
            "scheme": "height_sharded",
            "grid": [
                [0, 2, 0, 2],
                [1, 2, 1, 2],
                [2, 2, 2, 2],
                [3, 2, 3, 2],
                [4, 2, 4, 2],
                [5, 1, 5, 1],
                [5, 2, 5, 2],
                [6, 1, 6, 1],
                [6, 2, 6, 2],
                [7, 1, 7, 1],
                [7, 2, 7, 2],
                [8, 1, 8, 1],
                [8, 2, 8, 2],
                [9, 1, 9, 1],
                [9, 2, 9, 2],
                [10, 1, 10, 1],
            ],
            "shard_shape": [32, 256],
        },
    ),
    "Qwen 2154": (
        [1, 1, 1, 256],
        {"buffer": "dram", "scheme": "interleaved"},
        {"buffer": "l1", "scheme": "height_sharded", "grid": [[1, 0, 1, 0]], "shard_shape": [32, 256]},
    ),
    "Qwen 2067": (
        [1, 1, 16, 5120],
        {"buffer": "l1", "scheme": "width_sharded", "grid": [[0, 0, 7, 0]], "shard_shape": [32, 640]},
        {"buffer": "l1", "scheme": "width_sharded", "grid": [[0, 0, 9, 3]], "shard_shape": [32, 128]},
    ),
    "Qwen 2064": (
        [1, 1, 16, 4352],
        {"buffer": "l1", "scheme": "interleaved"},
        {"buffer": "l1", "scheme": "width_sharded", "grid": [[0, 0, 7, 0]], "shard_shape": [32, 544]},
    ),
    "FLUX 313": (
        [1, 1, 16, 128],
        {"buffer": "dram", "scheme": "interleaved"},
        {"buffer": "l1", "scheme": "height_sharded", "grid": [[0, 0, 0, 0]], "shard_shape": [32, 128]},
    ),
    "Qwen 2157": (
        [1, 1, 1, 256],
        {"buffer": "l1", "scheme": "height_sharded", "grid": [[0, 0, 0, 0]], "shard_shape": [32, 256]},
        {"buffer": "l1", "scheme": "block_sharded", "grid": [[0, 0, 0, 0]], "shard_shape": [32, 256]},
    ),
    "Qwen 2078": (
        [16, 128, 5120],
        {"buffer": "dram", "scheme": "interleaved"},
        {"buffer": "l1", "scheme": "interleaved"},
    ),
    "Qwen 2105": (
        [1, 77, 4352],
        {"buffer": "dram", "scheme": "interleaved"},
        {"buffer": "l1", "scheme": "interleaved"},
    ),
    "Qwen 2089": (
        [1, 64, 1536],
        {"buffer": "dram", "scheme": "interleaved"},
        {"buffer": "l1", "scheme": "interleaved"},
    ),
    "Qwen 2104": (
        [1, 256, 5120],
        {"buffer": "dram", "scheme": "interleaved"},
        {"buffer": "l1", "scheme": "interleaved"},
    ),
    "Qwen 2071": (
        [1, 1, 16, 12928],
        {"buffer": "l1", "scheme": "width_sharded", "grid": [[0, 0, 7, 0]], "shard_shape": [32, 1664]},
        {"buffer": "dram", "scheme": "interleaved"},
    ),
    "Qwen 2065": (
        [1, 1, 16, 8704],
        {"buffer": "l1", "scheme": "width_sharded", "grid": [[0, 0, 10, 2], [0, 3, 2, 3]], "shard_shape": [32, 256]},
        {"buffer": "l1", "scheme": "interleaved"},
    ),
    "Qwen 2073": (
        [1, 1, 16, 5120],
        {"buffer": "l1", "scheme": "width_sharded", "grid": [[0, 0, 9, 3]], "shard_shape": [32, 128]},
        {"buffer": "l1", "scheme": "width_sharded", "grid": [[0, 0, 7, 0]], "shard_shape": [32, 640]},
    ),
    "Qwen 2061": (
        [1, 16, 1, 256],
        {"buffer": "l1", "scheme": "height_sharded", "grid": [[0, 0, 10, 0], [0, 1, 4, 1]], "shard_shape": [32, 256]},
        {"buffer": "dram", "scheme": "interleaved"},
    ),
    "Qwen 2155": (
        [1, 1, 1, 256],
        {"buffer": "l1", "scheme": "block_sharded", "grid": [[0, 0, 0, 0]], "shard_shape": [32, 256]},
        {"buffer": "l1", "scheme": "interleaved"},
    ),
    "Qwen 2062": (
        [1, 16, 6, 256],
        {"buffer": "l1", "scheme": "height_sharded", "grid": [[0, 0, 10, 0], [0, 1, 4, 1]], "shard_shape": [32, 256]},
        {"buffer": "dram", "scheme": "interleaved"},
    ),
    "FLUX 314": (
        [1, 1, 32, 2048],
        {"buffer": "l1", "scheme": "width_sharded", "grid": [[0, 0, 10, 0], [0, 1, 4, 1]], "shard_shape": [32, 128]},
        {"buffer": "dram", "scheme": "interleaved"},
    ),
    "Qwen 2124": (
        [1, 8, 1, 256],
        {"buffer": "l1", "scheme": "height_sharded", "grid": [[0, 0, 7, 0]], "shard_shape": [32, 256]},
        {"buffer": "dram", "scheme": "interleaved"},
    ),
    # bfloat4_b in production; used here only for its target placement (DRAM width sharded).
    "Qwen 2178": (
        [1536, 5120],
        {"buffer": "dram", "scheme": "interleaved"},
        {"buffer": "dram", "scheme": "width_sharded", "grid": [[0, 0, 7, 0]], "shard_shape": [1536, 640]},
    ),
    # bfloat8_b in production.
    "Panoptic 11": (
        [1, 32, 64, 256],
        {"buffer": "l1", "scheme": "block_sharded", "grid": [[0, 0, 7, 7]], "shard_shape": [256, 32]},
        {"buffer": "dram", "scheme": "interleaved"},
    ),
}

# Regime the design selects for each bfloat16 production case (op_design.md, Regimes).
PRODUCTION_REGIMES = {
    "Qwen 2058": "pull_shard",  # target grid of 16 single-core ranges forming no rectangle
    "Qwen 2154": "pull_shard",  # one-core target grid away from core (0, 0)
    "Qwen 2067": "pull_shard",  # width grid of 8 -> width grid of 40
    "Qwen 2064": "pull_shard",  # L1 interleaved source
    "FLUX 313": "pull_shard",  # 4 pages, H = 16
    "Qwen 2157": "pull_shard",  # source and target shard on the same core
    "Qwen 2078": "pull_bank",  # 10240 pages, the largest bfloat16 case
    "Qwen 2105": "pull_bank",  # H = 77
    "Qwen 2089": "pull_bank",  # fewer pages than L1 banks
    "Qwen 2071": "push_shard",  # uneven last shard: 40 of 52 tile columns
    "Qwen 2065": "push_shard",  # 36-core grid of two ranges holding 34 shards
    "Qwen 2073": "push_shard",  # width grid of 40 -> width grid of 8
    "Qwen 2061": "push_shard",  # height grid of two ranges -> DRAM
    "Qwen 2155": "push_shard",  # block sharded source -> L1 interleaved
    "Qwen 2062": "push_shard",  # H = 6
    "FLUX 314": "push_shard",  # width grid of two ranges -> DRAM
}

# SUPPORTED cells with no bfloat16 production case; production shapes, pinned to their regime.
SUPPORTED_CELLS = {
    "push_bank": (
        [1, 1, 16, 4352],
        {"buffer": "l1", "scheme": "interleaved"},  # Qwen 2064's input
        {"buffer": "dram", "scheme": "interleaved"},
    ),
    "bounce": (
        [16, 128, 1536],
        {"buffer": "dram", "scheme": "interleaved"},  # Qwen 2077's shape
        {"buffer": "dram", "scheme": "interleaved"},
    ),
}

_BUFFER = {"dram": ttnn.BufferType.DRAM, "l1": ttnn.BufferType.L1}
_SCHEME = {
    "interleaved": ttnn.TensorMemoryLayout.INTERLEAVED,
    "height_sharded": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    "width_sharded": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    "block_sharded": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
}


def torch_reference(x):
    """The identity: the result holds the input's stored values, bit for bit."""
    return x.clone()


def _memory_config(memory, *, orientation=ttnn.ShardOrientation.ROW_MAJOR, shard_shape=None, grid=None):
    buffer_type = _BUFFER[memory["buffer"]]
    scheme = _SCHEME[memory["scheme"]]
    if memory["scheme"] == "interleaved":
        return ttnn.MemoryConfig(scheme, buffer_type)
    ranges = grid if grid is not None else memory["grid"]
    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(sx, sy), ttnn.CoreCoord(ex, ey)) for sx, sy, ex, ey in ranges]
    )
    shape = shard_shape if shard_shape is not None else memory["shard_shape"]
    return ttnn.MemoryConfig(scheme, buffer_type, ttnn.ShardSpec(core_grid, tuple(shape), orientation))


def _skip_unless_grid_fits(device, *memories):
    for memory in memories:
        if "grid" not in memory:
            continue
        size = device.dram_grid_size() if memory["buffer"] == "dram" else device.compute_with_storage_grid_size()
        for _, _, ex, ey in memory["grid"]:
            if ex > size.x - 1 or ey > size.y - 1:
                pytest.skip(f"shard grid end ({ex},{ey}) outside this device's {memory['buffer']} grid")


def _num_shards(shape, memory):
    rows_tiles = math.prod(shape[:-2]) * math.ceil(shape[-2] / 32)
    cols_tiles = math.ceil(shape[-1] / 32)
    shard_h_tiles = memory["shard_shape"][0] // 32
    shard_w_tiles = memory["shard_shape"][1] // 32
    return math.ceil(rows_tiles / shard_h_tiles) * math.ceil(cols_tiles / shard_w_tiles)


def _bank_regimes_available(device):
    info = ttnn.reports.get_device_info(device)
    return info.l1_num_banks == info.num_compute_cores


def _make_input(device, shape, memory, *, dtype=ttnn.bfloat16, seed=SEED):
    torch.manual_seed(seed)
    torch_input = torch.randn(shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        torch_input = torch_input.to(torch.bfloat16)
    return ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_memory_config(memory)
    )


def _cores(shard_spec):
    cores = []
    for core_range in shard_spec.grid.ranges():
        for y in range(core_range.start.y, core_range.end.y + 1):
            for x in range(core_range.start.x, core_range.end.x + 1):
                cores.append((x, y))
    return cores


def _check_result(output, tt_input, requested):
    assert list(output.shape) == list(tt_input.shape)
    assert output.dtype == tt_input.dtype
    assert output.layout == ttnn.TILE_LAYOUT

    got = output.memory_config()
    assert got.buffer_type == requested.buffer_type
    assert got.memory_layout == requested.memory_layout
    if requested.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        assert got.shard_spec is not None
        assert tuple(got.shard_spec.shape) == tuple(requested.shard_spec.shape)
        assert got.shard_spec.orientation == requested.shard_spec.orientation
        assert _cores(got.shard_spec) == _cores(requested.shard_spec)

    expected = torch_reference(ttnn.to_torch(tt_input))
    actual = ttnn.to_torch(output)
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected), f"{(actual != expected).sum().item()} of {expected.numel()} values differ"


@pytest.mark.parametrize("case_id", list(PRODUCTION_REGIMES))
def test_production_case(device, case_id):
    shape, in_memory, out_memory = CASES[case_id]
    _skip_unless_grid_fits(device, in_memory, out_memory)
    tt_input = _make_input(device, shape, in_memory)
    out_config = _memory_config(out_memory)

    regime = PRODUCTION_REGIMES[case_id]
    if regime == "pull_bank" and not _bank_regimes_available(device):
        regime = "bounce"
    plan = plan_of(tt_input, out_config)
    assert plan["regime"] == regime
    if regime == "pull_shard":
        assert len(plan["movers"]) == _num_shards(shape, out_memory)
    if regime == "push_shard":
        assert len(plan["movers"]) == _num_shards(shape, in_memory)

    output = g6_to_memory_config(tt_input, out_config)
    _check_result(output, tt_input, out_config)


@pytest.mark.parametrize("regime", list(SUPPORTED_CELLS))
def test_supported_cell_regimes(device, regime):
    shape, in_memory, out_memory = SUPPORTED_CELLS[regime]
    tt_input = _make_input(device, shape, in_memory)
    out_config = _memory_config(out_memory)

    expected = regime
    if regime == "push_bank" and not _bank_regimes_available(device):
        expected = "bounce"
    assert plan_of(tt_input, out_config)["regime"] == expected

    output = g6_to_memory_config(tt_input, out_config)
    _check_result(output, tt_input, out_config)


@pytest.mark.parametrize("case_id", ["Qwen 2104", "Qwen 2071", "Qwen 2067"])
def test_repeat_call_cache_and_determinism(device, case_id):
    shape, in_memory, out_memory = CASES[case_id]
    _skip_unless_grid_fits(device, in_memory, out_memory)
    out_config = _memory_config(out_memory)

    first_input = _make_input(device, shape, in_memory, seed=SEED)
    first = g6_to_memory_config(first_input, out_config)
    entries = device.num_program_cache_entries()

    # A second input at another address: the cached program must be reused with new addresses.
    second_input = _make_input(device, shape, in_memory, seed=SEED + 1)
    second = g6_to_memory_config(second_input, out_config)
    assert device.num_program_cache_entries() == entries
    _check_result(second, second_input, out_config)

    repeat = g6_to_memory_config(first_input, out_config)
    assert device.num_program_cache_entries() == entries
    assert torch.equal(ttnn.to_torch(repeat), ttnn.to_torch(first))
    _check_result(first, first_input, out_config)


def test_refused_cells(device, expect_error):
    shape, in_memory, out_memory = CASES["Panoptic 11"]
    _skip_unless_grid_fits(device, in_memory, out_memory)
    block_format_input = _make_input(device, shape, in_memory, dtype=ttnn.bfloat8_b)
    with expect_error(UnsupportedAxisValue, ANY_MESSAGE):
        g6_to_memory_config(block_format_input, _memory_config(out_memory))

    shape, in_memory, out_memory = CASES["Qwen 2178"]
    _skip_unless_grid_fits(device, in_memory, out_memory)
    bf16_input = _make_input(device, shape, in_memory)
    with expect_error(ExcludedCell, ANY_MESSAGE):
        g6_to_memory_config(bf16_input, _memory_config(out_memory))


def test_validation_errors(device, expect_error):
    shape, in_memory, out_memory = CASES["Qwen 2058"]
    _skip_unless_grid_fits(device, in_memory, out_memory)
    out_config = _memory_config(out_memory)
    torch.manual_seed(SEED)
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)

    host_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    with expect_error((ValueError, RuntimeError), ANY_MESSAGE):
        g6_to_memory_config(host_tensor, out_config)

    row_major_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error((ValueError, RuntimeError), ANY_MESSAGE):
        g6_to_memory_config(row_major_input, out_config)

    tt_input = _make_input(device, shape, in_memory)
    with expect_error((ValueError, RuntimeError), ANY_MESSAGE):
        g6_to_memory_config(tt_input, _memory_config(out_memory, orientation=ttnn.ShardOrientation.COL_MAJOR))
    with expect_error((ValueError, RuntimeError), ANY_MESSAGE):
        g6_to_memory_config(tt_input, _memory_config(out_memory, shard_shape=[16, 256]))
    # 16 shards are needed; the first 8 single-core ranges hold only 8.
    with expect_error((ValueError, RuntimeError), ANY_MESSAGE):
        g6_to_memory_config(tt_input, _memory_config(out_memory, grid=out_memory["grid"][:8]))

    rank5_input = ttnn.from_torch(
        torch.randn([1, 1, 16, 1, 256], dtype=torch.float32).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error((ValueError, RuntimeError), ANY_MESSAGE):
        g6_to_memory_config(rank5_input, ttnn.L1_MEMORY_CONFIG)


def test_validation_error_col_major_input(device, expect_error):
    shape, in_memory, out_memory = CASES["Qwen 2124"]
    _skip_unless_grid_fits(device, in_memory, out_memory)
    torch.manual_seed(SEED)
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    try:
        col_major_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_memory_config(in_memory, orientation=ttnn.ShardOrientation.COL_MAJOR),
        )
    except RuntimeError as exc:
        pytest.skip(f"ttnn cannot build a column-major sharded input here: {exc}")
    with expect_error((ValueError, RuntimeError), ANY_MESSAGE):
        g6_to_memory_config(col_major_input, _memory_config(out_memory))
