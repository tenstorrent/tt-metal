# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `tile_reorder` data-movement example (whole-tile write vs per-face writes).

See ttnn/ttnn/operations/examples/tile_reorder/README.md for what this illustrates.

    # both methods produce identical, correct output (DRAM, L1-interleaved and L1-sharded outputs)
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_tile_reorder.py::test_tile_reorder_correctness

    # when coalescing helps / when it does not scatter vs relocate
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_tile_reorder.py::test_tile_reorder_regimes
"""

import os

# Enable the on-device profiler IN-PROCESS (all three, before the device opens).
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest
import torch

import ttnn
from ttnn.operations.examples.tile_reorder import tile_reorder

from tests.ttnn.unit_tests.operations.examples.perf_test_utils import check_regime, grid_cores, interleaved_ns

TILE = 32
N_TRIALS = int(os.environ.get("TR_TRIALS", "12"))
PER_TRIAL = 3  # launches averaged into one trial


def _grid_shape(device):
    """One tile-row (32 bf16 tiles of a 1024-wide row) per Tensix core of the attached grid: (2048, 1024) on
    8x8, (1792, 1024) on 8x7, (3840, 1024) on 12x10. split_work_to_cores (row_wise=True) then gives each
    Tensix core exactly one tile-row, and the L1 HEIGHT_SHARDED output has one shard per Tensix core."""
    return (grid_cores(device) * TILE, 1024)


def _reference(torch_input):
    """Reverse the 32-wide column blocks (whole-tile reversal along width)."""
    h, w = torch_input.shape
    return torch_input.reshape(h, w // TILE, TILE).flip(1).reshape(h, w).contiguous()


def _local_height_sharded(device, shape):
    """L1 HEIGHT_SHARDED, one tile-row (32 x W) per Tensix core, ROW_MAJOR over the full grid.
    This matches the op's row-wise page split, so every Tensix core's output pages live in its OWN L1."""
    grid = device.compute_with_storage_grid_size()
    return ttnn.create_sharded_memory_config(
        shape=(TILE, shape[1]),
        core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _out_mem(device, name, shape):
    if name == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    if name == "l1_interleaved":
        return ttnn.L1_MEMORY_CONFIG
    if name == "l1_local_shard":
        return _local_height_sharded(device, shape)
    raise ValueError(name)


def _make_input(device, shape):
    torch.manual_seed(0)
    torch_input = torch.rand(shape, dtype=torch.float32)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return tt_input


@pytest.mark.parametrize("out_mem", ["dram", "l1_interleaved", "l1_local_shard"])
@pytest.mark.parametrize("method", ["scatter", "relocate"])
def test_tile_reorder_correctness(device, method, out_mem):
    """Both methods must produce the identical, exact whole-tile column reversal. The shape follows the
    grid (one tile-row per Tensix core) so the HEIGHT_SHARDED output has a shard per Tensix core on any board."""
    shape = _grid_shape(device)
    tt_input = _make_input(device, shape)
    expected = _reference(ttnn.to_torch(tt_input).to(torch.float32))
    out = tile_reorder(tt_input, method=method, memory_config=_out_mem(device, out_mem, shape))
    out = ttnn.to_torch(out).to(torch.float32)
    assert list(out.shape) == list(expected.shape), f"{out.shape} != {expected.shape}"
    assert torch.equal(out, expected)


# ---------------------------------------------------------------------------------------------
# When it helps / when it does not (numbers: WH B0 n150 and BH p150b in README.md). Baseline = method="scatter"
# (each 2 KB bf16 tile written as 4 x 512 B faces, a write barrier after each face); lever = method="relocate"
# (one 2 KB write + one barrier per tile). Input is always DRAM-interleaved; only the OUTPUT placement varies.
# The shape is one tile-row per Tensix core of the attached grid (_grid_shape), so every board runs it.
#
# The scatter writer pays 4 serialized write round trips per tile. That costs time only while the
# writer is the slower half of the reader->writer pipeline, i.e. while each write round trip is long
# (a remote DRAM or remote-L1 bank). When every output page (one tile) lives in the writing Tensix core's
# OWN L1 (a sharded output co-located with the work split), each face write is a short loopback, the
# reader's one-DRAM-read-per-tile becomes the bottleneck, and coalescing the writes changes nothing.
# ---------------------------------------------------------------------------------------------
# id: (output memory, kind, measured)
#   measured = speedup (scatter / relocate) measured per architecture: "wh" Wormhole B0 n150, "bh" Blackhole p150b.
#   perf_test_utils.check_regime derives the bound from the number measured on THIS architecture
#   (perf_test_utils.regime_bound: "helps" asserts at least the lower of half the measured gain and the measured
#   number less a 10% noise margin; "does_not_help" asserts at most the measured high plus that margin, never
#   below 1.0x), re-measures once before failing, and only logs on an architecture with no measured number.
#   "measure" regimes flip between boards: correctness is checked, the speedup is logged, nothing is asserted.
_TR_REGIMES = {
    # BH DRAM ceiling hides the barriers (BH 0.99x at 2048 rows); the numbers are kept for the log only.
    "measure_dram_out": ("dram", "measure", {"wh": 1.58, "bh": 1.10}),
    "helps_l1_interleaved_out": ("l1_interleaved", "helps", {"wh": 1.40, "bh": 1.50}),
    "flat_l1_local_shard_out": ("l1_local_shard", "does_not_help", {"wh": 1.01, "bh": 1.01}),
}


@pytest.mark.parametrize("regime", list(_TR_REGIMES), ids=list(_TR_REGIMES))
def test_tile_reorder_regimes(device, regime):
    """Coalesced whole-tile writes: one regime where they clearly help, one where they do not."""
    out_name, kind, measured = _TR_REGIMES[regime]
    shape = _grid_shape(device)
    tt_input = _make_input(device, shape)
    out_mem = _out_mem(device, out_name, shape)

    expected = _reference(ttnn.to_torch(tt_input).to(torch.float32))
    for method in ("scatter", "relocate"):
        out = ttnn.to_torch(tile_reorder(tt_input, method=method, memory_config=out_mem)).to(torch.float32)
        assert torch.equal(out, expected), f"{regime}: {method} output mismatch"

    def measure():
        samples = interleaved_ns(
            device,
            {
                "scatter": lambda: tile_reorder(tt_input, method="scatter", memory_config=out_mem),
                "relocate": lambda: tile_reorder(tt_input, method="relocate", memory_config=out_mem),
            },
            trials=N_TRIALS,
            launches=PER_TRIAL,
        )
        return samples["scatter"], samples["relocate"]

    tiles = (shape[0] // TILE) * (shape[1] // TILE)
    check_regime(
        device,
        regime,
        kind,
        measured,
        measure,
        detail=(
            f"shape={shape} ({tiles} bf16 tiles, {tiles // grid_cores(device)} tiles per Tensix core), "
            f"input DRAM interleaved, output {out_name}, {PER_TRIAL} launches per trial | "
            "baseline scatter (4 x 512 B writes per tile) -> lever relocate (1 x 2 KB write per tile)"
        ),
    )
