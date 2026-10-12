# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `tile_reorder` data-movement example (whole-tile write vs per-face writes).

See ttnn/ttnn/operations/examples/tile_reorder/README.md for what this illustrates.

    # both methods produce identical, correct output (DRAM, L1-interleaved and L1-sharded outputs)
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_tile_reorder.py::test_tile_reorder_correctness

    # when coalescing helps / when it does not (Wormhole B0 n150), scatter vs relocate
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_tile_reorder.py::test_tile_reorder_regimes
"""

import os

# Enable the on-device profiler IN-PROCESS (all three, before the device opens).
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import statistics

import pytest
import torch

import ttnn
from ttnn.operations.examples.tile_reorder import tile_reorder

from loguru import logger


TILE = 32
N_WARMUP = 5
N_TRIALS = int(os.environ.get("TR_TRIALS", "12"))
PER_TRIAL = 3  # launches averaged into one trial

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# 64 tile-rows x 32 tile-columns = 2048 bf16 tiles (4 MB). On the 8x8 grid, split_work_to_cores
# (row_wise=True) gives each Tensix core 32 contiguous output tiles = exactly one tile-row.
SHAPE = (2048, 1024)


def _reference(torch_input):
    """Reverse the 32-wide column blocks (whole-tile reversal along width)."""
    h, w = torch_input.shape
    return torch_input.reshape(h, w // TILE, TILE).flip(1).reshape(h, w).contiguous()


def _local_height_sharded(device, shape):
    """L1 HEIGHT_SHARDED, one tile-row (32 x W) per Tensix core, ROW_MAJOR over the full grid.
    This matches the op's row-wise page split, so every core's output pages live in its OWN L1."""
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


def _make_input(device, shape=SHAPE):
    torch.manual_seed(0)
    torch_input = torch.rand(shape, dtype=torch.float32)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return tt_input


def _read_kernel_ns(device):
    """Sum of on-device kernel duration over programs since the last read (flush-consumes)."""
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def _interleaved_trials(device, run_a, run_b, trials=N_TRIALS, per_trial=PER_TRIAL):
    """A,B,A,B... trials; each trial = mean device-kernel ns over `per_trial` launches."""
    for _ in range(N_WARMUP):
        run_a()
        run_b()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # flush warmup window
    a_ns, b_ns = [], []
    for _ in range(trials):
        for samples, run in ((a_ns, run_a), (b_ns, run_b)):
            for _ in range(per_trial):
                run()
            total = _read_kernel_ns(device)
            assert total is not None, "profiler produced no data (is the build profiler-enabled?)"
            samples.append(total / per_trial)
    return a_ns, b_ns


@pytest.mark.parametrize("out_mem", ["dram", "l1_interleaved", "l1_local_shard"])
@pytest.mark.parametrize("method", ["scatter", "relocate"])
def test_tile_reorder_correctness(device, method, out_mem):
    """Both methods must produce the identical, exact whole-tile column reversal."""
    tt_input = _make_input(device)
    expected = _reference(ttnn.to_torch(tt_input).to(torch.float32))
    out = tile_reorder(tt_input, method=method, memory_config=_out_mem(device, out_mem, SHAPE))
    out = ttnn.to_torch(out).to(torch.float32)
    assert list(out.shape) == list(expected.shape), f"{out.shape} != {expected.shape}"
    assert torch.equal(out, expected)


# ---------------------------------------------------------------------------------------------
# When it helps / when it does not (Wormhole B0 n150). Baseline = method="scatter" (each 2 KB tile
# written as 4 x 512 B faces, a write barrier after each face); lever = method="relocate" (one 2 KB
# write + one barrier per tile). Input is always DRAM-interleaved; only the OUTPUT placement varies.
#
# The scatter writer pays 4 serialized write round trips per tile. That costs time only while the
# writer is the slower half of the reader->writer pipeline, i.e. while each write round trip is long
# (a remote DRAM or remote-L1 bank). When every output page lives in the writer's OWN L1 (a sharded
# output co-located with the work split), each face write is a short loopback, the reader's
# one-DRAM-read-per-tile becomes the bottleneck, and coalescing the writes changes nothing.
# ---------------------------------------------------------------------------------------------
# id: (output memory, kind, threshold)
#   "helps"         -> assert speedup >= threshold (T_help = 1 + 0.5 * (measured - 1), >= 1.05)
#   "does_not_help" -> assert speedup <= threshold
_TR_REGIMES = {
    "helps_dram_out": ("dram", "helps", 1.29),
    "helps_l1_interleaved_out": ("l1_interleaved", "helps", 1.20),
    "flat_l1_local_shard_out": ("l1_local_shard", "does_not_help", 1.05),
}


@pytest.mark.parametrize("regime", list(_TR_REGIMES), ids=list(_TR_REGIMES))
def test_tile_reorder_regimes(device, regime):
    """Coalesced whole-tile writes: one regime where they clearly help, one where they do not."""
    # The regime thresholds below were calibrated on Wormhole B0 (n150). The boundary between
    # "helps" and "does not help" moves with the arch (grid size, DRAM banks, NoC, clock), so on
    # another arch re-measure the regimes and add its own thresholds before enabling this test.
    if not ttnn.device.is_wormhole_b0(device):
        pytest.skip("regime thresholds are calibrated on Wormhole B0 only; re-measure on this arch first")
    out_name, kind, threshold = _TR_REGIMES[regime]
    tt_input = _make_input(device)
    out_mem = _out_mem(device, out_name, SHAPE)

    expected = _reference(ttnn.to_torch(tt_input).to(torch.float32))
    for method in ("scatter", "relocate"):
        out = ttnn.to_torch(tile_reorder(tt_input, method=method, memory_config=out_mem)).to(torch.float32)
        assert torch.equal(out, expected), f"{regime}: {method} output mismatch"

    base, lever = _interleaved_trials(
        device,
        lambda: tile_reorder(tt_input, method="scatter", memory_config=out_mem),
        lambda: tile_reorder(tt_input, method="relocate", memory_config=out_mem),
    )
    base_med, lever_med = statistics.median(base), statistics.median(lever)
    speedup = base_med / lever_med
    grid = device.compute_with_storage_grid_size()
    tiles = (SHAPE[0] // TILE) * (SHAPE[1] // TILE)
    logger.info(
        f"\n=== tile_reorder regime {regime} (WH B0) ==="
        f"\n    shape={SHAPE} ({tiles} bf16 tiles, {tiles // (grid.x * grid.y)} tiles/core on {grid.x}x{grid.y})  "
        f"input=DRAM interleaved  output={out_name}"
        f"\n    baseline scatter  (4 x 512 B writes/tile): median {base_med:9.0f} ns  [min {min(base):.0f} - max {max(base):.0f}]"
        f"\n    lever    relocate (1 x 2 KB write/tile)  : median {lever_med:9.0f} ns  [min {min(lever):.0f} - max {max(lever):.0f}]"
        f"\n    speedup = {speedup:.2f}x  ({N_TRIALS} trials x {PER_TRIAL} launches)  kind={kind}  threshold={threshold}"
    )
    if kind == "helps":
        assert speedup >= threshold, f"{regime}: expected >= {threshold}x, got {speedup:.2f}x"
    else:
        assert speedup <= threshold, f"{regime}: expected <= {threshold}x, got {speedup:.2f}x"
