# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test for GitHub issue #14421: DRAM interleaved and L1 interleaved inputs
in the relu op should have different kernel duration times.

Tests all 4 memory layout combinations (input x output):
  1. DRAM -> DRAM
  2. DRAM -> L1
  3. L1   -> DRAM
  4. L1   -> L1

Expected ordering: L1->L1 is fastest, DRAM->DRAM is slowest.
Tile counts chosen to mirror the original issue chart (~100..12800 tiles).

Root cause analysis (Wormhole B0, 2 NOCs, 12 DRAM banks, 64 L1 banks):

  Reader uses NOC_0, writer uses NOC_1 (separate physical networks).
  L1-interleaved tiles are round-robin distributed across 64 L1 banks
  (one per compute core) via bank_id = tile_id % 64.  Each core processes
  a contiguous tile range so ~98% of L1 reads are REMOTE (via NOC).

  L1->L1: Each destination core's L1 is simultaneously accessed by:
    - NOC_0 read requests from reader cores fetching input tiles
    - NOC_1 write requests from writer cores storing output tiles
  This creates L1 slave port contention at destination cores.

  DRAM->L1: Reader traffic goes to 12 DRAM banks via NOC2AXI bridge,
  bypassing the L1 fabric entirely.  Destination L1 cores only receive
  NOC_1 write traffic -- no L1 slave port contention on the read path.

  At ~156 tiles/core (9984 tiles), the sustained bidirectional L1 traffic
  saturates the L1 slave interface, making L1->L1 slower than DRAM->L1.
  The crossover is confined to a narrow window (~9984-10048 tiles);
  at 11264+ tiles L1->L1 returns to being faster.
  The per-tile noc_async_read_barrier() in the reader kernel amplifies
  this: contention-induced read latency is serialized per tile.

  Key code paths:
    Reader kernel: unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp
    Writer kernel: unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp
    NOC assignment: tt_metal/impl/kernels/kernel_types.cpp
      -> preferred_noc_for_dram_read() = NOC_0
      -> preferred_noc_for_dram_write() = NOC_1
    Grid selection: tt_metal/common/work_split.cpp::split_work_to_cores()
    Tile-to-bank mapping: tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h
      -> bank_id = tile_id % NUM_L1_BANKS (64)
"""

import json
import pytest
import torch
import ttnn

MEMORY_CONFIGS = {
    "DRAM": ttnn.DRAM_MEMORY_CONFIG,
    "L1": ttnn.L1_MEMORY_CONFIG,
}

# Shapes chosen to produce tile counts matching the issue chart.
# Each tile is 32x32. num_tiles = (H/32) * (W/32).
# Shapes marked (*) evenly split across 64 cores (tiles_per_core * 64).
#   (1,1,  320,  320) ->    100 tiles
#   (1,1, 1024, 1024) ->   1024 tiles
#   (1,1, 1024, 2048) ->   2048 tiles
#   (1,1, 2048, 2560) ->   5120 tiles  (*)  80 tiles/core
#   (1,1, 3584, 2048) ->   7168 tiles  (*) 112 tiles/core
#   (1,1, 4096, 2048) ->   8192 tiles  (*) 128 tiles/core
#   (1,1, 4608, 2048) ->   9216 tiles  (*) 144 tiles/core
#   (1,1, 3072, 3328) ->   9984 tiles  (*) 156 tiles/core
#   (1,1, 3200, 3200) ->  10000 tiles       156.25 tiles/core (uneven)
#   (1,1, 2048, 5024) ->  10048 tiles  (*) 157 tiles/core
#   (1,1, 5632, 2048) ->  11264 tiles  (*) 176 tiles/core
#   (1,1, 6144, 2048) ->  12288 tiles  (*) 192 tiles/core
#   (1,1, 3200, 4096) ->  12800 tiles       200 tiles/core
#   (1,1, 7168, 2048) ->  14336 tiles  (*) 224 tiles/core
TILE_COUNT_SHAPES = [
    ((1, 1, 320, 320), 100),
    ((1, 1, 1024, 1024), 1024),
    ((1, 1, 1024, 2048), 2048),
    ((1, 1, 2048, 2560), 5120),
    ((1, 1, 3584, 2048), 7168),
    ((1, 1, 4096, 2048), 8192),
    ((1, 1, 4608, 2048), 9216),
    ((1, 1, 3072, 3328), 9984),
    ((1, 1, 3200, 3200), 10000),
    ((1, 1, 2048, 5024), 10048),
    ((1, 1, 5632, 2048), 11264),
    ((1, 1, 6144, 2048), 12288),
    ((1, 1, 3200, 4096), 12800),
    ((1, 1, 7168, 2048), 14336),
]


def _enable_profiler(monkeypatch):
    """Enable device profiler via environment variables."""
    for var, value in {
        "TT_METAL_DEVICE_PROFILER": "1",
        "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
        "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
        "TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES": "1",
    }.items():
        monkeypatch.setenv(var, value)


def _check_profiler_available():
    """Skip test if profiler bindings are not available."""
    profiler_mod = getattr(ttnn, "_ttnn", None)
    if profiler_mod is None or not hasattr(profiler_mod, "profiler"):
        pytest.skip("Profiler bindings not available in this build")


def _get_kernel_duration_and_cores(device) -> tuple:
    """
    Read device profiler and return (max_kernel_duration_ns, core_count)
    from the latest program execution data.
    """
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)

    latest_data = ttnn.get_latest_programs_perf_data()
    if not latest_data:
        pytest.skip("No profiler data returned")

    device_id = next(iter(latest_data))
    programs = list(latest_data[device_id])
    if not programs:
        pytest.skip("No program entries in profiler data")

    max_duration = 0
    core_count = 0
    for program in programs:
        core_count = max(core_count, program.core_count)
        for _name, result in program.program_analyses_results.items():
            if result.duration > max_duration:
                max_duration = result.duration

    return max_duration, core_count


def _measure_relu(device, input_tensor, output_memory_config, warmup_iterations=2):
    """Run relu with warmup, then return (kernel_duration_ns, core_count)."""
    for _ in range(warmup_iterations):
        out = ttnn.relu(input_tensor, memory_config=output_memory_config)
        ttnn.deallocate(out)

    # Flush profiler state
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)

    # Measured run
    out = ttnn.relu(input_tensor, memory_config=output_memory_config)
    duration, core_count = _get_kernel_duration_and_cores(device)
    ttnn.deallocate(out)
    return duration, core_count


@pytest.mark.parametrize(
    "shape, num_tiles",
    TILE_COUNT_SHAPES,
    ids=[f"{nt}_tiles" for _, nt in TILE_COUNT_SHAPES],
)
def test_relu_memory_layout_kernel_duration(shape, num_tiles, monkeypatch, tmp_path):
    """
    Measure relu kernel duration for all 4 input/output memory layout
    combinations and verify expected ordering:
      - L1->L1 < DRAM->L1  (L1 input should be faster than DRAM input)
      - L1->L1 < L1->DRAM  (L1 output should be faster than DRAM output)
      - L1->L1 < DRAM->DRAM (fully L1 should be fastest)

    See https://github.com/tenstorrent/tt-metal/issues/14421
    """
    _check_profiler_available()
    _enable_profiler(monkeypatch)

    if ttnn.GetNumAvailableDevices() < 1:
        pytest.skip("No devices available")

    device = ttnn.open_device(device_id=0)
    try:
        try:
            ttnn.get_all_programs_perf_data()
        except RuntimeError as exc:
            if "profiler_state_manager is nullptr" in str(exc):
                pytest.skip("Profiler state manager not initialized")
            raise

        torch.manual_seed(0)
        torch_input = torch.randn(shape, dtype=torch.bfloat16)

        durations = {}
        core_counts = {}
        for in_name, in_mem_config in MEMORY_CONFIGS.items():
            input_tensor = ttnn.from_torch(
                torch_input,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=in_mem_config,
            )
            for out_name, out_mem_config in MEMORY_CONFIGS.items():
                label = f"{in_name}->{out_name}"
                duration, cores = _measure_relu(device, input_tensor, out_mem_config)
                durations[label] = duration
                core_counts[label] = cores

            ttnn.deallocate(input_tensor)

        compute_grid = device.compute_with_storage_grid_size()
        max_cores = compute_grid.x * compute_grid.y
    finally:
        ttnn.close_device(device)

    # Print results
    print(f"\nShape: {shape}  ({num_tiles} tiles, device grid: {max_cores} cores)")
    for label in durations:
        cores = core_counts[label]
        tiles_per_core = num_tiles / cores if cores else 0
        print(
            f"  {label:12s} duration: {durations[label]:>8d} ns, "
            f"cores: {cores:>3d}, tiles/core: {tiles_per_core:.1f}"
        )

    # Dump JSON for plotting
    result = {
        "num_tiles": num_tiles,
        "shape": list(shape),
        "durations": durations,
        "core_counts": core_counts,
        "max_cores": max_cores,
    }
    results_dir = tmp_path.parent / "relu_perf_results"
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / f"{num_tiles}_tiles.json", "w") as f:
        json.dump(result, f)

    # Assertions: L1->L1 should be strictly faster than the other 3 combinations.
    # The DRAM->L1 comparison is the core of issue #14421 -- at larger tile counts
    # (e.g. 10000) L1->L1 can be equal to or slower than DRAM->L1 due to L1 slave
    # port contention when both NOC_0 (reader) and NOC_1 (writer) target the same
    # L1 banks simultaneously.  See module docstring for full root cause analysis.
    issue_url = "https://github.com/tenstorrent/tt-metal/issues/14421"

    assert durations["L1->L1"] < durations["DRAM->DRAM"], (
        f"Expected L1->L1 ({durations['L1->L1']} ns) < DRAM->DRAM ({durations['DRAM->DRAM']} ns). " f"See {issue_url}"
    )

    if durations["L1->L1"] >= durations["DRAM->L1"]:
        pytest.xfail(
            f"Known issue #14421: L1->L1 ({durations['L1->L1']} ns) >= "
            f"DRAM->L1 ({durations['DRAM->L1']} ns) at {num_tiles} tiles. "
            f"Root cause: L1 slave port contention from simultaneous NOC_0 "
            f"read + NOC_1 write traffic to same L1 banks."
        )
    if durations["L1->L1"] >= durations["L1->DRAM"]:
        pytest.xfail(
            f"Known issue #14421: L1->L1 ({durations['L1->L1']} ns) >= "
            f"L1->DRAM ({durations['L1->DRAM']} ns) at {num_tiles} tiles. "
            f"Root cause: L1 slave port contention from simultaneous NOC_0 "
            f"read + NOC_1 write traffic to same L1 banks."
        )
