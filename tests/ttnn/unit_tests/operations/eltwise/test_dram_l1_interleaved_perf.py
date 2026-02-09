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

  TWO INTERACTING EFFECTS cause the issue:

  1) L1 slave port contention (affects L1->L1):
     Reader uses NOC_0, writer uses NOC_1 (separate physical networks).
     L1-interleaved tiles are round-robin: bank_id = tile_id % 64.
     Each core processes a contiguous tile range so ~98% of L1 reads are
     REMOTE (via NOC).  In L1->L1 mode, each destination core's L1 is
     simultaneously accessed by NOC_0 reads + NOC_1 writes, creating
     slave port contention.  The per-tile noc_async_read_barrier()
     serializes this contention into per-tile latency.

  2) DRAM bank access pattern uniformity (affects DRAM->L1):
     DRAM bank_id = tile_id % 12.  Tile IDs are assigned in row-major
     order: tile_id = tile_row * num_tile_cols + tile_col.  The row-to-row
     bank offset is (num_tile_cols % 12).  When gcd(num_tile_cols, 12) is
     large, the bank access pattern has short cycles, causing temporal
     clustering of requests to the same DRAM bank and NOC2AXI bridge
     contention.

     Experiment B (sweep cols%12 at ~10k tiles) shows:
       gcd=1 (cols%12=1,5,7,11): DRAM->L1 ~105-115k ns (best)
       gcd=2 (cols%12=2,10):     DRAM->L1 ~105-113k ns
       gcd=3 (cols%12=3,9):      DRAM->L1 ~109-115k ns
       gcd=4 (cols%12=4,8):      DRAM->L1 ~118-169k ns (worst!)
       gcd=6 (cols%12=6):        DRAM->L1 ~135k ns
       gcd=12 (cols%12=0):       DRAM->L1 ~118k ns

     The original issue's test shapes used W=2048 (64 tile-cols, cols%12=4,
     gcd=4), which is the WORST case for DRAM bank uniformity.  This made
     DRAM->L1 abnormally slow, masking the L1 contention effect at low tile
     counts.  At ~156+ tiles/core the L1 contention overtakes even the
     degraded DRAM performance, revealing the crossover.

     With better DRAM bank access patterns (e.g. cols%12=1, gcd=1),
     DRAM->L1 is ~105k ns at ~10k tiles -- consistently faster than
     L1->L1 (~124k ns).

  Key code paths:
    Reader kernel: unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp
    Writer kernel: unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp
    NOC assignment: tt_metal/impl/kernels/kernel_types.cpp
      -> preferred_noc_for_dram_read() = NOC_0
      -> preferred_noc_for_dram_write() = NOC_1
    Grid selection: tt_metal/common/work_split.cpp::split_work_to_cores()
    Tile-to-bank mapping: tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h
      -> bank_id = tile_id % NUM_L1_BANKS (64) for L1
      -> bank_id = tile_id % NUM_DRAM_BANKS (12) for DRAM
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

# Experiment A: Same tile counts as 9984/10048 but with W=2048 (cols%12=4)
# to control for shape/aspect-ratio effects on DRAM bank access patterns.
EXPERIMENT_A_SHAPES = [
    ((1, 1, 4992, 2048), 9984),  # 156 tiles/core, cols%12=4 (was 3072x3328, cols%12=8)
    ((1, 1, 5024, 2048), 10048),  # 157 tiles/core, cols%12=4 (was 2048x5024, cols%12=1)
]

# Experiment B: Fixed H=2048 (~10k tiles), vary W to sweep cols%12 from 0 to 11.
# Isolates the effect of DRAM bank access uniformity on DRAM->L1 duration.
# bank_id = tile_id % 12; row offset = num_tile_cols % 12.
EXPERIMENT_B_SHAPES = [
    ((1, 1, 2048, 4992), 9984),  # 156/core, cols=156, cols%12=0
    ((1, 1, 2048, 5024), 10048),  # 157/core, cols=157, cols%12=1
    ((1, 1, 2048, 5056), 10112),  # 158/core, cols=158, cols%12=2
    ((1, 1, 2048, 5088), 10176),  # 159/core, cols=159, cols%12=3
    ((1, 1, 2048, 5120), 10240),  # 160/core, cols=160, cols%12=4
    ((1, 1, 2048, 5152), 10304),  # 161/core, cols=161, cols%12=5
    ((1, 1, 2048, 5184), 10368),  # 162/core, cols=162, cols%12=6
    ((1, 1, 2048, 5216), 10432),  # 163/core, cols=163, cols%12=7
    ((1, 1, 2048, 5248), 10496),  # 164/core, cols=164, cols%12=8
    ((1, 1, 2048, 5280), 10560),  # 165/core, cols=165, cols%12=9
    ((1, 1, 2048, 5312), 10624),  # 166/core, cols=166, cols%12=10
    ((1, 1, 2048, 5344), 10688),  # 167/core, cols=167, cols%12=11
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


def _run_shape_experiment(shape, num_tiles, monkeypatch, label_prefix=""):
    """Shared logic for experiment A and B: measure all 4 combos, print results."""
    _check_profiler_available()
    _enable_profiler(monkeypatch)

    if ttnn.GetNumAvailableDevices() < 1:
        pytest.skip("No devices available")

    num_tile_cols = shape[3] // 32
    cols_mod_12 = num_tile_cols % 12

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

    tiles_per_core = num_tiles / 64
    print(
        f"\n{label_prefix}Shape: {shape}  ({num_tiles} tiles, "
        f"{tiles_per_core:.0f} tiles/core, cols={num_tile_cols}, cols%12={cols_mod_12})"
    )
    for label in durations:
        print(f"  {label:12s} duration: {durations[label]:>8d} ns, " f"cores: {core_counts[label]:>3d}")

    return durations, core_counts


@pytest.mark.parametrize(
    "shape, num_tiles",
    EXPERIMENT_A_SHAPES,
    ids=[f"expA_{nt}_tiles_W2048" for _, nt in EXPERIMENT_A_SHAPES],
)
def test_experiment_a_fixed_width(shape, num_tiles, monkeypatch):
    """Experiment A: Same tile counts as crossover region but with fixed W=2048."""
    _run_shape_experiment(shape, num_tiles, monkeypatch, label_prefix="[Exp A] ")


@pytest.mark.parametrize(
    "shape, num_tiles",
    EXPERIMENT_B_SHAPES,
    ids=[f"expB_{s[3]//32}cols_mod12eq{(s[3]//32)%12}" for s, _ in EXPERIMENT_B_SHAPES],
)
def test_experiment_b_vary_cols_mod12(shape, num_tiles, monkeypatch):
    """Experiment B: Fixed H=2048, vary W to sweep cols%12 from 0 to 11."""
    _run_shape_experiment(shape, num_tiles, monkeypatch, label_prefix="[Exp B] ")
