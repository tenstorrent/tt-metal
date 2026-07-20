# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# This test benchmarks matmul_2d across different program configs (modes) and math fidelities.
#
# Modes (see tech_reports/GEMM_FLOPS/benchmark_modes.py for plot legend names):
#   - oob: Out-of-box (auto-selected program config, DRAM inputs)
#   - reuse_dram: MatmulMultiCoreReuseProgramConfig — L1 interleaved in0/out, DRAM interleaved in1
#   - mcast_2d_l1: MatmulMultiCoreReuseMultiCastProgramConfig — L1 block-sharded in0/out, DRAM in1
#   - mcast_2d_dram: MatmulMultiCoreReuseMultiCastProgramConfig — L1 interleaved in0/out, DRAM in1
#   - mcast_1d_in0: MatmulMultiCoreReuseMultiCast1DProgramConfig (mcast_in0=True) — L1 in0/out, DRAM in1
#   - mcast_1d_out: MatmulMultiCoreReuseMultiCast1DProgramConfig (mcast_in0=False) — L1 in0/out, DRAM in1
#   - dram_sharded: MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig — L1 width-sharded in0/out, DRAM width-sharded in1
#
# The inputs include:
#   - m, k, n: Full tensor dimensions in elements (see matmul_m/k/n_values).
#   - in0_block_w_div: A parameter to divide an in0 block into multiple chunks, helping to reduce L1 cache usage.
#   - num_out_blocks_h: A parameter to divide an output block into multiple chunks on height dim.
#   - num_out_blocks_w: A parameter to divide an output block into multiple chunks on width dim.
#
# Performance metrics:
#   - Inference time: The average time taken for inference in nanoseconds.
#   - TFLOPs: The number of tera floating-point operations per second.
#   - Host based utilization: The ratio of ideal cycles to inference cycles.
#   - Device based utilization: The ratio of ideal cycles to TRISC1 kernel duration (requires profiler build).
#
# Important notes regarding performance metrics calculation:
#   - If profiler build is not enabled (TT_METAL_DEVICE_PROFILER environment variable is not set) only host based utilization is calculated.
#   - If profiler build is enabled, additionally device based utilization is calculated using TRISC1 kernel duration that is
#     measured and reported by profiler build.
#   - Inference time is measured by host based on repeated(num_measurement_iterations times) execution of matmul operation.
#   - Measured inference time includes all host and device overheads. If profiler build is enabled it can have impact on
#     runtime performance of device and host operations.
#   - TFLOPs is calculated based on the formula: 2 * m * k * n / 1e12 / inference_time_avg
#       - factor 2 comes from fact that we are doing multiplication and addition in matmul operation.
#   - Host based utilization is calculated using formula: ideal_cycle / inference_cycle
#       - ideal_cycle is the number of cycles required to perform matmul on the input tensors:
#         m * k * n / (tile_h * tile_w * tile_h) * (cycle_per_tile / num_cores)
#           - cycle_per_tile is idealistic number of cycles required for Tensix engine to perform matmul on 2 tiles and it is based on
#             math fidelity: for LoFi it is 16, for HiFi2 it is 32, for HiFi3 it is 48 and for HiFi4 it is 64.
#           - num_cores is number of cores in grid that is used to execute matmul operation, 8x8 grid will have 64 cores.
#       - inference_cycle is calculated as: inference_time_avg * device_freq[Hz]
#           - device_freq is fixed value common for specific architecture, for Wormhole B0 it is 1000MHz, for Blackhole it is 1350MHz
#             This value can change and needs to be updated in code until there is python support to read device frequency from device.
#   - Host based utilization uses fixed device frequency value, in reality device frequency can vary due to frequency throttling. Calculated value
#     is worst case scenario for host based utilization since frequency throttling will reduce device frequency and thus increase utilization.
#   - Device based utilization is calculated using formula: ideal_cycle / trisc1_kernel_duration
#       - trisc1_kernel_duration is read from profiler log and it is average duration of TRISC1 kernel in number of cycles.

import csv
import itertools
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import ttnn
from loguru import logger
from models.common.utility_functions import is_blackhole, is_wormhole_b0, profiler
from tracy.common import PROFILER_DEVICE_SIDE_LOG, PROFILER_LOGS_DIR, rm
from tracy.device_post_proc_config import default_setup
from tracy.process_device_log import import_log_run_stats

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG
GEMM_FLOPS_BENCHMARK_ENV = "TTNN_RUN_GEMM_FLOPS_BENCHMARK"


SKIPPABLE_RUNTIME_ERROR_SUBSTRINGS = (
    "beyond max l1 size",
    "clash with l1 buffers",
    "does not fit",
    "failed to allocate",
    "insufficient",
    "invalid",
    "invalid sharding",
    "shard shape",
    "tile",
    "l1 memory",
    "not enough l1",
    "out of memory",
    "unable to find subblock",
    "unsupported",
    "in0_block_w is 0",
    "per_core_m is 0",
    "per_core_n is 0",
    "out_block_h is 0",
    "out_block_w is 0",
    "invalid matmul block params",
    "validation",
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

SUBBLOCK_HW_CHOICES = [
    (4, 2),
    (2, 4),
    (8, 1),
    (1, 8),  # subblock_hw = 8
    (7, 1),
    (1, 7),  # subblock_hw = 7
    (3, 2),
    (2, 3),
    (6, 1),
    (1, 6),  # subblock_hw = 6
    (5, 1),
    (1, 5),  # subblock_hw = 5
    (2, 2),
    (4, 1),
    (1, 4),  # subblock_hw = 4
    (3, 1),
    (1, 3),  # subblock_hw = 3
    (2, 1),
    (1, 2),  # subblock_hw = 2
    (1, 1),  # subblock_hw = 1
]

FIDELITY_CYCLES = {
    ttnn.MathFidelity.LoFi: 16,
    ttnn.MathFidelity.HiFi2: 32,
    ttnn.MathFidelity.HiFi3: 48,
    ttnn.MathFidelity.HiFi4: 64,
}


def get_subblock_sizes(m_tiles_per_core, n_tiles_per_core, out_sharded=False, fp32_dest_acc_en=False):
    for subblock_hw in SUBBLOCK_HW_CHOICES:
        out_subblock_h = subblock_hw[0]
        out_subblock_w = subblock_hw[1]

        if fp32_dest_acc_en:
            if (out_subblock_h * out_subblock_w) > 4:
                continue

        if out_sharded:
            if n_tiles_per_core % out_subblock_w != 0 or out_subblock_h != 1:
                continue

        if m_tiles_per_core % out_subblock_h == 0 and n_tiles_per_core % out_subblock_w == 0:
            return (out_subblock_h, out_subblock_w)

    return (1, 1)


def get_device_frequency():
    """Get default device frequency[MHz] based on architecture."""
    if is_wormhole_b0():
        return 1000
    elif is_blackhole():
        return 1350
    else:
        return None


def get_profiler_data():
    """Import profiler log and return device freq + TRISC1 kernel duration."""
    setup = default_setup()
    setup.deviceInputLog = profiler_log_path
    deviceData = import_log_run_stats(setup)
    data = {}

    # Add device frequency from profiler log
    data["device_freq"] = deviceData["deviceInfo"]["freq"]
    # Add TRISC1-Math kernel average duration time - TRISC1 kernel zones is always present in profiler log
    data["trisc1_kernel_duration"] = deviceData["devices"][0]["cores"]["DEVICE"]["analysis"][
        "device_trisc1_kernel_duration"
    ]["stats"]["Average"]

    return data


def get_profiler_build_enabled():
    return os.getenv("TT_METAL_DEVICE_PROFILER") is not None


def is_skippable_benchmark_runtime_error(error):
    """Return whether a benchmark config failed because it is not runnable for this shape/device."""
    message = str(error).lower()
    return any(substring in message for substring in SKIPPABLE_RUNTIME_ERROR_SUBSTRINGS)


def find_largest_divisor(n, max_divisor=8):
    for divisor in range(max_divisor, 0, -1):
        if n % divisor == 0:
            return divisor
    return 1


def get_dram_num_banks(device):
    """Match test_matmul_in1_dram_sharded_tiny_tile: WH uses 12 banks, BH uses dram grid width."""
    if is_wormhole_b0():
        return 12
    if is_blackhole():
        return device.dram_grid_size().x
    dram_grid = device.dram_grid_size()
    return dram_grid.x * dram_grid.y


def pad_to_dram_banks(num, tile_size, num_banks):
    lcm = tile_size * num_banks
    remainder = num % lcm
    if remainder == 0:
        return num
    return num + (lcm - remainder)


def find_width_shard_grid(width, tile_size, max_cols, max_rows, target_cores=32):
    """Find a (cols, rows, num_cores) grid that evenly width-shards `width` in tile units."""
    width_tiles = width // tile_size
    max_cores = max_cols * max_rows
    possible_cores = [cores for cores in range(1, max_cores + 1) if width_tiles % cores == 0]
    possible_cores.sort(key=lambda cores: abs(cores - min(target_cores, width_tiles)))
    for cores in possible_cores:
        for rows in range(1, max_rows + 1):
            if cores % rows != 0:
                continue
            cols = cores // rows
            if cols <= max_cols:
                return cols, rows, cores
    return None, None, None


def get_dram_sharded_layout(device, m, k, n, tile_h, tile_w):
    """Build memory configs and program config for DRAM-sharded matmul."""
    compute_grid = device.compute_with_storage_grid_size()
    cols, rows, shard_num_cores = find_width_shard_grid(k, tile_w, compute_grid.x, compute_grid.y)
    if shard_num_cores is None:
        raise RuntimeError("invalid sharding core_grid")

    num_banks = get_dram_num_banks(device)
    n_padded = pad_to_dram_banks(n, tile_w, num_banks)

    in0_memory_config = ttnn.create_sharded_memory_config(
        (1, 1, m, k),
        core_grid=ttnn.CoreGrid(y=rows, x=cols),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    dram_grid = device.dram_grid_size()
    dram_core_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid.x - 1, dram_grid.y - 1))}
    )
    in1_shard_shape = [k, n_padded // num_banks]
    in1_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_core_grid, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )
    out_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    k_tiles_per_core = (k // tile_w) // shard_num_cores
    n_tiles_per_core = max((n_padded // tile_w) // shard_num_cores, 1)
    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=find_largest_divisor(k_tiles_per_core),
        per_core_M=m // tile_h,
        per_core_N=n_tiles_per_core,
        fused_activation=None,
    )

    return {
        "in0_memory_config": in0_memory_config,
        "in1_memory_config": in1_memory_config,
        "out_mem_config": out_mem_config,
        "output_tile": ttnn.Tile([tile_h, tile_w]),
        "program_config": program_config,
        "n": n_padded,
    }


# ---------------------------------------------------------------------------
# Benchmark measurement helpers
# ---------------------------------------------------------------------------


def run_matmul_measurement(
    device,
    in0_t,
    in1_t,
    matmul_fn,
    use_trace,
    num_warmup_iterations,
    num_measurement_iterations,
    calc_device_utilization,
):
    """Run warmup + timed measurement loop, return (inference_time_avg, trisc1_kernel_duration or None)."""

    # Initial run to compile
    output_t = matmul_fn(in0_t, in1_t)

    # Warmup
    for _ in range(num_warmup_iterations):
        output_t = matmul_fn(in0_t, in1_t)

    if calc_device_utilization:
        ttnn.ReadDeviceProfiler(device)
        rm(profiler_log_path)

    ttnn.synchronize_device(device)

    # Measurement
    if use_trace:
        tid = None
        trace_capture_ended = False
        try:
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            for _ in range(num_measurement_iterations):
                output_t = matmul_fn(in0_t, in1_t)
            ttnn.end_trace_capture(device, tid, cq_id=0)
            trace_capture_ended = True

            profiler.start("run")
            try:
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(device)
            finally:
                profiler.end("run")
        finally:
            if tid is not None:
                try:
                    if not trace_capture_ended:
                        ttnn.end_trace_capture(device, tid, cq_id=0)
                finally:
                    ttnn.release_trace(device, tid)
    else:
        profiler.start("run")
        for _ in range(num_measurement_iterations):
            output_t = matmul_fn(in0_t, in1_t)
        ttnn.synchronize_device(device)
        profiler.end("run")

    trisc1_kernel_duration = None
    if calc_device_utilization:
        ttnn.ReadDeviceProfiler(device)
        profiler_data = get_profiler_data()
        trisc1_kernel_duration = profiler_data["trisc1_kernel_duration"]

    inference_time_avg = profiler.get("run") / num_measurement_iterations
    return inference_time_avg, trisc1_kernel_duration, output_t


def compute_utilization(
    m, k, n, tile_h, tile_w, math_fidelity, inference_time_avg, trisc1_kernel_duration, grid_size, compute_grid_size
):
    """Compute all utilization metrics. Returns a dict."""
    cycle_per_tile = FIDELITY_CYCLES[math_fidelity]
    device_freq = get_device_frequency()
    num_cores_user_grid = grid_size[0] * grid_size[1]
    num_cores_full_grid = compute_grid_size.x * compute_grid_size.y

    ideal_cycle_full_grid = m * k * n / tile_h / tile_w / 32 * cycle_per_tile / num_cores_full_grid
    ideal_cycle_user_grid = m * k * n / tile_h / tile_w / 32 * cycle_per_tile / num_cores_user_grid
    inference_cycle = inference_time_avg * device_freq * 1e6

    tflops = 2 * m * k * n / 1e12 / inference_time_avg
    utilization_full_grid = ideal_cycle_full_grid / inference_cycle
    utilization_user_grid = ideal_cycle_user_grid / inference_cycle

    result = {
        "tflops": tflops,
        "inference_time_avg": inference_time_avg,
        "utilization_user_grid": utilization_user_grid,
        "utilization_full_grid": utilization_full_grid,
    }

    if trisc1_kernel_duration is not None:
        result["utilization_user_grid_device"] = ideal_cycle_user_grid / np.mean(trisc1_kernel_duration)
        result["utilization_full_grid_device"] = ideal_cycle_full_grid / np.mean(trisc1_kernel_duration)

    return result


# ---------------------------------------------------------------------------
# Shape and configuration data
# ---------------------------------------------------------------------------

# Independent M/K/N dimension lists in elements. The benchmark runs the full
# Cartesian product of these lists (all m × k × n combinations).
_matmul_dim_values = [
    32,
    64,
    128,
    256,
    512,
    1024,
]
matmul_m_values = _matmul_dim_values
matmul_k_values = _matmul_dim_values
matmul_n_values = _matmul_dim_values

# Per-dtype, per-mode overrides for tuning params: (in0_block_w_div, num_out_blocks_h, num_out_blocks_w)
# Default is (1, 1, 1) for shapes not listed here.
#
# L1-sharded output (mcast_2d_l1) must satisfy matmul validation:
#   out_block_w == per_core_N or out_block_h == 1
# so only one of num_out_blocks_h / num_out_blocks_w may be > 1 (and only when it yields out_block_h == 1).
# DRAM output modes can use more aggressive multi-block splits to reduce L1 pressure.
tuning_overrides_l1 = {
    ttnn.bfloat16: {
        (2560, 4608, 4608): (2, 1, 1),
        (3840, 4608, 4608): (4, 1, 1),
    },
    ttnn.bfloat8_b: {
        (3840, 4608, 4608): (2, 1, 1),
        (3840, 4608, 6144): (2, 1, 1),
    },
    ttnn.bfloat4_b: {
        (3840, 6144, 6144): (2, 1, 1),
        (5120, 6144, 6144): (2, 1, 1),
    },
}

tuning_overrides_dram = {
    ttnn.bfloat16: {
        (3840, 4608, 6144): (2, 1, 1),
        (3840, 6144, 6144): (2, 1, 1),
        (5120, 6144, 6144): (1, 2, 2),
        (10240, 12288, 12288): (2, 4, 4),
        (20480, 24576, 24576): (4, 8, 8),
    },
    ttnn.bfloat8_b: {
        (5120, 6144, 6144): (1, 2, 2),
        (10240, 12288, 12288): (2, 4, 4),
        (20480, 24576, 24576): (4, 8, 8),
    },
    ttnn.bfloat4_b: {
        (10240, 12288, 12288): (2, 2, 2),
        (20480, 24576, 24576): (4, 4, 4),
    },
}

# Valid math fidelities per dtype
dtype_fidelities = {
    ttnn.bfloat16: [ttnn.MathFidelity.HiFi4],  # , ttnn.MathFidelity.HiFi2],
    # ttnn.bfloat8_b: [ttnn.MathFidelity.HiFi2],#, ttnn.MathFidelity.LoFi],
    # ttnn.bfloat4_b: [ttnn.MathFidelity.LoFi],
}

# Benchmark modes — keep in sync with tech_reports/GEMM_FLOPS/benchmark_modes.py
matmul_modes = [
    "oob",
    # "reuse_dram",
    "mcast_2d_l1",
    "mcast_2d_dram",
    # "mcast_1d_in0",
    # "mcast_1d_out",
    # "dram_sharded",
]


def get_tuning_params(dtype, mkn_shape, mode):
    """Get (in0_block_w_div, num_out_blocks_h, num_out_blocks_w) for a given dtype, (m,k,n), and mode."""
    if mode == "mcast_2d_l1":
        overrides = tuning_overrides_l1.get(dtype, {})
    elif mode in ("mcast_2d_dram", "mcast_1d_in0", "mcast_1d_out", "reuse_dram"):
        overrides = tuning_overrides_dram.get(dtype, {})
    else:
        return (1, 1, 1)
    return overrides.get(mkn_shape, (1, 1, 1))


def mode_uses_sharded_output(mode):
    return mode in ("dram_sharded", "mcast_2d_l1")


def build_mcast_2d_block_dims(m, k, n, grid_size, tile_h, tile_w, in0_block_w_div):
    """Compute per-core 2D multicast block dims in tile units (matches test_matmul.py)."""
    grid_x, grid_y = grid_size
    k_tiles = k // tile_w
    m_tiles = m // tile_h
    n_tiles = n // tile_w
    per_core_k_tiles = k_tiles // grid_x
    in0_block_w = per_core_k_tiles // in0_block_w_div
    per_core_m = m_tiles // grid_y
    per_core_n = n_tiles // grid_x
    return in0_block_w, per_core_m, per_core_n


def validate_block_params(block_params, mode):
    """Raise a skippable error when explicit block dims are invalid for this shape/grid."""
    checks = {
        "in0_block_w": block_params.get("in0_block_w"),
        "per_core_m": block_params.get("per_core_m"),
        "per_core_n": block_params.get("per_core_n"),
        "out_block_h": block_params.get("out_block_h"),
        "out_block_w": block_params.get("out_block_w"),
    }
    for name, value in checks.items():
        if value is None or value == 0:
            raise RuntimeError(f"invalid matmul block params: {name} is 0 for mode {mode}")


def build_mcast_block_params(m, k, n, grid_size, tile_h, tile_w, mode, tuning_params, out_sharded):
    """Compute shared block/subblock parameters for multicast program configs."""
    in0_block_w_div, num_out_blocks_h, num_out_blocks_w = tuning_params
    num_cores = grid_size[0] * grid_size[1]
    k_tiles = k // tile_w

    if mode in ("mcast_1d_in0", "mcast_1d_out"):
        mcast_in0 = mode == "mcast_1d_in0"
        if mcast_in0:
            per_core_m = m // tile_h
            per_core_n = (n // tile_w + num_cores - 1) // num_cores
        else:
            per_core_m = (m // tile_h + num_cores - 1) // num_cores
            per_core_n = n // tile_w
        in0_block_w = 2 if k_tiles % 2 == 0 else 1
    else:
        mcast_in0 = None
        in0_block_w, per_core_m, per_core_n = build_mcast_2d_block_dims(
            m, k, n, grid_size, tile_h, tile_w, in0_block_w_div
        )

    out_block_h = per_core_m // num_out_blocks_h
    out_block_w = per_core_n // num_out_blocks_w
    out_subblock_h, out_subblock_w = get_subblock_sizes(out_block_h, out_block_w, out_sharded)
    block_params = {
        "in0_block_w": in0_block_w,
        "per_core_m": per_core_m,
        "per_core_n": per_core_n,
        "out_block_h": out_block_h,
        "out_block_w": out_block_w,
        "out_subblock_h": out_subblock_h,
        "out_subblock_w": out_subblock_w,
        "mcast_in0": mcast_in0,
    }
    validate_block_params(block_params, mode)
    return block_params


def build_program_config(mode, m, k, n, grid_size, tile_h, tile_w, tuning_params, out_sharded, device):
    """Build the explicit matmul program config for a benchmark mode (None for oob)."""
    if mode == "oob":
        return None

    if mode == "dram_sharded":
        dram_layout = get_dram_sharded_layout(device, m, k, n, tile_h, tile_w)
        return dram_layout["program_config"]

    if mode == "reuse_dram":
        in0_block_w_div, _, _ = tuning_params
        per_core_m = m // tile_h
        per_core_n = n // tile_w
        in0_block_w = (k // tile_w) // in0_block_w_div
        out_subblock_h, out_subblock_w = get_subblock_sizes(per_core_m, per_core_n, out_sharded=False)
        validate_block_params(
            {
                "in0_block_w": in0_block_w,
                "per_core_m": per_core_m,
                "per_core_n": per_core_n,
                "out_block_h": per_core_m,
                "out_block_w": per_core_n,
            },
            mode,
        )
        return ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
        )

    block_params = build_mcast_block_params(m, k, n, grid_size, tile_h, tile_w, mode, tuning_params, out_sharded)

    if mode in ("mcast_2d_l1", "mcast_2d_dram"):
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=block_params["in0_block_w"],
            out_subblock_h=block_params["out_subblock_h"],
            out_subblock_w=block_params["out_subblock_w"],
            out_block_h=block_params["out_block_h"],
            out_block_w=block_params["out_block_w"],
            per_core_M=block_params["per_core_m"],
            per_core_N=block_params["per_core_n"],
            transpose_mcast=False,
            fused_activation=None,
        )

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=block_params["in0_block_w"],
        out_subblock_h=block_params["out_subblock_h"],
        out_subblock_w=block_params["out_subblock_w"],
        out_block_h=block_params["out_block_h"],
        out_block_w=block_params["out_block_w"],
        per_core_M=block_params["per_core_m"],
        per_core_N=block_params["per_core_n"],
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=block_params["mcast_in0"],
    )


def generate_torch_matrices(m, k, n):
    """Create host-side input matrices for a benchmark shape."""
    in0 = torch.ones([1, 1, m, k]).bfloat16()
    in1 = torch.randn([1, 1, k, n]).bfloat16()
    return in0, in1


def prepare_benchmark_tensors(device, mode, m, k, n, grid_size, tile_h, tile_w, dtype, in0, in1):
    """Create input tensors for a benchmark mode. Returns dict with tensors and memory metadata.

    Memory layouts follow tests/ttnn/unit_tests/operations/matmul/test_matmul.py:
      - run_matmul_2d_tiny_tile / run_matmul_1d_tiny_tile for interleaved multicast/reuse paths
      - test_matmul_in1_dram_sharded_tiny_tile for dram_sharded
    """
    if mode == "dram_sharded":
        dram_layout = get_dram_sharded_layout(device, m, k, n, tile_h, tile_w)
        n = dram_layout["n"]
        in0_memory_config = dram_layout["in0_memory_config"]
        in1_memory_config = dram_layout["in1_memory_config"]
        out_mem_config = dram_layout["out_mem_config"]
        output_tile = dram_layout["output_tile"]
        in0_sharded = True
        out_sharded = True
        in0_storage_type = "L1-sharded"
        in1_storage_type = "DRAM-sharded"
        out_storage_type = "L1-sharded"
    elif mode == "mcast_2d_l1":
        in0_memory_config = ttnn.create_sharded_memory_config(
            (1, 1, m, k),
            core_grid=ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        in1_memory_config = ttnn.DRAM_MEMORY_CONFIG
        out_mem_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            buffer_type=ttnn.BufferType.L1,
        )
        output_tile = ttnn.Tile([tile_h, 32]) if tile_h <= 16 else ttnn.Tile([tile_h, tile_w])
        in0_sharded = True
        out_sharded = True
        in0_storage_type = "L1-sharded"
        in1_storage_type = "DRAM"
        out_storage_type = "L1-sharded"
    elif mode == "oob":
        in0_memory_config = ttnn.DRAM_MEMORY_CONFIG
        in1_memory_config = ttnn.DRAM_MEMORY_CONFIG
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG
        output_tile = ttnn.Tile([tile_h, tile_w])
        in0_sharded = False
        out_sharded = False
        in0_storage_type = "DRAM"
        in1_storage_type = "DRAM"
        out_storage_type = "DRAM"
    else:
        # reuse_dram, mcast_2d_dram, mcast_1d_in0, mcast_1d_out: interleaved L1 activation/output, DRAM weights
        in0_memory_config = ttnn.L1_MEMORY_CONFIG
        in1_memory_config = ttnn.DRAM_MEMORY_CONFIG
        out_mem_config = ttnn.L1_MEMORY_CONFIG
        output_tile = ttnn.Tile([tile_h, tile_w])
        in0_sharded = False
        out_sharded = False
        in0_storage_type = "L1"
        in1_storage_type = "DRAM"
        out_storage_type = "L1"

    in0_t = ttnn.from_torch(
        in0,
        tile=ttnn.Tile((tile_h, 32)),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_memory_config,
    )
    in1_t = ttnn.from_torch(
        in1,
        tile=ttnn.Tile((32, tile_w)),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in1_memory_config,
    )

    return {
        "in0_t": in0_t,
        "in1_t": in1_t,
        "in0_sharded": in0_sharded,
        "out_sharded": out_sharded,
        "in0_storage_type": in0_storage_type,
        "in1_storage_type": in1_storage_type,
        "out_storage_type": out_storage_type,
        "out_mem_config": out_mem_config,
        "output_tile": output_tile,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Main benchmark test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.getenv(GEMM_FLOPS_BENCHMARK_ENV) != "1",
    reason=f"Benchmark is manual-only; set {GEMM_FLOPS_BENCHMARK_ENV}=1 to run",
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 3855488}], indirect=True)
@pytest.mark.parametrize("tile_h", [32])
@pytest.mark.parametrize("tile_w", [32])
@pytest.mark.parametrize("num_warmup_iterations", [5])
@pytest.mark.parametrize("num_measurement_iterations", [20])
def test_matmul_2d_host_perf(
    device,
    grid_size,
    tile_h,
    tile_w,
    num_warmup_iterations,
    num_measurement_iterations,
):
    ENVS = dict(os.environ)
    TT_METAL_HOME = Path(ENVS["TT_METAL_HOME"])
    ARTIFACTS_DIR = TT_METAL_HOME / "generated"
    FILE_NAME = ARTIFACTS_DIR / "matmul_benchmark_report.csv"

    # Calculate device utilization only when profiler build is enabled.
    # If profiler build is enabled, kernel execution time is available in profiler log by default.
    calc_device_utilization = get_profiler_build_enabled()

    # Get maximum available grid size for compute from device
    compute_grid_size = device.compute_with_storage_grid_size()
    # If user did not specify grid size, use maximum available grid size for compute
    if grid_size is None:
        grid_size = (compute_grid_size.x, compute_grid_size.y)
    # Check if requested grid size is within available compute grid size, skip test if not
    if compute_grid_size.y < grid_size[1] or compute_grid_size.x < grid_size[0]:
        pytest.skip(
            f"Skipping test as requested compute grid size {grid_size} exceeds available compute grid {compute_grid_size}"
        )

    with open(FILE_NAME, mode="w", newline="") as file:
        writer = csv.writer(file)
        header = [
            "m",
            "k",
            "n",
            "mode",
            "use_trace",
            "grid_size",
            "in0_sharded",
            "out_sharded",
            "in0_storage_type",
            "in1_storage_type",
            "out_storage_type",
            "dtype",
            "math_fidelity",
            "inference_time_avg [ns]",
            "TFLOPs (avg)",
            f"Host based utilization[%] (vs user selected grid {grid_size[0]}x{grid_size[1]})",
            f"Host based utilization[%] (vs full available grid {compute_grid_size.x}x{compute_grid_size.y})",
        ]
        if calc_device_utilization:
            header.extend(
                [
                    f"Device based utilization[%] (vs user selected grid {grid_size[0]}x{grid_size[1]})",
                    f"Device based utilization[%] (vs full available grid {compute_grid_size.x}x{compute_grid_size.y})",
                ]
            )
        writer.writerow(header)

        for dtype, fidelities in dtype_fidelities.items():
            for m, k, n in itertools.product(matmul_m_values, matmul_k_values, matmul_n_values):
                in0_torch, in1_torch = generate_torch_matrices(m, k, n)
                dram_in1_torch = in1_torch
                if "dram_sharded" in matmul_modes:
                    n_padded = get_dram_sharded_layout(device, m, k, n, tile_h, tile_w)["n"]
                    if n_padded != n:
                        dram_in1_torch = torch.randn([1, 1, k, n_padded]).bfloat16()

                for mode in matmul_modes:
                    in0_t = None
                    in1_t = None

                    try:
                        tuning_params = get_tuning_params(dtype, (m, k, n), mode)
                        out_sharded = mode_uses_sharded_output(mode)
                        program_config = build_program_config(
                            mode, m, k, n, grid_size, tile_h, tile_w, tuning_params, out_sharded, device
                        )

                        mode_in1_torch = dram_in1_torch if mode == "dram_sharded" else in1_torch
                        tensor_setup = prepare_benchmark_tensors(
                            device,
                            mode,
                            m,
                            k,
                            n,
                            grid_size,
                            tile_h,
                            tile_w,
                            dtype,
                            in0_torch,
                            mode_in1_torch,
                        )
                        in0_t = tensor_setup["in0_t"]
                        in1_t = tensor_setup["in1_t"]
                        in0_sharded = tensor_setup["in0_sharded"]
                        out_sharded = tensor_setup["out_sharded"]
                        in0_storage_type = tensor_setup["in0_storage_type"]
                        in1_storage_type = tensor_setup["in1_storage_type"]
                        out_storage_type = tensor_setup["out_storage_type"]
                        out_mem_config = tensor_setup["out_mem_config"]
                        output_tile = tensor_setup["output_tile"]
                        effective_n = tensor_setup["n"]

                        for use_trace in [False, True]:
                            for math_fidelity in fidelities:
                                profiler.clear()

                                compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                                    math_fidelity=math_fidelity,
                                    math_approx_mode=True,
                                    fp32_dest_acc_en=False,
                                    packer_l1_acc=True,
                                    throttle_level=ttnn.ThrottleLevel.NO_THROTTLE,
                                )

                                output_t = None
                                try:
                                    if mode == "oob":

                                        def matmul_fn(a, b):
                                            return ttnn.matmul(a, b, compute_kernel_config=compute_kernel_config)

                                    else:

                                        def matmul_fn(a, b):
                                            return ttnn.matmul(
                                                a,
                                                b,
                                                program_config=program_config,
                                                memory_config=out_mem_config,
                                                dtype=dtype,
                                                compute_kernel_config=compute_kernel_config,
                                                output_tile=output_tile,
                                            )

                                    # --- Measure ---
                                    inference_time_avg, trisc1_dur, output_t = run_matmul_measurement(
                                        device,
                                        in0_t,
                                        in1_t,
                                        matmul_fn,
                                        use_trace,
                                        num_warmup_iterations,
                                        num_measurement_iterations,
                                        calc_device_utilization,
                                    )

                                    # --- Compute utilization ---
                                    metrics = compute_utilization(
                                        m,
                                        k,
                                        effective_n,
                                        tile_h,
                                        tile_w,
                                        math_fidelity,
                                        inference_time_avg,
                                        trisc1_dur,
                                        grid_size,
                                        compute_grid_size,
                                    )

                                    device_util_str = ""
                                    if "utilization_user_grid_device" in metrics:
                                        device_util_str = (
                                            f", device util (user grid {grid_size[0]}x{grid_size[1]}): "
                                            f"{metrics['utilization_user_grid_device']*100:.2f}%, "
                                            f"device util (full grid {compute_grid_size.x}x{compute_grid_size.y}): "
                                            f"{metrics['utilization_full_grid_device']*100:.2f}%"
                                        )
                                    logger.info(
                                        f"[{mode}] {dtype} {math_fidelity} trace={use_trace} "
                                        f"M*K*N={m}*{k}*{effective_n} — "
                                        f"inference time (avg): {inference_time_avg:.9f}, "
                                        f"tflops (avg): {metrics['tflops']:.2f}, "
                                        f"utilization (vs user selected grid {grid_size[0]}x{grid_size[1]}): "
                                        f"{metrics['utilization_user_grid']*100:.2f}%, "
                                        f"utilization (vs full available grid {compute_grid_size.x}x{compute_grid_size.y}): "
                                        f"{metrics['utilization_full_grid']*100:.2f}%"
                                        f"{device_util_str}"
                                    )

                                    # --- Cleanup and write CSV ---
                                    ttnn.to_torch(output_t)

                                    csv_data = [
                                        m,
                                        k,
                                        effective_n,
                                        mode,
                                        f"{use_trace}",
                                        grid_size,
                                        in0_sharded,
                                        out_sharded,
                                        in0_storage_type,
                                        in1_storage_type,
                                        out_storage_type,
                                        dtype,
                                        math_fidelity,
                                        f"{inference_time_avg * 1e9:.2f}",
                                        f"{metrics['tflops']:.2f}",
                                        f"{metrics['utilization_user_grid'] * 100:.2f}",
                                        f"{metrics['utilization_full_grid'] * 100:.2f}",
                                    ]
                                    if calc_device_utilization:
                                        csv_data.extend(
                                            [
                                                f"{metrics['utilization_user_grid_device'] * 100:.2f}",
                                                f"{metrics['utilization_full_grid_device'] * 100:.2f}",
                                            ]
                                        )
                                    writer.writerow(csv_data)
                                    file.flush()

                                except RuntimeError as e:
                                    if not is_skippable_benchmark_runtime_error(e):
                                        raise
                                    logger.warning(
                                        f"Skipping [{mode}] {dtype} {math_fidelity} "
                                        f"({m},{k},{effective_n}) trace={use_trace} — {e}"
                                    )
                                    continue
                                finally:
                                    if output_t is not None:
                                        ttnn.deallocate(output_t)

                    except RuntimeError as e:
                        if not is_skippable_benchmark_runtime_error(e):
                            raise
                        logger.warning(f"Skipping [{mode}] {dtype} ({m},{k},{n}) — {e}")
                        continue
                    finally:
                        for tensor in (in0_t, in1_t):
                            if tensor is not None:
                                ttnn.deallocate(tensor)
