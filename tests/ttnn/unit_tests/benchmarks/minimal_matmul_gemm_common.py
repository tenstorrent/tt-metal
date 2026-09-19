# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared definitions for the partner GEMM benchmark of ttnn.experimental.minimal_matmul.

Used by test_minimal_matmul_block_sweep.py (phase 1, offline block sweep) and
test_minimal_matmul_benchmark.py (phase 2, partner-facing sheet CSV). Both files describe the method in
their header comments; this module holds what they share:

- the partner sheet shapes and the sheet data-type column -> ttnn dtype / fidelity / accumulation map
- the theoretical peak formula   num_cores * freq_hz * (8*16*16) * 2 / fidelity_passes / 1e12
- the measurement loop (compile run, warmup, eager loop or captured trace, optional TRISC1 profiler read)
- metrics, a cheap row-slice PCC check, input creation, grid resolution and the skippable-error filter
"""

import math
import os
import time

import numpy as np
import pytest
import torch
import ttnn
from models.common.utility_functions import is_blackhole, is_wormhole_b0
from tracy.common import PROFILER_DEVICE_SIDE_LOG, PROFILER_LOGS_DIR, rm
from tracy.device_post_proc_config import default_setup
from tracy.process_device_log import import_log_run_stats

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG

GEMM_FLOPS_BENCHMARK_ENV = "TTNN_RUN_GEMM_FLOPS_BENCHMARK"
DEVICE_PARAMS = {"l1_small_size": 24576, "trace_region_size": 8388608}
FLOP_PER_CORE_PER_CYCLE = 8 * 16 * 16 * 2  # 8x16x16 MAC array, multiply + add

# ---------------------------------------------------------------------------
# Sheet definition
# ---------------------------------------------------------------------------

# Partner sheet rows, column order (M, N, K): input is [M, K], weight is [K, N].
SHEET_SHAPES = [
    (8192, 4096, 1536),
    (8192, 4096, 12288),
    (8192, 5120, 25600),
    (8192, 8192, 8192),
    (8192, 16384, 16384),
    (8192, 24576, 4096),
    (8192, 32768, 16384),
    (16384, 16384, 16384),
    (32, 2048, 2048),
    (32, 4096, 4096),
]

FP16_NOTE = "not supported: no FLOAT16 tensor dtype in tt-metal (hardware format exists, unshipped for compute)"

# Sheet data-type column -> how it is run. "passes" is the fidelity pass count in the peak formula,
# "measure_as" points a column at the column whose measurement it shares, dtype None is reported as N/A.
DTYPE_COLUMNS = {
    "BF16": dict(dtype=ttnn.bfloat16, fidelity=ttnn.MathFidelity.HiFi2, passes=2, fp32_acc=False, note=""),
    "FP16": dict(dtype=None, fidelity=ttnn.MathFidelity.HiFi4, passes=4, fp32_acc=False, note=FP16_NOTE),
    "TF32": dict(
        dtype=ttnn.float32,
        fidelity=ttnn.MathFidelity.HiFi4,
        passes=4,
        fp32_acc=True,
        measure_as="FP32",
        note="same measurement as FP32: the FPU consumes fp32 inputs as TF32",
    ),
    "FP32": dict(dtype=ttnn.float32, fidelity=ttnn.MathFidelity.HiFi4, passes=4, fp32_acc=True, note=""),
    "FP8": dict(
        dtype=ttnn.bfloat8_b, fidelity=ttnn.MathFidelity.LoFi, passes=1, fp32_acc=False, note="block-fp8 (bfloat8_b)"
    ),
    "FP4": dict(
        dtype=ttnn.bfloat4_b, fidelity=ttnn.MathFidelity.LoFi, passes=1, fp32_acc=False, note="block-fp4 (bfloat4_b)"
    ),
}
SHEET_COLUMN_ORDER = ["BF16", "FP16", "TF32", "FP32", "FP8", "FP4"]
# Columns that need their own measurement (TF32 shares FP32's run; FP16 has no tensor dtype).
SWEPT_COLUMNS = [c for c, s in DTYPE_COLUMNS.items() if s["dtype"] is not None and "measure_as" not in s]

TORCH_INPUT_DTYPE = {ttnn.float32: torch.float32}  # everything else is generated as torch.bfloat16

SKIPPABLE_RUNTIME_ERROR_SUBSTRINGS = (
    "beyond max l1 size",
    "circular buffer",
    "clash with l1 buffers",
    "does not fit",
    "failed to allocate",
    "insufficient",
    "invalid",
    "l1 memory",
    "not enough l1",
    "out of memory",
    "unable to find subblock",
    "unsupported",
    "validation",
)


def shape_id(shape):
    return "x".join(str(d) for d in shape)


def default_block_config(M, N, fp32_acc):
    """Mirror determine_default_block_sizes in minimal_matmul_program_descriptor.cpp (config=None path)."""
    if fp32_acc:
        return (8, 8, 8, 2, 2)
    return (8, 8, 8, 2, 4) if N >= M else (8, 8, 8, 4, 2)


def make_matmul_config(blocking, core_grid):
    """ttnn.MinimalMatmulConfig from (M_block, K_block, N_block, subblock_h, subblock_w) in tiles."""
    Mb, Kb, Nb, sh, sw = blocking
    return ttnn.MinimalMatmulConfig(
        M_block_size=Mb,
        K_block_size=Kb,
        N_block_size=Nb,
        subblock_h=sh,
        subblock_w=sw,
        compute_with_storage_grid_size=core_grid,
    )


def make_compute_kernel_config(device, spec):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=spec["fidelity"],
        math_approx_mode=True,
        fp32_dest_acc_en=spec["fp32_acc"],
        packer_l1_acc=True,
    )


def minimal_matmul_fn(config, compute_kernel_config):
    def op_fn(a, b):
        return ttnn.experimental.minimal_matmul(a, b, config=config, compute_kernel_config=compute_kernel_config)

    return op_fn


# ---------------------------------------------------------------------------
# Device / profiler helpers (adapted from test_benchmark.py)
# ---------------------------------------------------------------------------


def get_device_frequency_hz():
    """Nominal device clock. Replaced by the profiler-reported clock when a profiler build is used."""
    if is_wormhole_b0():
        return 1.0e9
    if is_blackhole():
        return 1.35e9
    raise RuntimeError("Unknown architecture: cannot derive the theoretical peak")


def get_profiler_build_enabled():
    return os.getenv("TT_METAL_DEVICE_PROFILER") is not None


def get_profiler_data():
    """Import profiler log and return device freq [MHz] + average TRISC1 kernel duration [cycles]."""
    setup = default_setup()
    setup.deviceInputLog = profiler_log_path
    device_data = import_log_run_stats(setup)
    return {
        "device_freq": device_data["deviceInfo"]["freq"],
        "trisc1_kernel_duration": device_data["devices"][0]["cores"]["DEVICE"]["analysis"][
            "device_trisc1_kernel_duration"
        ]["stats"]["Average"],
    }


def is_skippable_benchmark_runtime_error(error):
    """Return whether a config failed because it is not runnable for this shape/device (L1, validation)."""
    message = str(error).lower()
    return any(substring in message for substring in SKIPPABLE_RUNTIME_ERROR_SUBSTRINGS)


def runtime_error_reason(error):
    """First informative line of a TT_THROW message (the assert text rather than the TT_THROW location)."""
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    for line in lines:
        if any(sub in line.lower() for sub in SKIPPABLE_RUNTIME_ERROR_SUBSTRINGS):
            return line
    for line in lines:
        if "TT_THROW" not in line and not line.endswith(":"):
            return line
    return lines[0] if lines else str(error)


def theoretical_tflops(num_cores, freq_hz, passes):
    return num_cores * freq_hz * FLOP_PER_CORE_PER_CYCLE / passes / 1e12


def resolve_grid(device, grid_size):
    """grid_size is the conftest --grid-size fixture value read as (x, y), as in test_benchmark.py."""
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size is None:
        grid_size = (compute_grid_size.x, compute_grid_size.y)
    if compute_grid_size.x < grid_size[0] or compute_grid_size.y < grid_size[1]:
        pytest.skip(f"Requested grid {grid_size} exceeds available compute grid {compute_grid_size}")
    return grid_size


def make_inputs(device, M, N, K, dtype):
    torch_dtype = TORCH_INPUT_DTYPE.get(dtype, torch.bfloat16)
    in0 = torch.randn((M, K), dtype=torch_dtype)
    in1 = torch.randn((K, N), dtype=torch_dtype)
    in0_t = ttnn.from_torch(
        in0, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    in1_t = ttnn.from_torch(
        in1, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return in0, in1, in0_t, in1_t


# ---------------------------------------------------------------------------
# Correctness and measurement
# ---------------------------------------------------------------------------


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if denom > 0 else 0.0


def check_rows_pcc(in0, in1, out_t, num_rows=32):
    """Cheap correctness check: PCC of one tile-aligned band of num_rows output rows against torch. The band is
    sliced on device so only rows * N elements are transferred; a full reference for 16384^3 on CPU is
    impractical."""
    M, N = in0.shape[0], in1.shape[1]
    num_rows = min(num_rows, M)
    row0 = int(torch.randint(0, M // num_rows, (1,))) * num_rows if M > num_rows else 0
    band_t = ttnn.slice(out_t, [row0, 0], [row0 + num_rows, N])
    actual = ttnn.to_torch(band_t).float()
    ttnn.deallocate(band_t)
    reference = in0[row0 : row0 + num_rows].float() @ in1.float()
    return pcc(reference, actual)


def run_measurement(
    device,
    in0_t,
    in1_t,
    op_fn,
    use_trace,
    num_warmup_iterations,
    num_measurement_iterations,
    calc_device_utilization,
    trace_executions=1,
):
    """Compile run, warmup, then time num_measurement_iterations ops: either an eager loop or one captured
    trace executed trace_executions times (the fastest execution is kept, so contention from other threads
    driving other chips can only be filtered out, never inflate a result).

    Returns (inference_time_avg_s, trisc1_kernel_duration_cycles or None, device_freq_hz or None, output)."""
    output_t = op_fn(in0_t, in1_t)
    for _ in range(num_warmup_iterations):
        output_t = op_fn(in0_t, in1_t)

    if calc_device_utilization:
        ttnn.ReadDeviceProfiler(device)
        rm(profiler_log_path)

    ttnn.synchronize_device(device)

    if use_trace:
        tid = None
        trace_capture_ended = False
        try:
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            for _ in range(num_measurement_iterations):
                output_t = op_fn(in0_t, in1_t)
            ttnn.end_trace_capture(device, tid, cq_id=0)
            trace_capture_ended = True

            elapsed = math.inf
            for _ in range(trace_executions):
                t0 = time.perf_counter()
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(device)
                elapsed = min(elapsed, time.perf_counter() - t0)
        finally:
            if tid is not None:
                try:
                    if not trace_capture_ended:
                        ttnn.end_trace_capture(device, tid, cq_id=0)
                finally:
                    ttnn.release_trace(device, tid)
    else:
        t0 = time.perf_counter()
        for _ in range(num_measurement_iterations):
            output_t = op_fn(in0_t, in1_t)
        ttnn.synchronize_device(device)
        elapsed = time.perf_counter() - t0

    trisc1_kernel_duration = None
    device_freq_hz = None
    if calc_device_utilization:
        ttnn.ReadDeviceProfiler(device)
        profiler_data = get_profiler_data()
        trisc1_kernel_duration = float(np.mean(profiler_data["trisc1_kernel_duration"]))
        device_freq_hz = float(profiler_data["device_freq"]) * 1e6

    return elapsed / num_measurement_iterations, trisc1_kernel_duration, device_freq_hz, output_t


DEVICE_METRIC_COLUMNS = ["device_time_ms", "device_tflops", "device_utilization_pct"]


def compute_metrics(M, N, K, passes, num_cores, inference_time_avg, trisc1_kernel_duration, device_freq_hz):
    """TFLOP/s and utilization vs the theoretical peak; device_* entries only when a TRISC1 duration is given."""
    flops = 2 * M * N * K
    freq_hz = device_freq_hz if device_freq_hz else get_device_frequency_hz()
    peak = theoretical_tflops(num_cores, freq_hz, passes)
    tflops = flops / inference_time_avg / 1e12
    metrics = {
        "flops": flops,
        "time_avg_ms": inference_time_avg * 1e3,
        "measured_tflops": tflops,
        "theoretical_tflops": peak,
        "utilization_pct": tflops / peak * 100.0,
    }
    if trisc1_kernel_duration is not None:
        device_time_s = trisc1_kernel_duration / freq_hz
        device_tflops = flops / device_time_s / 1e12
        metrics["device_time_ms"] = device_time_s * 1e3
        metrics["device_tflops"] = device_tflops
        metrics["device_utilization_pct"] = device_tflops / peak * 100.0
    return metrics


def fmt(value, digits):
    return "" if value is None or value == "" else f"{value:.{digits}f}"


def device_metric_fields(metrics):
    """Formatted device_* CSV fields present in metrics."""
    return {c: fmt(metrics[c], 4 if c == "device_time_ms" else 2) for c in DEVICE_METRIC_COLUMNS if c in metrics}
