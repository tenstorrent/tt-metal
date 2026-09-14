# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Phase 2 of the partner GEMM benchmark: run ttnn.experimental.minimal_matmul once per sheet cell with
# a pre-determined blocking and write the sheet CSV (theoretical peak, measured, utilization).
#
# The blocking per (dtype column, M, N, K) comes from BEST_BLOCKING below, produced offline by the
# phase-1 sweep (test_minimal_matmul_block_sweep.py). Cells without an entry fall back to the op's own
# default blocking (config=None, marked "default") for bf16 accumulation, and to an L1-safe explicit
# blocking (marked "fallback-fp32") for fp32 accumulation: the op's 8/8/8 default overflows Blackhole L1
# with 4 KB fp32 tiles. A cell whose blocking is not runnable is written as N/A with the error in note.
#
# Output: $TT_METAL_HOME/generated/minimal_matmul_gemm_sheet.csv, one row per (shape, dtype column,
# eager/trace), in sheet order, plus one PEAK row per dtype column (best measured TFLOP/s over shapes).
#
# Sheet data-type column -> ttnn mapping (fidelity = number of passes through the 5x7-bit multiplier):
#   BF16 -> ttnn.bfloat16,  HiFi2 (2 passes), bf16 dest accumulation
#   FP16 -> N/A. tt-metal has no FLOAT16 tensor dtype (tt_metal/api/tt-metalium/tensor/tensor_types.hpp);
#           the hardware format exists but is unshipped for compute (tt_metal/jit_build/data_format.cpp).
#   TF32 -> ttnn.float32,   HiFi4 (4 passes), fp32 dest accumulation   (same measurement as FP32)
#   FP32 -> ttnn.float32,   HiFi4 (4 passes), fp32 dest accumulation   (FPU consumes fp32 inputs as TF32)
#   FP8  -> ttnn.bfloat8_b, LoFi  (1 pass),   bf16 dest accumulation   (block-fp8)
#   FP4  -> ttnn.bfloat4_b, LoFi  (1 pass),   bf16 dest accumulation
#   FP6  -> struck from the sheet (not natively supported).  HiFi3 is not used.
#
# Metrics:
#   flops               op count 2*M*N*K (operations, not per second)
#   theoretical_tflops  num_cores * freq_hz * (8*16*16) * 2 / passes / 1e12   (Blackhole: 1.35 GHz)
#   measured_tflops     flops / host-timed average op time (eager includes dispatch; trace strips it)
#   utilization_pct     measured_tflops / theoretical_tflops * 100
#   pcc                 PCC of 32 random output rows vs torch (cheap correctness check)
#   device_* columns    only with a profiler build (TT_METAL_DEVICE_PROFILER set): rates from the
#                       average TRISC1 (math) kernel duration, i.e. compute-only ("peak" in the sheet's
#                       sense of stripping data movement). Without the profiler, every row is a
#                       "sustained" full-pipeline measurement and PEAK rows are the best over shapes.
#
# Running (manual-only, gated by an env var so it never runs in CI by accident):
#   TTNN_RUN_GEMM_FLOPS_BENCHMARK=1 pytest tests/ttnn/unit_tests/benchmarks/test_minimal_matmul_benchmark.py
#   --grid-size 12x10   restrict the op to an x-by-y core grid (default: full compute grid)
#   -k eager / -k trace choose one dispatch mode
#   TTNN_MINIMAL_MATMUL_SHAPES=8192x8192x8192,32x2048x2048   run only these sheet rows (M x N x K)

import csv
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
SHEET_CSV_NAME = "minimal_matmul_gemm_sheet.csv"
SHAPES_ENV = "TTNN_MINIMAL_MATMUL_SHAPES"

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

# Sheet column order and how each is run. "measure_as" points a column at the column whose
# measurement it shares; "dtype" None means the column is reported as N/A.
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

TORCH_INPUT_DTYPE = {ttnn.float32: torch.float32}  # everything else is generated as torch.bfloat16

# Minimum acceptable PCC of the 32-row check per column (bfloat4_b carries a 4-bit mantissa).
PCC_THRESHOLD = {"BF16": 0.99, "TF32": 0.99, "FP32": 0.99, "FP8": 0.99, "FP4": 0.90}

# (dtype_column, M, N, K) -> (M_block, K_block, N_block, subblock_h, subblock_w), blocks in tiles.
# Paste the BEST_BLOCKING literal logged at the end of a phase-1 sweep here. Missing keys use the op's
# default blocking (8/8/8, sub-block 2x4 or 4x2 with bf16 accumulation, 2x2 with fp32 accumulation).
BEST_BLOCKING = {}

FLOP_PER_CORE_PER_CYCLE = 8 * 16 * 16 * 2  # 8x16x16 MAC array, multiply + add

# Explicit fallback for fp32 accumulation when BEST_BLOCKING has no entry (bf16 accumulation uses config=None).
FALLBACK_BLOCKING_FP32_ACC = (4, 8, 4, 2, 2)

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


def is_skippable_benchmark_runtime_error(error):
    """Return whether a cell failed because its blocking is not runnable for this shape/device (L1, validation)."""
    message = str(error).lower()
    return any(substring in message for substring in SKIPPABLE_RUNTIME_ERROR_SUBSTRINGS)


# ---------------------------------------------------------------------------
# Utility functions (adapted from test_benchmark.py)
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


def theoretical_tflops(num_cores, freq_hz, passes):
    return num_cores * freq_hz * FLOP_PER_CORE_PER_CYCLE / passes / 1e12


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if denom > 0 else 0.0


def check_rows_pcc(in0, in1, out_t, num_rows=32):
    """Cheap correctness check: PCC of a random row slice of the output against torch. A full reference
    for 16384^3 on CPU is impractical."""
    M = in0.shape[0]
    rows = torch.randperm(M)[: min(num_rows, M)]
    reference = in0[rows].float() @ in1.float()
    actual = ttnn.to_torch(out_t)[rows].float()
    return pcc(reference, actual)


def run_measurement(device, in0_t, in1_t, op_fn, use_trace, num_warmup_iterations, num_measurement_iterations):
    """Compile, warm up, then time num_measurement_iterations ops (eager or one trace).
    Returns (inference_time_avg_s, trisc1_kernel_duration_cycles or None, device_freq_hz or None, output)."""
    calc_device_utilization = get_profiler_build_enabled()

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
            output_t = op_fn(in0_t, in1_t)
        ttnn.synchronize_device(device)
        profiler.end("run")

    trisc1_kernel_duration = None
    device_freq_hz = None
    if calc_device_utilization:
        ttnn.ReadDeviceProfiler(device)
        profiler_data = get_profiler_data()
        trisc1_kernel_duration = float(np.mean(profiler_data["trisc1_kernel_duration"]))
        device_freq_hz = float(profiler_data["device_freq"]) * 1e6

    inference_time_avg = profiler.get("run") / num_measurement_iterations
    return inference_time_avg, trisc1_kernel_duration, device_freq_hz, output_t


def compute_metrics(M, N, K, passes, num_cores, inference_time_avg, trisc1_kernel_duration, device_freq_hz):
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


def blocking_for(dtype_column, M, N, K, core_grid, fp32_acc):
    """Return (ttnn.MinimalMatmulConfig or None, label) for a sheet cell."""
    blocking = BEST_BLOCKING.get((dtype_column, M, N, K))
    label = "best"
    if blocking is None:
        if not fp32_acc:
            return None, "default"
        blocking = FALLBACK_BLOCKING_FP32_ACC
        label = "fallback-fp32"
    Mb, Kb, Nb, sh, sw = blocking
    config = ttnn.MinimalMatmulConfig(
        M_block_size=Mb,
        K_block_size=Kb,
        N_block_size=Nb,
        subblock_h=sh,
        subblock_w=sw,
        compute_with_storage_grid_size=core_grid,
    )
    return config, f"{label} {Mb}/{Kb}/{Nb} sub {sh}x{sw}"


def shape_id(shape):
    return "x".join(str(d) for d in shape)


def selected_shapes():
    """SHEET_SHAPES, optionally narrowed by the TTNN_MINIMAL_MATMUL_SHAPES env var (comma-separated MxNxK)."""
    raw = os.getenv(SHAPES_ENV)
    if not raw:
        return SHEET_SHAPES
    wanted = {token.strip() for token in raw.split(",") if token.strip()}
    shapes = [s for s in SHEET_SHAPES if shape_id(s) in wanted]
    unknown = wanted - {shape_id(s) for s in shapes}
    assert not unknown, f"{SHAPES_ENV} names shapes not in SHEET_SHAPES: {sorted(unknown)}"
    return shapes


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

BASE_COLUMNS = [
    "M",
    "N",
    "K",
    "flops",
    "dtype_column",
    "ttnn_dtype",
    "fidelity",
    "passes",
    "accumulation",
    "grid",
    "blocking",
    "use_trace",
    "time_avg_ms",
    "theoretical_tflops",
    "measured_tflops",
    "utilization_pct",
    "pcc",
    "note",
]
DEVICE_COLUMNS = ["device_time_ms", "device_tflops", "device_utilization_pct"]


def csv_columns():
    return BASE_COLUMNS + (DEVICE_COLUMNS if get_profiler_build_enabled() else [])


def fmt(value, digits):
    return "" if value is None or value == "" else f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# Benchmark test: one item per dispatch mode, all sheet cells inside
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.getenv(GEMM_FLOPS_BENCHMARK_ENV) != "1",
    reason=f"Benchmark is manual-only; set {GEMM_FLOPS_BENCHMARK_ENV}=1 to run",
)
@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 8388608}], indirect=True)
@pytest.mark.parametrize("use_trace", [False, True], ids=["eager", "trace"])
@pytest.mark.parametrize("num_warmup_iterations", [5])
@pytest.mark.parametrize("num_measurement_iterations", [20])
def test_minimal_matmul_gemm_sheet(
    device,
    grid_size,
    use_trace,
    num_warmup_iterations,
    num_measurement_iterations,
):
    artifacts_dir = Path(os.environ["TT_METAL_HOME"]) / "generated"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    csv_path = artifacts_dir / SHEET_CSV_NAME
    # eager runs first and creates the file; trace appends so both modes land in one CSV.
    write_header = not use_trace or not csv_path.exists()

    grid_x, grid_y = resolve_grid(device, grid_size)
    core_grid = ttnn.CoreCoord(grid_x, grid_y)
    full_grid = device.compute_with_storage_grid_size()
    grid_str = f"{grid_x}x{grid_y}"
    num_cores = grid_x * grid_y

    columns = csv_columns()
    peak_by_column = {}
    pcc_failures = []
    unrunnable = []

    with open(csv_path, "w" if write_header else "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(columns)

        def write_row(row):
            writer.writerow([row.get(c, "") for c in columns])
            f.flush()

        for M, N, K in selected_shapes():
            flops = 2 * M * N * K
            measured = {}  # dtype_column -> (metrics, row_pcc, blocking_label, grid used)

            for dtype_column in SHEET_COLUMN_ORDER:
                spec = DTYPE_COLUMNS[dtype_column]
                row = dict(
                    M=M,
                    N=N,
                    K=K,
                    flops=flops,
                    dtype_column=dtype_column,
                    ttnn_dtype=str(spec["dtype"]) if spec["dtype"] is not None else "N/A",
                    fidelity=str(spec["fidelity"]),
                    passes=spec["passes"],
                    accumulation="fp32" if spec["fp32_acc"] else "bf16",
                    grid=grid_str,
                    use_trace=use_trace,
                    note=spec["note"],
                )

                if spec["dtype"] is None:
                    row.update(
                        blocking="N/A",
                        theoretical_tflops=fmt(
                            theoretical_tflops(num_cores, get_device_frequency_hz(), spec["passes"]), 2
                        ),
                        measured_tflops="N/A",
                        utilization_pct="N/A",
                    )
                    write_row(row)
                    logger.info(f"{dtype_column} {shape_id((M, N, K))}: N/A ({spec['note']})")
                    continue

                source_column = spec.get("measure_as", dtype_column)
                if source_column not in measured:
                    src = DTYPE_COLUMNS[source_column]
                    profiler.clear()
                    compute_kernel_config = ttnn.init_device_compute_kernel_config(
                        device.arch(),
                        math_fidelity=src["fidelity"],
                        math_approx_mode=True,
                        fp32_dest_acc_en=src["fp32_acc"],
                        packer_l1_acc=True,
                    )
                    config, blocking_label = blocking_for(source_column, M, N, K, core_grid, src["fp32_acc"])
                    cores_used = num_cores if config is not None else full_grid.x * full_grid.y
                    grid_used = grid_str if config is not None else f"{full_grid.x}x{full_grid.y}"

                    def op_fn(a, b, config=config, compute_kernel_config=compute_kernel_config):
                        return ttnn.experimental.minimal_matmul(
                            a, b, config=config, compute_kernel_config=compute_kernel_config
                        )

                    in0, in1, in0_t, in1_t = make_inputs(device, M, N, K, src["dtype"])
                    output_t = None
                    try:
                        inference_time_avg, trisc1_dur, device_freq_hz, output_t = run_measurement(
                            device, in0_t, in1_t, op_fn, use_trace, num_warmup_iterations, num_measurement_iterations
                        )
                        metrics = compute_metrics(
                            M, N, K, src["passes"], cores_used, inference_time_avg, trisc1_dur, device_freq_hz
                        )
                        row_pcc = check_rows_pcc(in0, in1, output_t)
                        measured[source_column] = (metrics, row_pcc, blocking_label, grid_used)
                    except RuntimeError as e:
                        if not is_skippable_benchmark_runtime_error(e):
                            raise
                        reason = str(e).splitlines()[0]
                        for line in str(e).splitlines():
                            if "TT_THROW" not in line and line.strip():
                                reason = line.strip()
                                break
                        logger.warning(
                            f"{source_column} {shape_id((M, N, K))} blocking {blocking_label} not runnable: {reason}"
                        )
                        measured[source_column] = (None, None, blocking_label, f"not runnable: {reason}")
                    finally:
                        for t in (output_t, in0_t, in1_t):
                            if t is not None:
                                ttnn.deallocate(t)

                metrics, row_pcc, blocking_label, grid_used = measured[source_column]
                if metrics is None:
                    unrunnable.append((dtype_column, (M, N, K), blocking_label))
                    row.update(
                        blocking=blocking_label,
                        theoretical_tflops=fmt(
                            theoretical_tflops(num_cores, get_device_frequency_hz(), spec["passes"]), 2
                        ),
                        measured_tflops="N/A",
                        utilization_pct="N/A",
                        note=(row["note"] + "; " if row["note"] else "") + grid_used,
                    )
                    row["grid"] = grid_str
                    write_row(row)
                    continue
                threshold = PCC_THRESHOLD[dtype_column]
                note = row["note"]
                if row_pcc < threshold:
                    pcc_failures.append((dtype_column, (M, N, K), row_pcc))
                    note = (note + "; " if note else "") + f"PCC {row_pcc:.4f} below {threshold}"
                row.update(
                    grid=grid_used,
                    blocking=blocking_label,
                    time_avg_ms=fmt(metrics["time_avg_ms"], 4),
                    theoretical_tflops=fmt(metrics["theoretical_tflops"], 2),
                    measured_tflops=fmt(metrics["measured_tflops"], 2),
                    utilization_pct=fmt(metrics["utilization_pct"], 2),
                    pcc=fmt(row_pcc, 5),
                    note=note,
                )
                for c in DEVICE_COLUMNS:
                    if c in metrics:
                        row[c] = fmt(metrics[c], 4 if c == "device_time_ms" else 2)
                write_row(row)

                device_str = (
                    f", device util {metrics['device_utilization_pct']:.1f}%"
                    if "device_utilization_pct" in metrics
                    else ""
                )
                logger.info(
                    f"{dtype_column} {shape_id((M, N, K))} trace={use_trace} blocking {blocking_label} grid {grid_used}: "
                    f"{metrics['time_avg_ms']:.3f} ms, {metrics['measured_tflops']:.1f} / {metrics['theoretical_tflops']:.0f} TFLOP/s "
                    f"= {metrics['utilization_pct']:.1f}%{device_str}, pcc {row_pcc:.4f}"
                )

                best = peak_by_column.get(dtype_column)
                if best is None or metrics["measured_tflops"] > best["measured_tflops"]:
                    peak_by_column[dtype_column] = dict(
                        metrics, shape=(M, N, K), blocking=blocking_label, grid=grid_used
                    )

        # PEAK summary rows: best sustained rate per dtype column over the sheet shapes.
        for dtype_column in SHEET_COLUMN_ORDER:
            spec = DTYPE_COLUMNS[dtype_column]
            best = peak_by_column.get(dtype_column)
            row = dict(
                M="PEAK",
                N="",
                K="",
                dtype_column=dtype_column,
                ttnn_dtype=str(spec["dtype"]) if spec["dtype"] is not None else "N/A",
                fidelity=str(spec["fidelity"]),
                passes=spec["passes"],
                accumulation="fp32" if spec["fp32_acc"] else "bf16",
                use_trace=use_trace,
            )
            if best is None:
                row.update(measured_tflops="N/A", utilization_pct="N/A", note=spec["note"])
            else:
                row.update(
                    flops=best["flops"],
                    grid=best["grid"],
                    blocking=best["blocking"],
                    theoretical_tflops=fmt(best["theoretical_tflops"], 2),
                    measured_tflops=fmt(best["measured_tflops"], 2),
                    utilization_pct=fmt(best["utilization_pct"], 2),
                    note=f"best over sheet shapes, at {shape_id(best['shape'])}",
                )
                for c in DEVICE_COLUMNS:
                    if c in best:
                        row[c] = fmt(best[c], 4 if c == "device_time_ms" else 2)
                logger.info(
                    f"PEAK {dtype_column} trace={use_trace}: {best['measured_tflops']:.1f} TFLOP/s "
                    f"({best['utilization_pct']:.1f}%) at {shape_id(best['shape'])}"
                )
            write_row(row)

    logger.info(f"Sheet CSV written to {csv_path}")
    assert not unrunnable, f"Cells with a blocking that is not runnable (written as N/A): {unrunnable}"
    assert not pcc_failures, f"PCC below threshold for: {pcc_failures}"
