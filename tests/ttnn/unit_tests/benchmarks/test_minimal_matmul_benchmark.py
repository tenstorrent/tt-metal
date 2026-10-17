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
# Shared definitions live in minimal_matmul_gemm_common.py.
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

import pytest
import ttnn
from loguru import logger

from tests.ttnn.unit_tests.benchmarks.minimal_matmul_gemm_common import (
    DEVICE_METRIC_COLUMNS,
    DEVICE_PARAMS,
    DTYPE_COLUMNS,
    GEMM_FLOPS_BENCHMARK_ENV,
    SHEET_COLUMN_ORDER,
    SHEET_SHAPES,
    check_rows_pcc,
    compute_metrics,
    device_metric_fields,
    fmt,
    get_device_frequency_hz,
    get_profiler_build_enabled,
    is_skippable_benchmark_runtime_error,
    make_compute_kernel_config,
    make_inputs,
    make_matmul_config,
    minimal_matmul_fn,
    resolve_grid,
    run_measurement,
    runtime_error_reason,
    shape_id,
    theoretical_tflops,
)

SHEET_CSV_NAME = "minimal_matmul_gemm_sheet.csv"
SHAPES_ENV = "TTNN_MINIMAL_MATMUL_SHAPES"

# Minimum acceptable PCC of the 32-row check per column (bfloat4_b carries a 4-bit mantissa).
PCC_THRESHOLD = {"BF16": 0.99, "TF32": 0.99, "FP32": 0.99, "FP8": 0.99, "FP4": 0.90}

# (dtype_column, M, N, K) -> (M_block, K_block, N_block, subblock_h, subblock_w) in tiles, or None when the
# sweep winner was the op's own default blocking (config=None: 8/8/8, sub-block 2x4 or 4x2 with bf16
# accumulation, 2x2 with fp32 accumulation). Paste the BEST_BLOCKING literal logged at the end of a phase-1
# sweep here. Missing keys use the op default for bf16 accumulation and FALLBACK_BLOCKING_FP32_ACC for fp32.
BEST_BLOCKING = {
    ("BF16", 32, 2048, 2048): (1, 4, 8, 1, 8),  # 5.8 TFLOP/s (grid 12x10)
    ("BF16", 32, 4096, 4096): (1, 16, 4, 1, 4),  # 7.9 TFLOP/s (grid 12x10)
    ("BF16", 8192, 4096, 1536): (16, 4, 8, 1, 8),  # 159.7 TFLOP/s (grid 12x10)
    ("BF16", 8192, 4096, 12288): (8, 8, 8, 8, 1),  # 261.6 TFLOP/s (grid 12x10)
    ("BF16", 8192, 5120, 25600): (8, 4, 8, 8, 1),  # 266.3 TFLOP/s (grid 12x10)
    ("BF16", 8192, 8192, 8192): (4, 4, 8, 1, 8),  # 247.5 TFLOP/s (grid 12x10)
    ("BF16", 8192, 16384, 16384): (8, 4, 16, 8, 1),  # 242.5 TFLOP/s (grid 12x10)
    ("BF16", 8192, 24576, 4096): (8, 4, 16, 2, 4),  # 243.5 TFLOP/s (grid 12x10)
    ("BF16", 8192, 32768, 16384): (8, 4, 8, 8, 1),  # 244.8 TFLOP/s (grid 12x10)
    ("BF16", 16384, 16384, 16384): (8, 4, 16, 8, 1),  # 256.0 TFLOP/s (grid 12x10)
    ("FP32", 32, 2048, 2048): (1, 4, 8, 1, 4),  # 3.5 TFLOP/s (grid 12x10)
    ("FP32", 32, 4096, 4096): (1, 4, 16, 1, 4),  # 4.6 TFLOP/s (grid 12x10)
    ("FP32", 8192, 4096, 1536): (8, 4, 8, 4, 1),  # 78.0 TFLOP/s (grid 12x10)
    ("FP32", 8192, 4096, 12288): (8, 8, 4, 1, 4),  # 113.3 TFLOP/s (grid 12x10)
    ("FP32", 8192, 5120, 25600): (8, 4, 8, 1, 4),  # 120.5 TFLOP/s (grid 12x10)
    ("FP32", 8192, 8192, 8192): (4, 4, 8, 4, 1),  # 116.6 TFLOP/s (grid 12x10)
    ("FP32", 8192, 16384, 16384): (4, 4, 16, 4, 1),  # 114.8 TFLOP/s (grid 12x10)
    ("FP32", 8192, 24576, 4096): (4, 4, 16, 4, 1),  # 112.0 TFLOP/s (grid 12x10)
    ("FP32", 8192, 32768, 16384): (4, 8, 8, 4, 1),  # 114.7 TFLOP/s (grid 12x10)
    ("FP32", 16384, 16384, 16384): (4, 4, 16, 4, 1),  # 115.0 TFLOP/s (grid 12x10)
    ("FP8", 32, 2048, 2048): (1, 8, 8, 1, 8),  # 7.7 TFLOP/s (grid 12x10)
    ("FP8", 32, 4096, 4096): (1, 16, 8, 1, 8),  # 11.5 TFLOP/s (grid 12x10)
    ("FP8", 8192, 4096, 1536): (8, 2, 8, 8, 1),  # 311.1 TFLOP/s (grid 12x10)
    ("FP8", 8192, 4096, 12288): (8, 8, 8, 1, 8),  # 517.4 TFLOP/s (grid 12x10)
    ("FP8", 8192, 5120, 25600): (8, 16, 16, 1, 8),  # 531.6 TFLOP/s (grid 12x10)
    ("FP8", 8192, 8192, 8192): (16, 8, 8, 1, 8),  # 480.7 TFLOP/s (grid 12x10)
    ("FP8", 8192, 16384, 16384): (16, 8, 8, 1, 8),  # 497.1 TFLOP/s (grid 12x10)
    ("FP8", 8192, 24576, 4096): (8, 8, 16, 8, 1),  # 477.0 TFLOP/s (grid 12x10)
    ("FP8", 8192, 32768, 16384): (16, 8, 8, 1, 8),  # 504.6 TFLOP/s (grid 12x10)
    ("FP8", 16384, 16384, 16384): (8, 16, 16, 8, 1),  # 516.2 TFLOP/s (grid 12x10)
    ("FP4", 32, 2048, 2048): (1, 16, 8, 1, 8),  # 8.9 TFLOP/s (grid 12x10)
    ("FP4", 32, 4096, 4096): (1, 8, 16, 1, 8),  # 14.0 TFLOP/s (grid 12x10)
    ("FP4", 8192, 4096, 1536): (8, 8, 8, 8, 1),  # 440.6 TFLOP/s (grid 12x10)
    ("FP4", 8192, 4096, 12288): (16, 8, 8, 8, 1),  # 544.2 TFLOP/s (grid 12x10)
    ("FP4", 8192, 5120, 25600): (16, 16, 16, 8, 1),  # 563.0 TFLOP/s (grid 12x10)
    ("FP4", 8192, 8192, 8192): (16, 8, 8, 1, 8),  # 544.6 TFLOP/s (grid 12x10)
    ("FP4", 8192, 16384, 16384): (16, 16, 8, 1, 8),  # 550.5 TFLOP/s (grid 12x10)
    ("FP4", 8192, 24576, 4096): (16, 8, 8, 1, 8),  # 556.1 TFLOP/s (grid 12x10)
    ("FP4", 8192, 32768, 16384): (16, 8, 16, 1, 8),  # 547.7 TFLOP/s (grid 12x10)
    ("FP4", 16384, 16384, 16384): (16, 16, 16, 8, 1),  # 565.5 TFLOP/s (grid 12x10)
}

# Explicit fallback for fp32 accumulation when BEST_BLOCKING has no entry (bf16 accumulation uses config=None).
FALLBACK_BLOCKING_FP32_ACC = (4, 8, 4, 2, 2)


def blocking_for(dtype_column, M, N, K, core_grid, fp32_acc):
    """Return (ttnn.MinimalMatmulConfig or None, label) for a sheet cell."""
    key = (dtype_column, M, N, K)
    if key in BEST_BLOCKING and BEST_BLOCKING[key] is None:
        return None, "default (sweep winner)"
    blocking = BEST_BLOCKING.get(key)
    label = "best"
    if blocking is None:
        if not fp32_acc:
            return None, "default"
        blocking = FALLBACK_BLOCKING_FP32_ACC
        label = "fallback-fp32"
    Mb, Kb, Nb, sh, sw = blocking
    return make_matmul_config(blocking, core_grid), f"{label} {Mb}/{Kb}/{Nb} sub {sh}x{sw}"


@pytest.fixture(scope="session")
def sheet_report():
    """Row appender for the sheet CSV. The file is truncated once per pytest session, so an eager-only or
    trace-only invocation never appends to rows from an earlier run."""
    artifacts_dir = Path(os.environ["TT_METAL_HOME"]) / "generated"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    csv_path = artifacts_dir / SHEET_CSV_NAME
    columns = csv_columns()
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(columns)

    def append(row):
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(c, "") for c in columns])

    append.path = csv_path
    yield append


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


def csv_columns():
    return BASE_COLUMNS + (DEVICE_METRIC_COLUMNS if get_profiler_build_enabled() else [])


# ---------------------------------------------------------------------------
# Benchmark test: one item per dispatch mode, all sheet cells inside
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.getenv(GEMM_FLOPS_BENCHMARK_ENV) != "1",
    reason=f"Benchmark is manual-only; set {GEMM_FLOPS_BENCHMARK_ENV}=1 to run",
)
@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("use_trace", [False, True], ids=["eager", "trace"])
@pytest.mark.parametrize("num_warmup_iterations", [5])
@pytest.mark.parametrize("num_measurement_iterations", [20])
def test_minimal_matmul_gemm_sheet(
    device,
    grid_size,
    sheet_report,
    use_trace,
    num_warmup_iterations,
    num_measurement_iterations,
):
    csv_path = sheet_report.path

    grid_x, grid_y = resolve_grid(device, grid_size)
    core_grid = ttnn.CoreCoord(grid_x, grid_y)
    full_grid = device.compute_with_storage_grid_size()
    grid_str = f"{grid_x}x{grid_y}"
    num_cores = grid_x * grid_y

    peak_by_column = {}
    pcc_failures = []
    unrunnable = []
    write_row = sheet_report

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
                    theoretical_tflops=fmt(theoretical_tflops(num_cores, get_device_frequency_hz(), spec["passes"]), 2),
                    measured_tflops="N/A",
                    utilization_pct="N/A",
                )
                write_row(row)
                logger.info(f"{dtype_column} {shape_id((M, N, K))}: N/A ({spec['note']})")
                continue

            source_column = spec.get("measure_as", dtype_column)
            if source_column not in measured:
                src = DTYPE_COLUMNS[source_column]
                compute_kernel_config = make_compute_kernel_config(device, src)
                config, blocking_label = blocking_for(source_column, M, N, K, core_grid, src["fp32_acc"])
                cores_used = num_cores if config is not None else full_grid.x * full_grid.y
                grid_used = grid_str if config is not None else f"{full_grid.x}x{full_grid.y}"

                op_fn = minimal_matmul_fn(config, compute_kernel_config)

                in0, in1, in0_t, in1_t = make_inputs(device, M, N, K, src["dtype"])
                output_t = None
                try:
                    inference_time_avg, trisc1_dur, device_freq_hz, output_t = run_measurement(
                        device,
                        in0_t,
                        in1_t,
                        op_fn,
                        use_trace,
                        num_warmup_iterations,
                        num_measurement_iterations,
                        get_profiler_build_enabled(),
                    )
                    metrics = compute_metrics(
                        M, N, K, src["passes"], cores_used, inference_time_avg, trisc1_dur, device_freq_hz
                    )
                    row_pcc = check_rows_pcc(in0, in1, output_t)
                    measured[source_column] = (metrics, row_pcc, blocking_label, grid_used)
                except RuntimeError as e:
                    if not is_skippable_benchmark_runtime_error(e):
                        raise
                    reason = runtime_error_reason(e)
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
                    theoretical_tflops=fmt(theoretical_tflops(num_cores, get_device_frequency_hz(), spec["passes"]), 2),
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
            row.update(device_metric_fields(metrics))
            write_row(row)

            device_str = (
                f", device util {metrics['device_utilization_pct']:.1f}%" if "device_utilization_pct" in metrics else ""
            )
            logger.info(
                f"{dtype_column} {shape_id((M, N, K))} trace={use_trace} blocking {blocking_label} grid {grid_used}: "
                f"{metrics['time_avg_ms']:.3f} ms, {metrics['measured_tflops']:.1f} / {metrics['theoretical_tflops']:.0f} TFLOP/s "
                f"= {metrics['utilization_pct']:.1f}%{device_str}, pcc {row_pcc:.4f}"
            )

            best = peak_by_column.get(dtype_column)
            if best is None or metrics["measured_tflops"] > best["measured_tflops"]:
                peak_by_column[dtype_column] = dict(metrics, shape=(M, N, K), blocking=blocking_label, grid=grid_used)

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
            row.update(device_metric_fields(best))
            logger.info(
                f"PEAK {dtype_column} trace={use_trace}: {best['measured_tflops']:.1f} TFLOP/s "
                f"({best['utilization_pct']:.1f}%) at {shape_id(best['shape'])}"
            )
        write_row(row)

    logger.info(f"Sheet CSV written to {csv_path}")
    assert not unrunnable, f"Cells with a blocking that is not runnable (written as N/A): {unrunnable}"
    assert not pcc_failures, f"PCC below threshold for: {pcc_failures}"
