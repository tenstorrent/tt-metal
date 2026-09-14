# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Phase 1 of the partner GEMM benchmark: offline block-size sweep for ttnn.experimental.minimal_matmul.
#
# For every sheet shape (M, N, K) and every supported sheet data type, this test runs the op with
# config=None ("oob", the op's own default blocking) and then with every candidate
# ttnn.MinimalMatmulConfig (M/K/N block sizes in tiles x sub-block shapes). Every run is appended to
#   $TT_METAL_HOME/generated/minimal_matmul_block_sweep.csv
# and at session end the best (highest TFLOP/s) config per (dtype column, M, N, K) is written to
#   $TT_METAL_HOME/generated/minimal_matmul_best_blocking.json
# and logged as a Python literal ready to paste into BEST_BLOCKING in test_minimal_matmul_benchmark.py
# (phase 2, the partner-facing script).
#
# Sheet data-type column -> ttnn mapping (fidelity = number of passes through the 5x7-bit multiplier):
#   BF16 -> ttnn.bfloat16,  HiFi2 (2 passes), bf16 dest accumulation
#   TF32 -> ttnn.float32,   HiFi4 (4 passes), fp32 dest accumulation   (same run as FP32)
#   FP32 -> ttnn.float32,   HiFi4 (4 passes), fp32 dest accumulation   (FPU consumes fp32 inputs as TF32)
#   FP8  -> ttnn.bfloat8_b, LoFi  (1 pass),   bf16 dest accumulation   (block-fp8)
#   FP4  -> ttnn.bfloat4_b, LoFi  (1 pass),   bf16 dest accumulation
#   FP16 -> not swept: tt-metal has no FLOAT16 tensor dtype (tt_metal/api/tt-metalium/tensor/tensor_types.hpp);
#           the hardware format exists but is unshipped for compute (tt_metal/jit_build/data_format.cpp).
#   FP6  -> struck from the sheet (not natively supported).  HiFi3 is not used.
#
# Metrics per row:
#   time_avg_ms        host-timed average per op over num_measurement_iterations (eager, includes dispatch)
#   tflops             2*M*N*K / time_avg          (TFLOP/s; "flops" in the CSV is the op count 2*M*N*K)
#   theoretical_tflops num_cores * freq_hz * (8*16*16) * 2 / passes / 1e12
#   utilization_pct    tflops / theoretical_tflops * 100
#   device_* columns   only with a profiler build (TT_METAL_DEVICE_PROFILER set): the same rates from
#                      the average TRISC1 (math) kernel duration, i.e. compute-only, no host dispatch.
#
# Running (manual-only, gated by an env var so it never runs in CI by accident):
#   TTNN_RUN_GEMM_FLOPS_BENCHMARK=1 pytest tests/ttnn/unit_tests/benchmarks/test_minimal_matmul_block_sweep.py
#   --grid-size 12x10                     restrict the op to an x-by-y core grid (default: full compute grid)
#   -k "8192x8192x8192 and BF16"          narrow to one shape / data type
#   TTNN_MINIMAL_MATMUL_BLOCK_SIZES=4,8   narrow the candidate block sizes (default 1,2,4,8,16)
#   TTNN_MINIMAL_MATMUL_SWEEP_RESET=1    start a fresh sweep CSV (default: append when the header matches,
#                                         so the sweep can be split across several pytest invocations)
# Each test item is one (shape, dtype) sweep and can take many minutes; the pytest.ini 300 s timeout is
# disabled per test below. Total for the full sheet is on the order of 1-2 h on one Blackhole chip.

import csv
import json
import math
import os
from itertools import product
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
BLOCK_SIZES_ENV = "TTNN_MINIMAL_MATMUL_BLOCK_SIZES"
SWEEP_RESET_ENV = "TTNN_MINIMAL_MATMUL_SWEEP_RESET"

SWEEP_CSV_NAME = "minimal_matmul_block_sweep.csv"
BEST_BLOCKING_JSON_NAME = "minimal_matmul_best_blocking.json"

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

# Sheet data-type column -> how it is run. "passes" is the fidelity pass count used in the peak formula.
DTYPE_COLUMNS = {
    "BF16": dict(dtype=ttnn.bfloat16, fidelity=ttnn.MathFidelity.HiFi2, passes=2, fp32_acc=False),
    "FP32": dict(dtype=ttnn.float32, fidelity=ttnn.MathFidelity.HiFi4, passes=4, fp32_acc=True),
    "FP8": dict(dtype=ttnn.bfloat8_b, fidelity=ttnn.MathFidelity.LoFi, passes=1, fp32_acc=False),
    "FP4": dict(dtype=ttnn.bfloat4_b, fidelity=ttnn.MathFidelity.LoFi, passes=1, fp32_acc=False),
}
SWEPT_COLUMNS = ["BF16", "FP32", "FP8", "FP4"]  # TF32 shares the FP32 run; FP16 has no tensor dtype

TORCH_INPUT_DTYPE = {ttnn.float32: torch.float32}  # everything else is generated as torch.bfloat16

# Candidate block sizes (tiles). Sub-block candidates are derived per (M_block, N_block): every divisor
# pair whose area fits the dest register count (8 tiles with bf16 accumulation, 4 with fp32, see
# get_dest_reg_count in ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.cpp), keeping
# only the pairs of maximal area, e.g. 1x8 / 2x4 / 4x2 / 8x1 for an 8x8 block, 1x4 for a 1x4 block.
DEFAULT_BLOCK_SIZES = [1, 2, 4, 8, 16]
DEST_REGS = {False: 8, True: 4}

FLOP_PER_CORE_PER_CYCLE = 8 * 16 * 16 * 2  # 8x16x16 MAC array, multiply + add


def get_block_sizes():
    raw = os.getenv(BLOCK_SIZES_ENV)
    if not raw:
        return DEFAULT_BLOCK_SIZES
    return sorted({int(x) for x in raw.split(",") if x.strip()})


def default_block_config(M, N, fp32_acc):
    """Mirror determine_default_block_sizes in minimal_matmul_program_descriptor.cpp (config=None path)."""
    if fp32_acc:
        return (8, 8, 8, 2, 2)
    return (8, 8, 8, 2, 4) if N >= M else (8, 8, 8, 4, 2)


def iter_subblocks(Mb, Nb, dest_regs):
    """Divisor pairs (sh, sw) of (Mb, Nb) with sh*sw <= dest_regs, restricted to the maximal area."""
    pairs = [
        (sh, sw)
        for sh in range(1, Mb + 1)
        if Mb % sh == 0
        for sw in range(1, Nb + 1)
        if Nb % sw == 0 and sh * sw <= dest_regs
    ]
    best_area = max(sh * sw for sh, sw in pairs)
    return [(sh, sw) for sh, sw in pairs if sh * sw == best_area]


def iter_block_configs(M, N, grid_x, grid_y, fp32_acc):
    """Yield (M_block, K_block, N_block, subblock_h, subblock_w) candidates that pass op validation and
    are not larger than the per-core tile extent along M (split over grid y) and N (split over grid x)."""
    m_tiles_per_core = max(1, math.ceil(math.ceil(M / 32) / grid_y))
    n_tiles_per_core = max(1, math.ceil(math.ceil(N / 32) / grid_x))
    block_sizes = get_block_sizes()
    for Mb, Kb, Nb in product(block_sizes, repeat=3):
        if Mb > m_tiles_per_core or Nb > n_tiles_per_core:
            continue
        for sh, sw in iter_subblocks(Mb, Nb, DEST_REGS[fp32_acc]):
            yield (Mb, Kb, Nb, sh, sw)


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


def is_skippable_benchmark_runtime_error(error):
    """Return whether a config failed because it is not runnable for this shape/device (L1, validation)."""
    message = str(error).lower()
    return any(substring in message for substring in SKIPPABLE_RUNTIME_ERROR_SUBSTRINGS)


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
        "tflops": tflops,
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


# ---------------------------------------------------------------------------
# CSV / best-map plumbing
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
    "fp32_acc",
    "grid",
    "mode",
    "M_block",
    "K_block",
    "N_block",
    "subblock_h",
    "subblock_w",
    "time_avg_ms",
    "tflops",
    "theoretical_tflops",
    "utilization_pct",
    "pcc",
]
DEVICE_COLUMNS = ["device_time_ms", "device_tflops", "device_utilization_pct"]


def csv_columns():
    return BASE_COLUMNS + (DEVICE_COLUMNS if get_profiler_build_enabled() else [])


def summarize_best_blocking(sweep_csv_path, best_json_path):
    """Pick the highest-TFLOP/s row per (dtype_column, M, N, K) and emit JSON + a paste-able literal."""
    best = {}
    with open(sweep_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["dtype_column"], int(row["M"]), int(row["N"]), int(row["K"]))
            tflops = float(row["tflops"])
            if key not in best or tflops > best[key]["tflops"]:
                best[key] = {
                    "tflops": tflops,
                    "mode": row["mode"],
                    "blocking": tuple(
                        int(row[c]) for c in ("M_block", "K_block", "N_block", "subblock_h", "subblock_w")
                    ),
                }
    if not best:
        return
    json_payload = [
        dict(
            dtype_column=k[0], M=k[1], N=k[2], K=k[3], tflops=v["tflops"], mode=v["mode"], blocking=list(v["blocking"])
        )
        for k, v in sorted(best.items())
    ]
    with open(best_json_path, "w") as f:
        json.dump(json_payload, f, indent=2)

    lines = ["BEST_BLOCKING = {"]
    for k, v in sorted(best.items()):
        lines.append(f"    {k!r}: {v['blocking']!r},  # {v['tflops']:.1f} TFLOP/s ({v['mode']})")
    lines.append("}")
    logger.info(
        "Best blocking per (dtype_column, M, N, K); paste into test_minimal_matmul_benchmark.py:\n" + "\n".join(lines)
    )
    logger.info(f"Best blocking JSON written to {best_json_path}")


@pytest.fixture(scope="session")
def sweep_report():
    """Append rows to the sweep CSV. An existing CSV with the same header is appended to, so the sweep can be
    split across pytest sessions; set TTNN_MINIMAL_MATMUL_SWEEP_RESET=1 to start a fresh CSV."""
    artifacts_dir = Path(os.environ["TT_METAL_HOME"]) / "generated"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    sweep_csv_path = artifacts_dir / SWEEP_CSV_NAME
    reuse = False
    if sweep_csv_path.exists() and os.getenv(SWEEP_RESET_ENV) != "1":
        with open(sweep_csv_path, newline="") as f:
            reuse = next(csv.reader(f), None) == csv_columns()
    if not reuse:
        with open(sweep_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_columns())

    def append(row):
        with open(sweep_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(c, "") for c in csv_columns()])

    yield append
    summarize_best_blocking(sweep_csv_path, artifacts_dir / BEST_BLOCKING_JSON_NAME)


# ---------------------------------------------------------------------------
# Sweep test: one item per (shape, dtype column)
# ---------------------------------------------------------------------------


def shape_id(shape):
    return "x".join(str(d) for d in shape)


@pytest.mark.skipif(
    os.getenv(GEMM_FLOPS_BENCHMARK_ENV) != "1",
    reason=f"Benchmark is manual-only; set {GEMM_FLOPS_BENCHMARK_ENV}=1 to run",
)
@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("M, N, K", SHEET_SHAPES, ids=[shape_id(s) for s in SHEET_SHAPES])
@pytest.mark.parametrize("dtype_column", SWEPT_COLUMNS)
@pytest.mark.parametrize("num_warmup_iterations", [3])
@pytest.mark.parametrize("num_measurement_iterations", [10])
def test_minimal_matmul_block_sweep(
    device,
    grid_size,
    sweep_report,
    M,
    N,
    K,
    dtype_column,
    num_warmup_iterations,
    num_measurement_iterations,
):
    spec = DTYPE_COLUMNS[dtype_column]
    dtype, fidelity, passes, fp32_acc = spec["dtype"], spec["fidelity"], spec["passes"], spec["fp32_acc"]

    grid_x, grid_y = resolve_grid(device, grid_size)
    core_grid = ttnn.CoreCoord(grid_x, grid_y)
    full_grid = device.compute_with_storage_grid_size()
    num_cores_user = grid_x * grid_y
    num_cores_full = full_grid.x * full_grid.y

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=True,
    )

    in0, in1, in0_t, in1_t = make_inputs(device, M, N, K, dtype)

    # oob (config=None, full device grid) first, then every candidate on the requested grid.
    candidates = [("oob", None, default_block_config(M, N, fp32_acc), num_cores_full, f"{full_grid.x}x{full_grid.y}")]
    for blocking in iter_block_configs(M, N, grid_x, grid_y, fp32_acc):
        candidates.append(("sweep", blocking, blocking, num_cores_user, f"{grid_x}x{grid_y}"))
    logger.info(f"{shape_id((M, N, K))} {dtype_column}: {len(candidates) - 1} sweep candidates + oob")

    best = None
    try:
        for mode, blocking, recorded_blocking, num_cores, grid_str in candidates:
            profiler.clear()
            config = None
            if blocking is not None:
                Mb, Kb, Nb, sh, sw = blocking
                config = ttnn.MinimalMatmulConfig(
                    M_block_size=Mb,
                    K_block_size=Kb,
                    N_block_size=Nb,
                    subblock_h=sh,
                    subblock_w=sw,
                    compute_with_storage_grid_size=core_grid,
                )

            def op_fn(a, b, config=config):
                return ttnn.experimental.minimal_matmul(
                    a, b, config=config, compute_kernel_config=compute_kernel_config
                )

            output_t = None
            try:
                inference_time_avg, trisc1_dur, device_freq_hz, output_t = run_measurement(
                    device, in0_t, in1_t, op_fn, False, num_warmup_iterations, num_measurement_iterations
                )
                metrics = compute_metrics(M, N, K, passes, num_cores, inference_time_avg, trisc1_dur, device_freq_hz)
                row_pcc = check_rows_pcc(in0, in1, output_t) if mode == "oob" else ""
            except RuntimeError as e:
                if not is_skippable_benchmark_runtime_error(e):
                    raise
                logger.warning(
                    f"Skipping {dtype_column} {shape_id((M, N, K))} {mode} {recorded_blocking}: {str(e).splitlines()[0]}"
                )
                continue
            finally:
                if output_t is not None:
                    ttnn.deallocate(output_t)

            Mb, Kb, Nb, sh, sw = recorded_blocking
            row = dict(
                M=M,
                N=N,
                K=K,
                flops=metrics["flops"],
                dtype_column=dtype_column,
                ttnn_dtype=str(dtype),
                fidelity=str(fidelity),
                passes=passes,
                fp32_acc=fp32_acc,
                grid=grid_str,
                mode=mode,
                M_block=Mb,
                K_block=Kb,
                N_block=Nb,
                subblock_h=sh,
                subblock_w=sw,
                time_avg_ms=f"{metrics['time_avg_ms']:.4f}",
                tflops=f"{metrics['tflops']:.2f}",
                theoretical_tflops=f"{metrics['theoretical_tflops']:.2f}",
                utilization_pct=f"{metrics['utilization_pct']:.2f}",
                pcc=f"{row_pcc:.5f}" if row_pcc != "" else "",
            )
            for c in DEVICE_COLUMNS:
                if c in metrics:
                    row[c] = f"{metrics[c]:.2f}" if c != "device_time_ms" else f"{metrics[c]:.4f}"
            sweep_report(row)

            device_str = (
                f", device util {metrics['device_utilization_pct']:.1f}%" if "device_utilization_pct" in metrics else ""
            )
            logger.info(
                f"[{mode}] {dtype_column} {shape_id((M, N, K))} blocks {Mb}/{Kb}/{Nb} sub {sh}x{sw} grid {grid_str}: "
                f"{metrics['time_avg_ms']:.3f} ms, {metrics['tflops']:.1f} TFLOP/s, "
                f"{metrics['utilization_pct']:.1f}% of {metrics['theoretical_tflops']:.0f}{device_str}"
                + (f", pcc {row_pcc:.4f}" if row_pcc != "" else "")
            )
            if best is None or metrics["tflops"] > best[0]:
                best = (metrics["tflops"], mode, recorded_blocking)
    finally:
        ttnn.deallocate(in0_t)
        ttnn.deallocate(in1_t)

    assert best is not None, f"No runnable config for {dtype_column} {shape_id((M, N, K))}"
    logger.info(f"BEST {dtype_column} {shape_id((M, N, K))}: {best[2]} ({best[1]}) at {best[0]:.1f} TFLOP/s")
