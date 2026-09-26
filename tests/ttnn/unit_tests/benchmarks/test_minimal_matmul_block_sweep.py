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
# (phase 2, the partner-facing script). Shared definitions live in minimal_matmul_gemm_common.py.
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
# Timing: after a compile run and warmup, num_measurement_iterations ops are captured in a trace and the
# trace is executed TRACE_EXECUTIONS times; the fastest host-timed execution divided by the iteration count
# is the per-op time. Trace strips host dispatch, so the sweep ranks blockings by device time, which is what the
# blocking choice should be based on, and the number is independent of how many chips run concurrently.
#
# Parallelism across chips (TTNN_MINIMAL_MATMUL_NUM_CHIPS=N, default 1): the system mesh is opened once,
# carved into 1x1 submeshes, and the candidate configs of each (shape, dtype) are dealt round-robin to N
# chips. One Python thread per chip creates its own inputs on its submesh and runs its share; the op and
# synchronize bindings release the GIL, so kernel compiles and device runs overlap across chips. N is
# capped at 16 per the dI/dt guidance in the meeting notes. With N=1 the ordinary conftest `device`
# fixture is used and no mesh is opened.
#
# Peak search (test_minimal_matmul_peak_search): the sheet shapes stream both operands from DRAM and top out
# where the DRAM pipeline does, so a second test measures the compute-bound point of the op instead. It sizes
# M and N so that each core's whole output is a single M_block x N_block (PEAK_BLOCKS per axis, on the grid in
# use) and sweeps K (PEAK_K), K_block and the sub-block. Every input tile then crosses DRAM exactly once and
# the bandwidth the FPU needs at peak falls with the block size (about 480 GB/s for 8x8 tiles at the BF16
# HiFi2 peak on 12x10, about 240 GB/s for 16x16). Rows go to generated/minimal_matmul_peak_gemm.csv and the
# best rate per dtype column is logged at session end.
#
# Metrics per row:
#   time_avg_ms        per-op time from the traced execution (see Timing)
#   tflops             2*M*N*K / time_avg          (TFLOP/s; "flops" in the CSV is the op count 2*M*N*K)
#   theoretical_tflops num_cores * freq_hz * (8*16*16) * 2 / passes / 1e12
#   utilization_pct    tflops / theoretical_tflops * 100
#   device_* columns   only with a profiler build, TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1
#                      and N=1: the same rates from the average TRISC1 (math) kernel duration, with the peak
#                      taken at the profiler-reported clock. The mid-run dump flag is required, otherwise the
#                      device log is only written at device close.
#
# Running (manual-only, gated by an env var so it never runs in CI by accident):
#   TTNN_RUN_GEMM_FLOPS_BENCHMARK=1 pytest tests/ttnn/unit_tests/benchmarks/test_minimal_matmul_block_sweep.py
#   TTNN_MINIMAL_MATMUL_NUM_CHIPS=8       spread each sweep over 8 chips (1x1 submeshes), default 1
#   --grid-size 12x10                     restrict the op to an x-by-y core grid (default: full compute grid)
#   -k "8192x8192x8192 and BF16"          narrow to one shape / data type
#   TTNN_MINIMAL_MATMUL_BLOCK_SIZES=4,8   narrow the candidate block sizes (default 1,2,4,8,16)
#   TTNN_MINIMAL_MATMUL_SWEEP_RESET=1     start a fresh sweep CSV (default: append when the header matches,
#                                         so the sweep can be split across several pytest invocations)
# Each test item is one (shape, dtype) sweep of up to ~250 configs; the pytest.ini 300 s timeout is disabled
# per test below.

import csv
import json
import math
import os
import threading
from itertools import product
from pathlib import Path

import pytest
import ttnn
from loguru import logger

from tests.ttnn.unit_tests.benchmarks.minimal_matmul_gemm_common import (
    DEVICE_METRIC_COLUMNS,
    DEVICE_PARAMS,
    DTYPE_COLUMNS,
    GEMM_FLOPS_BENCHMARK_ENV,
    SHEET_SHAPES,
    SWEPT_COLUMNS,
    check_rows_pcc,
    compute_metrics,
    default_block_config,
    device_metric_fields,
    fmt,
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
)

BLOCK_SIZES_ENV = "TTNN_MINIMAL_MATMUL_BLOCK_SIZES"
SWEEP_RESET_ENV = "TTNN_MINIMAL_MATMUL_SWEEP_RESET"
NUM_CHIPS_ENV = "TTNN_MINIMAL_MATMUL_NUM_CHIPS"
MAX_PARALLEL_CHIPS = 16  # dI/dt guidance from the meeting notes: at most ~16 chips launching GEMMs at once
TRACE_EXECUTIONS = 3  # timed executions of the captured trace per config; the minimum is reported

SWEEP_CSV_NAME = "minimal_matmul_block_sweep.csv"
PEAK_CSV_NAME = "minimal_matmul_peak_gemm.csv"
BEST_BLOCKING_JSON_NAME = "minimal_matmul_best_blocking.json"

# Candidate block sizes (tiles). Sub-block candidates are derived per (M_block, N_block): every divisor
# pair whose area fits the dest register count (8 tiles with bf16 accumulation, 4 with fp32, see
# get_dest_reg_count in ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.cpp), keeping
# only the pairs of maximal area, e.g. 1x8 / 2x4 / 4x2 / 8x1 for an 8x8 block, 1x4 for a 1x4 block.
DEFAULT_BLOCK_SIZES = [1, 2, 4, 8, 16]

# Peak search: one output block per core (M = M_block*32*cores along M, N = N_block*32*cores along N), so
# every input tile crosses DRAM once and the compute-to-traffic ratio is set by the block size alone.
PEAK_BLOCKS = [8, 16]
PEAK_K = [4096, 8192, 16384]
DEST_REGS = {False: 8, True: 4}

FLOP_PER_CORE_PER_CYCLE = 8 * 16 * 16 * 2  # 8x16x16 MAC array, multiply + add


def get_block_sizes():
    raw = os.getenv(BLOCK_SIZES_ENV)
    if not raw:
        return DEFAULT_BLOCK_SIZES
    return sorted({int(x) for x in raw.split(",") if x.strip()})


def get_num_chips():
    n = int(os.getenv(NUM_CHIPS_ENV, "1"))
    assert 1 <= n <= MAX_PARALLEL_CHIPS, f"{NUM_CHIPS_ENV} must be in [1, {MAX_PARALLEL_CHIPS}], got {n}"
    return n


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


def block_cap(tiles_per_core, block_sizes):
    """Largest block size worth sweeping along one axis: the smallest candidate at or above the per-core tile
    extent. The op clamps the last block to a partial block, so every larger candidate measures the same
    partial block again."""
    for b in block_sizes:
        if b >= tiles_per_core:
            return b
    return block_sizes[-1]


def iter_block_configs(M, N, grid_x, grid_y, fp32_acc):
    """Yield (M_block, K_block, N_block, subblock_h, subblock_w) candidates that pass op validation.

    minimal_matmul parallelizes M over grid y and N over grid x, and transposes that mapping when M > N
    (minimal_matmul_program_descriptor.cpp, transpose_core_grid). Blocks beyond the per-core extent are
    capped at the first candidate that covers it (see block_cap)."""
    transpose_core_grid = M > N
    m_axis_cores = grid_x if transpose_core_grid else grid_y
    n_axis_cores = grid_y if transpose_core_grid else grid_x
    m_tiles_per_core = math.ceil(math.ceil(M / 32) / m_axis_cores)
    n_tiles_per_core = math.ceil(math.ceil(N / 32) / n_axis_cores)
    block_sizes = get_block_sizes()
    m_cap = block_cap(m_tiles_per_core, block_sizes)
    n_cap = block_cap(n_tiles_per_core, block_sizes)
    for Mb, Kb, Nb in product(block_sizes, repeat=3):
        if Mb > m_cap or Nb > n_cap:
            continue
        for sh, sw in iter_subblocks(Mb, Nb, DEST_REGS[fp32_acc]):
            yield (Mb, Kb, Nb, sh, sw)


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
    "chip",
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


def device_utilization_enabled():
    """Device (TRISC1) utilization needs a profiler build and a single chip; the per-mesh profiler log is
    not separable per submesh."""
    return get_profiler_build_enabled() and get_num_chips() == 1


def csv_columns():
    return BASE_COLUMNS + (DEVICE_METRIC_COLUMNS if device_utilization_enabled() else [])


def summarize_best_blocking(sweep_csv_path, best_json_path):
    """Pick the highest-TFLOP/s row per (dtype_column, M, N, K) and emit JSON + a paste-able literal.

    An `oob` winner (config=None, the op's own default blocking on the full grid) is emitted as None so that
    phase 2 replays it as config=None rather than as an explicit config on the requested grid. Rows from
    more than one grid in the same CSV are flagged: their winners are not comparable."""
    best = {}
    grids = set()
    with open(sweep_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            grids.add(row["grid"])
            key = (row["dtype_column"], int(row["M"]), int(row["N"]), int(row["K"]))
            tflops = float(row["tflops"])
            if key not in best or tflops > best[key]["tflops"]:
                blocking = tuple(int(row[c]) for c in ("M_block", "K_block", "N_block", "subblock_h", "subblock_w"))
                best[key] = {
                    "tflops": tflops,
                    "mode": row["mode"],
                    "grid": row["grid"],
                    "blocking": None if row["mode"] == "oob" else blocking,
                    "oob_default": blocking if row["mode"] == "oob" else None,
                }
    if not best:
        return
    if len(grids) > 1:
        logger.warning(
            f"Sweep CSV mixes rows from several grids {sorted(grids)}; winners are not comparable across grids"
        )
    json_payload = [
        dict(
            dtype_column=k[0],
            M=k[1],
            N=k[2],
            K=k[3],
            tflops=v["tflops"],
            mode=v["mode"],
            grid=v["grid"],
            blocking=None if v["blocking"] is None else list(v["blocking"]),
        )
        for k, v in sorted(best.items())
    ]
    with open(best_json_path, "w") as f:
        json.dump(json_payload, f, indent=2)

    lines = ["BEST_BLOCKING = {"]
    for k, v in sorted(best.items()):
        if v["blocking"] is None:
            lines.append(
                f"    {k!r}: None,  # {v['tflops']:.1f} TFLOP/s (op default {v['oob_default']}, grid {v['grid']})"
            )
        else:
            lines.append(f"    {k!r}: {v['blocking']!r},  # {v['tflops']:.1f} TFLOP/s (grid {v['grid']})")
    lines.append("}")
    logger.info(
        "Best blocking per (dtype_column, M, N, K); paste into test_minimal_matmul_benchmark.py:\n" + "\n".join(lines)
    )
    logger.info(f"Best blocking JSON written to {best_json_path}")


@pytest.fixture(scope="session")
def sweep_report():
    """Thread-safe row appender for the sweep CSV. An existing CSV with the same header is appended to, so the
    sweep can be split across pytest sessions; set TTNN_MINIMAL_MATMUL_SWEEP_RESET=1 to start a fresh CSV."""
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

    lock = threading.Lock()

    def append(row):
        with lock, open(sweep_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(c, "") for c in csv_columns()])

    yield append
    summarize_best_blocking(sweep_csv_path, artifacts_dir / BEST_BLOCKING_JSON_NAME)


def summarize_peak(peak_csv_path):
    best = {}
    with open(peak_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = row["dtype_column"]
            if key not in best or float(row["tflops"]) > float(best[key]["tflops"]):
                best[key] = row
    for key, r in sorted(best.items()):
        logger.info(
            f"PEAK {key}: {float(r['tflops']):.1f} TFLOP/s ({float(r['utilization_pct']):.1f}% of "
            f"{float(r['theoretical_tflops']):.0f}) at {r['M']}x{r['N']}x{r['K']} blocks "
            f"{r['M_block']}/{r['K_block']}/{r['N_block']} sub {r['subblock_h']}x{r['subblock_w']} grid {r['grid']}"
        )


@pytest.fixture(scope="session")
def peak_report():
    """Row appender for the peak-search CSV (same columns and append/reset rules as the sweep CSV)."""
    artifacts_dir = Path(os.environ["TT_METAL_HOME"]) / "generated"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    peak_csv_path = artifacts_dir / PEAK_CSV_NAME
    reuse = False
    if peak_csv_path.exists() and os.getenv(SWEEP_RESET_ENV) != "1":
        with open(peak_csv_path, newline="") as f:
            reuse = next(csv.reader(f), None) == csv_columns()
    if not reuse:
        with open(peak_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_columns())
    lock = threading.Lock()

    def append(row):
        with lock, open(peak_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(c, "") for c in csv_columns()])

    yield append
    summarize_peak(peak_csv_path)


@pytest.fixture(scope="module")
def sweep_chips(request):
    """The chips the sweep runs on. N=1: the conftest `device` fixture, requested per test. N>1: the system
    mesh, opened once per module and carved into 1x1 submeshes, of which the first N are used."""
    n = get_num_chips()
    if n == 1:
        yield None
        return
    mesh = ttnn.open_mesh_device(**DEVICE_PARAMS)
    submeshes = mesh.create_submeshes(ttnn.MeshShape(1, 1))
    assert len(submeshes) >= n, f"{NUM_CHIPS_ENV}={n} but the system mesh has only {len(submeshes)} chips"
    logger.info(f"Opened system mesh {mesh.shape} and carved {len(submeshes)} 1x1 submeshes; sweeping on {n}")
    yield submeshes[:n]
    for submesh in mesh.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh)


# ---------------------------------------------------------------------------
# Sweep test: one item per (shape, dtype column), configs dealt round-robin across chips
# ---------------------------------------------------------------------------


def run_candidates_on_chips(chips, candidates, ctx, report):
    """Deal the candidates round-robin to the chips, one thread per chip; returns the collected results."""
    results = []
    if len(chips) == 1:
        sweep_chip_worker(0, chips[0], candidates, ctx, report, results)
        return results
    errors = []

    def run(chip_index, chip):
        try:
            sweep_chip_worker(chip_index, chip, candidates[chip_index :: len(chips)], ctx, report, results)
        except BaseException as e:  # re-raised in the main thread
            logger.exception(f"chip {chip_index} worker failed")
            errors.append(e)

    threads = [threading.Thread(target=run, args=(i, chip), name=f"sweep-chip-{i}") for i, chip in enumerate(chips)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return results


def sweep_chip_worker(chip_index, chip, candidates, ctx, report, results):
    """Run this chip's share of the candidates. `results` collects (tflops, mode, blocking) per row."""
    M, N, K = ctx["shape"]
    dtype_column, spec = ctx["dtype_column"], ctx["spec"]
    dtype, fidelity, passes = spec["dtype"], spec["fidelity"], spec["passes"]
    in0, in1, in0_t, in1_t = make_inputs(chip, M, N, K, dtype)
    try:
        for mode, blocking, recorded_blocking, num_cores, grid_str in candidates:
            config = make_matmul_config(blocking, ctx["core_grid"]) if blocking is not None else None
            op_fn = minimal_matmul_fn(config, ctx["compute_kernel_config"])

            output_t = None
            try:
                inference_time_avg, trisc1_dur, device_freq_hz, output_t = run_measurement(
                    chip,
                    in0_t,
                    in1_t,
                    op_fn,
                    True,
                    ctx["num_warmup_iterations"],
                    ctx["num_measurement_iterations"],
                    ctx["calc_device_utilization"],
                    trace_executions=TRACE_EXECUTIONS,
                )
                metrics = compute_metrics(M, N, K, passes, num_cores, inference_time_avg, trisc1_dur, device_freq_hz)
                row_pcc = check_rows_pcc(in0, in1, output_t) if mode == "oob" else ""
            except RuntimeError as e:
                if not is_skippable_benchmark_runtime_error(e):
                    raise
                logger.warning(
                    f"chip {chip_index}: skipping {dtype_column} {shape_id((M, N, K))} {mode} {recorded_blocking}: "
                    f"{runtime_error_reason(e)}"
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
                fp32_acc=spec["fp32_acc"],
                grid=grid_str,
                chip=chip_index,
                mode=mode,
                M_block=Mb,
                K_block=Kb,
                N_block=Nb,
                subblock_h=sh,
                subblock_w=sw,
                time_avg_ms=fmt(metrics["time_avg_ms"], 4),
                tflops=fmt(metrics["measured_tflops"], 2),
                theoretical_tflops=fmt(metrics["theoretical_tflops"], 2),
                utilization_pct=fmt(metrics["utilization_pct"], 2),
                pcc=fmt(row_pcc, 5),
                **device_metric_fields(metrics),
            )
            report(row)

            device_str = (
                f", device util {metrics['device_utilization_pct']:.1f}%" if "device_utilization_pct" in metrics else ""
            )
            logger.info(
                f"chip {chip_index} [{mode}] {dtype_column} {shape_id((M, N, K))} blocks {Mb}/{Kb}/{Nb} sub {sh}x{sw} "
                f"grid {grid_str}: {metrics['time_avg_ms']:.3f} ms, {metrics['measured_tflops']:.1f} TFLOP/s, "
                f"{metrics['utilization_pct']:.1f}% of {metrics['theoretical_tflops']:.0f}{device_str}"
                + (f", pcc {row_pcc:.4f}" if row_pcc != "" else "")
            )
            results.append((metrics["measured_tflops"], mode, recorded_blocking))
    finally:
        ttnn.deallocate(in0_t)
        ttnn.deallocate(in1_t)


@pytest.mark.skipif(
    os.getenv(GEMM_FLOPS_BENCHMARK_ENV) != "1",
    reason=f"Benchmark is manual-only; set {GEMM_FLOPS_BENCHMARK_ENV}=1 to run",
)
@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("M, N, K", SHEET_SHAPES, ids=[shape_id(s) for s in SHEET_SHAPES])
@pytest.mark.parametrize("dtype_column", SWEPT_COLUMNS)
@pytest.mark.parametrize("num_warmup_iterations", [3])
@pytest.mark.parametrize("num_measurement_iterations", [10])
def test_minimal_matmul_block_sweep(
    request,
    device_params,
    grid_size,
    sweep_report,
    sweep_chips,
    M,
    N,
    K,
    dtype_column,
    num_warmup_iterations,
    num_measurement_iterations,
):
    chips = sweep_chips if sweep_chips is not None else [request.getfixturevalue("device")]
    spec = DTYPE_COLUMNS[dtype_column]

    grid_x, grid_y = resolve_grid(chips[0], grid_size)
    full_grid = chips[0].compute_with_storage_grid_size()
    num_cores_user = grid_x * grid_y
    num_cores_full = full_grid.x * full_grid.y

    ctx = dict(
        shape=(M, N, K),
        dtype_column=dtype_column,
        spec=spec,
        core_grid=ttnn.CoreCoord(grid_x, grid_y),
        compute_kernel_config=make_compute_kernel_config(chips[0], spec),
        num_warmup_iterations=num_warmup_iterations,
        num_measurement_iterations=num_measurement_iterations,
        calc_device_utilization=device_utilization_enabled(),
    )

    # oob (config=None, full device grid) first, then every candidate on the requested grid.
    candidates = [
        ("oob", None, default_block_config(M, N, spec["fp32_acc"]), num_cores_full, f"{full_grid.x}x{full_grid.y}")
    ]
    for blocking in iter_block_configs(M, N, grid_x, grid_y, spec["fp32_acc"]):
        candidates.append(("sweep", blocking, blocking, num_cores_user, f"{grid_x}x{grid_y}"))
    logger.info(
        f"{shape_id((M, N, K))} {dtype_column}: {len(candidates) - 1} sweep candidates + oob on {len(chips)} chip(s)"
    )

    results = run_candidates_on_chips(chips, candidates, ctx, sweep_report)
    assert results, f"No runnable config for {dtype_column} {shape_id((M, N, K))}"
    best = max(results, key=lambda r: r[0])
    logger.info(f"BEST {dtype_column} {shape_id((M, N, K))}: {best[2]} ({best[1]}) at {best[0]:.1f} TFLOP/s")


def peak_shape(Mb, Nb, grid_x, grid_y):
    """M, N such that every core owns exactly one Mb x Nb output block under the op's grid mapping
    (M over grid y and N over grid x, transposed when M > N). Returns None if no orientation is consistent."""
    for m_cores, n_cores in ((grid_y, grid_x), (grid_x, grid_y)):
        M, N = Mb * 32 * m_cores, Nb * 32 * n_cores
        transposed = M > N
        if (m_cores == grid_x) == transposed:
            return M, N
    return None


@pytest.mark.skipif(
    os.getenv(GEMM_FLOPS_BENCHMARK_ENV) != "1",
    reason=f"Benchmark is manual-only; set {GEMM_FLOPS_BENCHMARK_ENV}=1 to run",
)
@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("Mb, Nb", list(product(PEAK_BLOCKS, PEAK_BLOCKS)), ids=lambda v: f"{v}")
@pytest.mark.parametrize("K", PEAK_K, ids=lambda k: f"K{k}")
@pytest.mark.parametrize("dtype_column", SWEPT_COLUMNS)
@pytest.mark.parametrize("num_warmup_iterations", [3])
@pytest.mark.parametrize("num_measurement_iterations", [10])
def test_minimal_matmul_peak_search(
    request,
    device_params,
    grid_size,
    peak_report,
    sweep_chips,
    Mb,
    Nb,
    K,
    dtype_column,
    num_warmup_iterations,
    num_measurement_iterations,
):
    """Compute-bound point of the op: one output block per core, K_block and sub-block swept."""
    chips = sweep_chips if sweep_chips is not None else [request.getfixturevalue("device")]
    spec = DTYPE_COLUMNS[dtype_column]

    grid_x, grid_y = resolve_grid(chips[0], grid_size)
    shape = peak_shape(Mb, Nb, grid_x, grid_y)
    if shape is None:
        pytest.skip(f"no consistent grid orientation for blocks {Mb}x{Nb} on {grid_x}x{grid_y}")
    M, N = shape
    grid_str = f"{grid_x}x{grid_y}"
    num_cores = grid_x * grid_y

    ctx = dict(
        shape=(M, N, K),
        dtype_column=dtype_column,
        spec=spec,
        core_grid=ttnn.CoreCoord(grid_x, grid_y),
        compute_kernel_config=make_compute_kernel_config(chips[0], spec),
        num_warmup_iterations=num_warmup_iterations,
        num_measurement_iterations=num_measurement_iterations,
        calc_device_utilization=device_utilization_enabled(),
    )
    candidates = [
        ("peak", (Mb, Kb, Nb, sh, sw), (Mb, Kb, Nb, sh, sw), num_cores, grid_str)
        for Kb in get_block_sizes()
        for sh, sw in iter_subblocks(Mb, Nb, DEST_REGS[spec["fp32_acc"]])
    ]
    logger.info(
        f"peak {dtype_column} {shape_id((M, N, K))} blocks {Mb}x{Nb} per core: {len(candidates)} candidates on {len(chips)} chip(s)"
    )

    results = run_candidates_on_chips(chips, candidates, ctx, peak_report)
    if not results:
        pytest.skip(f"no runnable K_block / sub-block for {dtype_column} blocks {Mb}x{Nb} (L1)")
    best = max(results, key=lambda r: r[0])
    logger.info(f"PEAK-CANDIDATE {dtype_column} {shape_id((M, N, K))}: {best[2]} at {best[0]:.1f} TFLOP/s")
