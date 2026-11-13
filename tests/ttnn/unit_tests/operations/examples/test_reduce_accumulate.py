# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness + single-core device profiling for the accumulate+SFPU-finalize reduce fast path.

A SUM/mean reduce built as elementwise ACCUMULATE (pairwise add_tiles into DEST) + within-tile SFPU
FINALIZE, across all three reduce dims (row=width, col=height, scalar=both), compared against the
standard reduce library. Variants: helper (baseline reduce), fast (accumulate+SFPU), dispatch
(fast when num_tiles >= DISPATCH_MIN_TILES, else helper). Correctness is the only pass/fail; perf
(DEVICE KERNEL DURATION [ns]) and accuracy (max/mean abs + bf16-ULP vs the fp64 mean) are measured.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import socket
import statistics
from pathlib import Path

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.unit_tests.operations.examples.perf_test_utils import check_regime, interleaved_ns, read_kernel_ns
from ttnn.operations.examples.reduce_accumulate import (
    BASELINE,
    DIMS,
    DTYPES,
    VARIANTS,
    create_sharded_memory_config,
    dispatch_min,
    input_shape,
    run_op,
)

TILE = 32
_DEFAULT_WIDTHS = (1, 2, 4, 8, 16, 32)  # tiles reduced

# Per-accum correctness tolerance (max abs err vs fp64 mean) — catches wiring/scale bugs while
# allowing the real bf16 quantization + accumulation error, which the report quantifies.
_MAX_ABS_TOL = {"fp32": 0.05, "bf16": 1.00}


# =============================================================================
# Inputs + golden (positive [0,1) distribution: all-positive, nonzero mean -> ULP meaningful)
# =============================================================================
def _make_input(device, dim, num_tiles, seed=13):
    torch.manual_seed(seed)
    h, w = input_shape(dim, num_tiles)
    data = torch.rand(h, w)  # fp32 "true" data
    if dim == "row":
        golden = data.to(torch.float64).mean(dim=1)  # [32] per-row
    elif dim == "col":
        golden = data.to(torch.float64).mean(dim=0)  # [32] per-col
    else:
        golden = data.to(torch.float64).mean().reshape(1)  # scalar
    x_dev = ttnn.from_torch(
        data.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config((h, w)),
    )
    return x_dev, golden


def _readout(output, dim):
    """The meaningful slice of the single output tile for each reduce dim."""
    t = ttnn.to_torch(output).to(torch.float64)
    if dim == "row":
        return t[:, 0]  # REDUCE_ROW -> per-row means in column 0
    if dim == "col":
        return t[0, :]  # REDUCE_COL -> per-col means in row 0
    return t[0, 0].reshape(1)  # REDUCE_SCALAR -> single mean at [0, 0]


def _ulp_bf16(x):
    x = x.abs().to(torch.float64).clamp_min(2.0**-14)
    e = torch.floor(torch.log2(x))
    return torch.pow(torch.tensor(2.0, dtype=torch.float64), e - 7)


def _accuracy(output, golden, dim):
    diff = (_readout(output, dim) - golden).abs()
    return diff.max().item(), diff.mean().item(), (diff / _ulp_bf16(golden)).max().item()


def _check(output, golden, dim, accum, label):
    max_abs, mean_abs, max_ulp = _accuracy(output, golden, dim)
    assert max_abs < _MAX_ABS_TOL[accum], f"{label}: max-abs {max_abs:.4f} >= {_MAX_ABS_TOL[accum]}"
    return max_abs, mean_abs, max_ulp


# =============================================================================
# In-process device-kernel timing (validated pattern)
# =============================================================================
def _measure(device, runners, trials, kernel_iters):
    for run in runners.values():
        run()
    ttnn.synchronize_device(device)
    read_kernel_ns(device)
    samples = {key: [] for key in runners}
    for trial in range(trials + 1):
        for key, run in runners.items():
            run()
            duration = read_kernel_ns(device)
            assert duration is not None, f"no profiler data for {key}"
            if trial:
                samples[key].append(duration / kernel_iters)
    return samples


# =============================================================================
# Config knobs
# =============================================================================
def _int(name, default):
    return int(os.environ.get(name, default))


def _arch_label(device):
    if name := os.environ.get("ARCH_NAME"):
        return name
    a = str(device.arch()).rsplit(".", 1)[-1]
    return {"WORMHOLE_B0": "WH_B0", "BLACKHOLE": "BH", "GRAYSKULL": "GS"}.get(a, a)


def _selected(env_name, allowed):
    sel = os.environ.get(env_name)
    chosen = tuple(sel.split(",")) if sel else allowed
    unknown = set(chosen) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {env_name}: {sorted(unknown)}; valid: {allowed}")
    return tuple(v for v in allowed if v in chosen)


def _selected_widths():
    sel = os.environ.get("RA_WIDTHS")
    widths = tuple(int(x) for x in sel.split(",")) if sel else _DEFAULT_WIDTHS
    for n in widths:
        if n < 1:
            raise ValueError(f"width (tiles) must be positive, got {n}")
    return widths


# =============================================================================
# Tests
# =============================================================================
def test_reduce_accumulate_correctness(device):
    variants = _selected("RA_VARIANTS", VARIANTS)
    dims = _selected("RA_DIMS", DIMS)
    accums = _selected("RA_ACCUMS", DTYPES)
    for dim in dims:
        for num_tiles in (1, 3, 8, 32):  # odd + even, small + large
            x_dev, golden = _make_input(device, dim, num_tiles)
            for accum in accums:
                for variant in variants:
                    out = run_op(x_dev, variant=variant, dim=dim, num_tiles=num_tiles, accum=accum, kernel_iters=2)
                    ma, me, ul = _check(out, golden, dim, accum, f"{variant}/{dim}/{accum} N={num_tiles}")
                    logger.info(
                        f"{variant:9s} {dim:7s} {accum:5s} N={num_tiles:2d}  max_abs={ma:.5f} mean_abs={me:.5f} ulp={ul:.2f}"
                    )


def test_reduce_accumulate_device_perf(device):
    variants = _selected("RA_VARIANTS", VARIANTS)
    dims = _selected("RA_DIMS", DIMS)
    accums = _selected("RA_ACCUMS", DTYPES)
    widths = _selected_widths()
    trials = _int("RA_TRIALS", "5")
    kernel_iters = _int("RA_KERNEL_ITERS", "200")

    # inputs shared across variants/accums — keyed by (dim, width)
    inputs, goldens = {}, {}
    for dim in dims:
        for n in widths:
            inputs[(dim, n)], goldens[(dim, n)] = _make_input(device, dim, n)

    # Accuracy sweep (both accum dtypes) + correctness gate.
    acc = {}  # (variant, dim, accum, n) -> (max_abs, mean_abs, max_ulp)
    for dim in dims:
        for n in widths:
            for accum in accums:
                for variant in variants:
                    out = run_op(inputs[(dim, n)], variant=variant, dim=dim, num_tiles=n, accum=accum, kernel_iters=1)
                    acc[(variant, dim, accum, n)] = _check(
                        out, goldens[(dim, n)], dim, accum, f"{variant}/{dim}/{accum} N={n}"
                    )

    # Perf sweep (fp32 accumulation — the natural fast-path config; bf16 accum perf is within noise).
    perf_accum = "fp32" if "fp32" in accums else accums[0]
    runners = {
        (variant, dim, n): (
            lambda v=variant, d=dim, w=n: run_op(
                inputs[(d, w)], variant=v, dim=d, num_tiles=w, accum=perf_accum, kernel_iters=kernel_iters
            )
        )
        for dim in dims
        for n in widths
        for variant in variants
    }
    samples = _measure(device, runners, trials, kernel_iters)

    report = _format_report(
        samples,
        acc,
        variants,
        dims,
        accums,
        widths,
        perf_accum=perf_accum,
        box=socket.gethostname(),
        arch=_arch_label(device),
        trials=trials,
        kernel_iters=kernel_iters,
        thresholds={dim: dispatch_min(dim, device) for dim in DIMS},
    )
    logger.info("\n" + report)
    if report_path := os.environ.get("RA_REPORT"):
        Path(report_path).write_text(report)


# =============================================================================
# Report
# =============================================================================
def _format_report(
    samples, acc, variants, dims, accums, widths, *, perf_accum, box, arch, trials, kernel_iters, thresholds
):
    def med(variant, dim, n):
        return statistics.median(samples[(variant, dim, n)])

    def std_pct(variant, dim, n):
        vals = samples[(variant, dim, n)]
        m = statistics.median(vals)
        return (statistics.pstdev(vals) / m * 100) if (len(vals) > 1 and m) else 0.0

    has_base = BASELINE in variants
    width_cols = " | ".join(f"{n}t" for n in widths)
    lines = [
        "# Reduce via accumulate + SFPU finalize vs the standard reduce library (single core)",
        "",
        f"box={box}  arch={arch}  cores=1  placement=single-core sharded-L1  "
        f"N={trials} (median)  kernel-iters={kernel_iters} (steady-state)",
        "problem: SUM/mean reduce of N tiles, built as pairwise add_tiles accumulate + SFPU finalize, "
        "vs the reduce library (FPU). Widths are the tile count reduced. Output fp32, input bf16, HiFi4.",
        "",
        "variants: helper (standard reduce library) | fast (accumulate + SFPU finalize) | "
        f"dispatch (fast when N >= the per-dim threshold for this arch [row={thresholds['row']}, "
        f"col={thresholds['col']}, scalar={thresholds['scalar']} tiles], else helper).",
        "dims: row (reduce width, REDUCE_ROW) | col (reduce height, REDUCE_COL) | scalar (reduce both).",
        f"perf measured at accum={perf_accum} (data-independent; bf16 accum within noise).",
        "",
    ]

    # ---- Perf: one table per dim, rows=variant, cols=width, ns + speedup vs helper. ----
    lines += ["## Perf — median ns per reduce; speedup vs helper", ""]
    for dim in dims:
        lines += [f"### dim = {dim}", "", f"| variant | {width_cols} |", "|" + "---|" * (len(widths) + 1)]
        for variant in variants:
            cells = []
            for n in widths:
                m = med(variant, dim, n)
                spd = ""
                if has_base and variant != BASELINE:
                    base = med(BASELINE, dim, n)
                    spd = f"  ({base / m:.2f}x)" if base else ""
                cells.append(f"{m:.0f}±{std_pct(variant, dim, n):.0f}%{spd}")
            lines.append(f"| {variant} | " + " | ".join(cells) + " |")
        lines.append("")

    # ---- Accuracy: per dim, rows = variant x accum, cols = width, cell = max_abs | ULP. ----
    lines += [
        "## Accuracy — error vs fp64 mean  (cell = max_abs \\| max ULP_bf16)",
        "",
    ]
    for dim in dims:
        lines += [f"### dim = {dim}", "", f"| variant.accum | {width_cols} |", "|" + "---|" * (len(widths) + 1)]
        for accum in accums:
            for variant in variants:
                cells = []
                for n in widths:
                    cell = acc.get((variant, dim, accum, n))
                    cells.append(f"{cell[0]:.1e} \\| {cell[2]:.1f}u" if cell else "—")
                lines.append(f"| {variant}.{accum} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += [
        f"Notes: the fast path's crossover depends on the reduce dim AND the arch (`dispatch` thresholds for "
        f"{arch}: row={thresholds['row']}, col={thresholds['col']}, scalar={thresholds['scalar']} tiles; "
        "per-arch crossovers in README.md, When it helps). `dispatch` falls back to the library below the "
        "threshold, so it is never slower at the measured widths. "
        "It generalizes to all three dims (SFPU `sfpu_reduce` does REDUCE_ROW/COL; scalar is ROW then COL). "
        "Accuracy: fast ~= helper in fp32 and MORE accurate in bf16 (SFPU collapses columns in fp32 before one "
        "rounding); on **scalar** fast multiplies by 1/N once vs the library's AVG-scalar 1/sqrt(N)-twice, so it "
        "is ~100x more accurate in fp32.",
    ]
    return "\n".join(lines) + "\n"


# =============================================================================
# Regimes: where the accumulate+SFPU-finalize fast path helps vs does not
# =============================================================================
# (id, reduce dim, tiles reduced per output tile, kind, measured speedup per arch)
#   speedup = helper_ns / fast_ns (baseline `helper`, lever `fast`; both on 1 Tensix core, sharded L1, bf16 in,
#   fp32 accumulation, HiFi4, 200 in-kernel iters, ns per in-kernel iter).
#   measured = {"wh": Wormhole B0 n150, "bh": Blackhole p150b}, from README.md "When it helps".
# The bound is derived per arch from the number measured on THAT arch by `perf_test_utils.regime_bound`
# (helps: about half the measured gain, at least the noise margin below it; does_not_help: at most the
# measured value plus the noise margin, never below 1.0x). An arch with no measured number is logged, not
# asserted; a missed bound gets one confirmation pass (pooled samples) before it fails. `measure` regimes
# only log: their verdict differs between boards.
# These regimes compare `helper` and `fast` directly, so the `dispatch` thresholds do not affect them.
_REGIME_TRIALS = 11
_REGIME_KERNEL_ITERS = 200
_REGIMES = [
    ("helps_row_32t", "row", 32, "helps", {"wh": 1.96, "bh": 1.73}),
    ("helps_col_32t", "col", 32, "helps", {"wh": 1.72, "bh": 2.97}),
    ("helps_scalar_32t", "scalar", 32, "helps", {"wh": 2.93, "bh": 3.73}),
    ("not_row_1t", "row", 1, "does_not_help", {"wh": 0.50, "bh": 0.47}),
    ("not_row_4t", "row", 4, "does_not_help", {"wh": 0.83, "bh": 0.69}),
    ("not_col_4t", "col", 4, "does_not_help", {"wh": 0.81, "bh": 0.96}),
    # BH: 0.97x on the old reduce helper, 1.05x on the current one -- flat, so only logged.
    ("flat_scalar_4t", "scalar", 4, "measure", {"wh": 0.95, "bh": (0.97, 1.05)}),
]


@pytest.mark.parametrize("regime", _REGIMES, ids=[r[0] for r in _REGIMES])
def test_reduce_accumulate_regimes(device, regime):
    name, dim, n, kind, measured = regime
    x_dev, golden = _make_input(device, dim, n)

    def runner(variant):
        return lambda: run_op(
            x_dev, variant=variant, dim=dim, num_tiles=n, accum="fp32", kernel_iters=_REGIME_KERNEL_ITERS
        )

    # correctness of both variants (reuses the file's accuracy gate)
    for variant in ("helper", "fast"):
        out = run_op(x_dev, variant=variant, dim=dim, num_tiles=n, accum="fp32", kernel_iters=1)
        _check(out, golden, dim, "fp32", f"{variant}/{dim} N={n}")

    def measure():
        samples = interleaved_ns(
            device,
            {"helper": runner("helper"), "fast": runner("fast")},
            trials=_REGIME_TRIALS,
            per_launch_divisor=_REGIME_KERNEL_ITERS,
        )
        return samples["helper"], samples["fast"]

    check_regime(
        device,
        name,
        kind,
        measured,
        measure,
        detail=f"dim={dim} tiles reduced={n}, 1 Tensix core, ns per in-kernel iter (of {_REGIME_KERNEL_ITERS})",
    )
