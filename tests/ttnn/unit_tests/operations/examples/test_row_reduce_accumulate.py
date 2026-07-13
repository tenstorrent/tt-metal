# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness + single-core device profiling for the row-mean accumulation example.

Four methods take the mean across a row of `width_tiles` tiles (reduce_fold / l1_accum /
dest_accum / dest_accum_pairs), each at three precision configs "<input>-<accum>" (fp32-fp32,
bf16-fp32, bf16-bf16), over three input distributions (signal / uniform / positive). Every variant
computes the same mean; correctness is the only pass/fail. Perf (DEVICE KERNEL DURATION [ns]) and
accumulation accuracy (max/mean abs error + bf16-ULP vs the fp64 mean of the original data) are
measured and reported, never asserted.

Perf is DATA-INDEPENDENT (no data-dependent branches — every method does fixed work), so it is
measured on a single distribution; accuracy is swept over all distributions. The golden is the fp64
mean of the ORIGINAL fp32 data, so BOTH input quantization and accumulation error count. fp32-fp32
vs bf16-fp32 isolates the input-precision effect; bf16-fp32 vs bf16-bf16 the accumulation effect.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import socket
import statistics
from pathlib import Path

import torch
import ttnn
from loguru import logger

from ttnn.operations.examples.row_reduce_accumulate import (
    BASELINE,
    METHODS,
    PRECISIONS,
    create_sharded_memory_config,
    run_op,
    split_precision,
)

TILE = 32
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

_DEFAULT_WIDTHS_ELEMS = (32, 64, 128, 256, 512, 1024)

# Input value distributions (all seeded, so fp32/bf16 input see the same real data).
#   signal   — per-row linspace base + small noise: strong per-row signal, large magnitudes.
#   uniform  — uniform[-1, 1): zero-mean, mixed sign (cancellation; near-zero row means).
#   positive — uniform[0, 1): all positive (monotonically growing running sum -> bf16 swamping).
DISTRIBUTIONS = ("signal", "uniform", "positive")

# Per-precision correctness tolerance (max abs err vs the fp64 golden). Distribution-independent and
# generous: it catches wiring / scale bugs (garbage or wrong scale -> error >> tol) while allowing
# the real quantization + accumulation error, which the report quantifies.
_MAX_ABS_TOL = {"fp32-fp32": 0.05, "bf16-fp32": 0.30, "bf16-bf16": 1.00}

_TORCH_DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16}
_TTNN_DTYPE = {"fp32": ttnn.float32, "bf16": ttnn.bfloat16}


# =============================================================================
# Inputs + golden
# =============================================================================
def _gen_data(width_tiles, distribution, seed=13):
    """fp32 'true' data [32, 32*width_tiles] for a distribution (deterministic per seed)."""
    torch.manual_seed(seed)
    w = width_tiles * TILE
    if distribution == "signal":
        row_base = torch.linspace(0.25, 4.0, TILE).unsqueeze(1)
        return row_base + (torch.rand(TILE, w) - 0.5) * 0.5
    if distribution == "uniform":
        return torch.rand(TILE, w) * 2.0 - 1.0  # [-1, 1)
    if distribution == "positive":
        return torch.rand(TILE, w)  # [0, 1)
    raise ValueError(f"unknown distribution {distribution!r}; valid: {DISTRIBUTIONS}")


def _make_input(device, width_tiles, input_dtype, distribution, seed=13):
    """Resident input row [32, 32*width_tiles] in `input_dtype`, + fp64 row-mean of the ORIGINAL data."""
    data = _gen_data(width_tiles, distribution, seed)
    golden = data.to(torch.float64).mean(dim=1)
    x_dev = ttnn.from_torch(
        data.to(_TORCH_DTYPE[input_dtype]),
        dtype=_TTNN_DTYPE[input_dtype],
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config(width_tiles),
    )
    return x_dev, golden


def _row_means(output):
    """REDUCE_ROW writes the per-row reduction into column 0 of the single output tile."""
    return ttnn.to_torch(output).to(torch.float64)[:, 0]


def _ulp_bf16(x):
    """bf16 representable step at |x|: bf16 has 7 explicit mantissa bits, so ulp(2^e) = 2^(e-7)."""
    x = x.abs().to(torch.float64).clamp_min(2.0**-14)
    e = torch.floor(torch.log2(x))
    return torch.pow(torch.tensor(2.0, dtype=torch.float64), e - 7)


def _accuracy(output, golden):
    """(max abs err, mean abs err, max bf16-ULP err) of the row means vs the fp64 golden."""
    diff = (_row_means(output) - golden).abs()
    return diff.max().item(), diff.mean().item(), (diff / _ulp_bf16(golden)).max().item()


def _check(output, golden, precision, label):
    max_abs, mean_abs, max_ulp = _accuracy(output, golden)
    assert max_abs < _MAX_ABS_TOL[precision], f"{label}: max-abs {max_abs:.4f} >= {_MAX_ABS_TOL[precision]}"
    return max_abs, mean_abs, max_ulp


# =============================================================================
# In-process device-kernel timing (validated pattern — do not reinvent)
# =============================================================================
def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    total, found = 0.0, False
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _measure(device, runners, trials, kernel_iters):
    for run in runners.values():
        run()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # discard warmup window
    samples = {key: [] for key in runners}
    for trial in range(trials + 1):
        for key, run in runners.items():
            run()
            duration = _read_kernel_ns(device)
            assert duration is not None, f"no profiler data for {key}"
            if trial:  # discard first timed pass
                samples[key].append(duration / kernel_iters)
    return samples


# =============================================================================
# Config knobs (env-driven so __main__ can set them)
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
    sel = os.environ.get("RRA_WIDTHS")
    elems = tuple(int(x) for x in sel.split(",")) if sel else _DEFAULT_WIDTHS_ELEMS
    for e in elems:
        if e < 1 or e % TILE:
            raise ValueError(f"width (elements) must be a positive multiple of {TILE}, got {e}")
    return tuple(e // TILE for e in elems)


# =============================================================================
# Tests
# =============================================================================
def test_row_reduce_accumulate_correctness(device):
    methods = _selected("RRA_METHODS", METHODS)
    precisions = _selected("RRA_PRECISIONS", PRECISIONS)
    distributions = _selected("RRA_DISTS", DISTRIBUTIONS)
    # Widths cover both parities so the pairs seed path is exercised: odd (1, 3, 7, 15) seeds one tile
    # via copy_tile, even (8, 32) seeds the first pair.
    for distribution in distributions:
        for width_tiles in (1, 3, 7, 8, 15, 32):
            for precision in precisions:
                input_dtype, _ = split_precision(precision)
                x_dev, golden = _make_input(device, width_tiles, input_dtype, distribution)
                for method in methods:
                    out = run_op(x_dev, method=method, precision=precision, width_tiles=width_tiles, kernel_iters=2)
                    ma, me, ul = _check(out, golden, precision, f"{method}/{precision}/{distribution} W={width_tiles}t")
                    logger.info(
                        f"{method:18s} {precision:10s} {distribution:9s} W={width_tiles:2d}t  "
                        f"max_abs={ma:.5f} mean_abs={me:.5f} ulp_bf16={ul:.2f}"
                    )


def test_row_reduce_accumulate_device_perf(device):
    methods = _selected("RRA_METHODS", METHODS)
    precisions = _selected("RRA_PRECISIONS", PRECISIONS)
    distributions = _selected("RRA_DISTS", DISTRIBUTIONS)
    widths = _selected_widths()
    trials = _int("RRA_TRIALS", "5")
    kernel_iters = _int("RRA_KERNEL_ITERS", "200")
    perf_dist = distributions[0]  # perf is data-independent — one distribution suffices

    # ---- Accuracy sweep (all distributions), cheap kernel_iters=1; also the correctness gate. ----
    acc = {}  # (method, precision, width, distribution) -> (max_abs, mean_abs, max_ulp)
    for distribution in distributions:
        for precision in precisions:
            input_dtype, _ = split_precision(precision)
            for width_tiles in widths:
                x_dev, golden = _make_input(device, width_tiles, input_dtype, distribution)
                for method in methods:
                    out = run_op(x_dev, method=method, precision=precision, width_tiles=width_tiles, kernel_iters=1)
                    acc[(method, precision, width_tiles, distribution)] = _check(
                        out, golden, precision, f"{method}/{precision}/{distribution} W={width_tiles}t"
                    )

    # ---- Perf sweep (single distribution, data-independent). ----
    perf_inputs = {}
    for precision in precisions:
        input_dtype, _ = split_precision(precision)
        for width_tiles in widths:
            key = (input_dtype, width_tiles)
            if key not in perf_inputs:
                perf_inputs[key], _ = _make_input(device, width_tiles, input_dtype, perf_dist)
    runners = {
        (method, precision, width_tiles): (
            lambda m=method, p=precision, w=width_tiles, dt=split_precision(precision)[0]: run_op(
                perf_inputs[(dt, w)], method=m, precision=p, width_tiles=w, kernel_iters=kernel_iters
            )
        )
        for precision in precisions
        for width_tiles in widths
        for method in methods
    }
    samples = _measure(device, runners, trials, kernel_iters)

    report = _format_report(
        samples,
        acc,
        methods,
        precisions,
        distributions,
        widths,
        perf_dist=perf_dist,
        box=socket.gethostname(),
        arch=_arch_label(device),
        trials=trials,
        kernel_iters=kernel_iters,
    )
    logger.info("\n" + report)
    if report_path := os.environ.get("RRA_REPORT"):
        Path(report_path).write_text(report)


# =============================================================================
# Report
# =============================================================================
def _format_report(
    samples, acc, methods, precisions, distributions, widths, *, perf_dist, box, arch, trials, kernel_iters
):
    def med(method, precision, w):
        return statistics.median(samples[(method, precision, w)])

    def std_pct(method, precision, w):
        vals = samples[(method, precision, w)]
        m = statistics.median(vals)
        return (statistics.pstdev(vals) / m * 100) if (len(vals) > 1 and m) else 0.0

    has_base = BASELINE in methods
    width_cols = " | ".join(f"{w}t ({w * TILE}e)" for w in widths)
    w_narrow, w_wide = widths[0], widths[-1]
    perf_prec = "bf16-fp32" if "bf16-fp32" in precisions else precisions[0]

    lines = [
        "# Row-mean reduce — cross-tile accumulation methods (single core)",
        "",
        f"box={box}  arch={arch}  cores=1  placement=single-core sharded-L1  "
        f"N={trials} (median)  kernel-iters={kernel_iters} (steady-state)",
        "problem: mean across a row of W tiles ([32, 32*W] -> [32,1] row-means), output fp32, fidelity=HiFi4.",
        "",
        "Axes: METHOD (how the W tiles are summed) x PRECISION '<input>-<accum>' x input DISTRIBUTION.",
        "  methods:  reduce_fold (fold sum into the reduce, baseline) | l1_accum (packer L1-accumulate) |",
        "            dest_accum (add_tiles acc_to_dest, 1 tile/add) | dest_accum_pairs (2 tiles/add)",
        "  precision: fp32-fp32 (both precise) | bf16-fp32 (lossy input, precise accum) | bf16-bf16 (+accum loss)",
        "  distributions: signal (per-row linspace+noise, large) | uniform [-1,1) | positive [0,1)",
        "  (l1_accum's packer L1-acc is fp32-DEST-only, so its '-bf16' rounds only the L1 accumulator CB.)",
        "",
    ]

    # ======================= THE OVERVIEW TABLE =======================
    dist_order = [d for d in ("signal", "uniform", "positive") if d in distributions]
    lines += [
        "## Overview  (perf: bf16 input, ns per row-mean; accuracy: bf16 ACCUMULATION vs fp64, at the widest row)",
        "",
        f"Perf = median ns at {w_narrow}t (narrow) and {w_wide}t (wide, ×vs reduce_fold), bf16 input, "
        f"kernel-iters={kernel_iters}. Accuracy = `max_abs \\| max ULP_bf16` of the **bf16-bf16** config at "
        f"{w_wide}t, per input distribution — where precision is actually lost (fp32 accumulation is ~exact).",
        "",
        "| method | ns @"
        + f"{w_narrow}t"
        + " | ns @"
        + f"{w_wide}t"
        + " (×) | "
        + " | ".join(f"{d}: max\\|ULP" for d in dist_order)
        + " |",
        "|---|---:|---:|" + "---:|" * len(dist_order),
    ]
    for method in methods:
        n_narrow = med(method, perf_prec, w_narrow)
        n_wide = med(method, perf_prec, w_wide)
        spd = ""
        if has_base and method != BASELINE:
            base = med(BASELINE, perf_prec, w_wide)
            spd = f" ({base / n_wide:.2f}x)" if base else ""
        acc_cells = []
        for dist in dist_order:
            cell = acc.get((method, "bf16-bf16", w_wide, dist)) if "bf16-bf16" in precisions else None
            acc_cells.append(f"{cell[0]:.1e} \\| {cell[2]:.0f}u" if cell else "—")
        lines.append(f"| {method} | {n_narrow:.0f} | {n_wide:.0f}{spd} | " + " | ".join(acc_cells) + " |")
    lines += [
        "",
        "How to read it:",
        "- **fp32 accumulation is essentially exact** (fp32-fp32 and bf16-fp32 keep max_abs ≤ ~3e-3 for every "
        "method/distribution/width), so the bf16-bf16 numbers above are the whole accuracy story.",
        "- **bf16 *input* alone is nearly free** for a wide mean: its error *averages DOWN* with width (see the "
        "bf16-fp32 detail); bf16 *accumulation* is what *grows UP* with width and separates the methods.",
        "- On **signal / positive** (nonzero mean) the ordering is reduce_fold (worst) > dest_accum > "
        "dest_accum_pairs > l1_accum (best) — the running sum swamps small increments in bf16, worst when it "
        "is folded whole into one bf16 DEST (reduce_fold). **dest_accum_pairs is both fastest and the more "
        "accurate DEST-add method.**",
        "- On **uniform** (zero-mean) max_abs is tiny for ALL methods (~1e-3) — a near-zero mean has little "
        "magnitude to lose — but ULP is large because it divides that error by the ~0 mean (relative error is "
        "high near zero). So max_abs, not ULP, is the honest metric for cancellation-heavy data; there the "
        "method choice barely matters in absolute terms.",
        "- **SFPU finalize** (dest_accum_sfpu / dest_accum_pairs_sfpu) does the within-tile collapse on the SFPU "
        "in DEST (`sfpu_reduce` + a scalar-multiply for 1/N) instead of the FPU reduce library. It reads DEST "
        "natively (no pack->L1->unpack round-trip) yet is NOT faster — the SFPU vector reduce costs more than "
        "the FPU matmul-reduce, marginally outweighing the saved round-trip — but it is slightly MORE accurate "
        "in bf16 (it collapses the columns in fp32 internally before the single output rounding).",
        "- fp32 input roughly halves the wide-row perf win (it unpacks 2× the bytes); see the per-precision perf.",
        "",
    ]

    # ======================= PERF (per precision) =======================
    lines += [
        f"## Perf — median ns per row-mean, distribution={perf_dist} (data-independent); speedup vs reduce_fold",
        "",
    ]
    for precision in precisions:
        lines += [f"### {precision}", "", f"| method | {width_cols} |", "|" + "---|" * (len(widths) + 1)]
        for method in methods:
            cells = []
            for w in widths:
                m = med(method, precision, w)
                spd = ""
                if has_base and method != BASELINE:
                    base = med(BASELINE, precision, w)
                    spd = f"  ({base / m:.2f}x)" if base else ""
                cells.append(f"{m:.0f}±{std_pct(method, precision, w):.0f}%{spd}")
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
        lines.append("")

    # ======================= ACCURACY (per distribution, full metrics) =======================
    lines += [
        "## Accuracy — error vs fp64 mean of the original data  (cell = max_abs \\| mean_abs \\| max ULP_bf16)",
        "",
    ]
    for distribution in distributions:
        lines += [f"### distribution = {distribution}", ""]
        for precision in precisions:
            lines += [f"**{precision}**", "", f"| method | {width_cols} |", "|" + "---|" * (len(widths) + 1)]
            for method in methods:
                cells = []
                for w in widths:
                    cell = acc.get((method, precision, w, distribution))
                    if cell:
                        ma, me, ul = cell
                        cells.append(f"{ma:.1e} \\| {me:.1e} \\| {ul:.2f}u")
                    else:
                        cells.append("—")
                lines.append(f"| {method} | " + " | ".join(cells) + " |")
            lines.append("")

    return "\n".join(lines) + "\n"
