# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fidelity x DEST sweep for the reduce_accumulate example (helper vs fast, per reduce dim).

Extends the committed reduce_accumulate perf/accuracy sweep with two new axes:
  * math_fidelity in {LoFi, HiFi2, HiFi3, HiFi4}
  * DEST mode = accum dtype {fp32 (fp32_dest_acc_en=True), bf16 (fp32_dest_acc_en=False)}

Same in-process device-profiler timing + fp64-golden accuracy methodology as the committed test.
Input is bf16, output fp32, single core, sharded L1. Correctness gate + perf + accuracy, reported.
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

from ttnn.operations.examples.reduce_accumulate import create_sharded_memory_config, input_shape, run_op

TILE = 32
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

FIDELITIES = ("LoFi", "HiFi2", "HiFi3", "HiFi4")
_FID = {f: getattr(ttnn.MathFidelity, f) for f in FIDELITIES}
VARIANTS = ("helper", "fast", "dispatch")  # dispatch routes helper<->fast per (dim, tiles) threshold
DIMS = ("row", "col", "scalar")
ACCUMS = ("fp32", "bf16")  # DEST mode: fp32 = fp32_dest_acc_en on, bf16 = off
WIDTHS = (1, 2, 4, 8, 16, 32)
TRIALS = 5
KERNEL_ITERS = 200
_MAX_ABS_TOL = {"fp32": 0.05, "bf16": 1.00}


# ---- inputs + golden (positive [0,1), matching the committed reduce_accumulate test) ----
def _make_input(device, dim, num_tiles, seed=13):
    torch.manual_seed(seed)
    h, w = input_shape(dim, num_tiles)
    data = torch.rand(h, w)
    if dim == "row":
        golden = data.to(torch.float64).mean(dim=1)
    elif dim == "col":
        golden = data.to(torch.float64).mean(dim=0)
    else:
        golden = data.to(torch.float64).mean().reshape(1)
    x_dev = ttnn.from_torch(
        data.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config((h, w)),
    )
    return x_dev, golden


def _readout(output, dim):
    t = ttnn.to_torch(output).to(torch.float64)
    if dim == "row":
        return t[:, 0]
    if dim == "col":
        return t[0, :]
    return t[0, 0].reshape(1)


def _ulp_bf16(x):
    x = x.abs().to(torch.float64).clamp_min(2.0**-14)
    e = torch.floor(torch.log2(x))
    return torch.pow(torch.tensor(2.0, dtype=torch.float64), e - 7)


def _accuracy(output, golden, dim):
    diff = (_readout(output, dim) - golden).abs()
    return diff.max().item(), diff.mean().item(), (diff / _ulp_bf16(golden)).max().item()


# ---- in-process device-kernel timing (validated pattern, verbatim) ----
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
    _read_kernel_ns(device)
    samples = {key: [] for key in runners}
    for trial in range(trials + 1):
        for key, run in runners.items():
            run()
            duration = _read_kernel_ns(device)
            assert duration is not None, f"no profiler data for {key}"
            if trial:
                samples[key].append(duration / kernel_iters)
    return samples


def _arch_label(device):
    a = str(device.arch()).rsplit(".", 1)[-1]
    return {"WORMHOLE_B0": "WH_B0", "BLACKHOLE": "BH", "GRAYSKULL": "GS"}.get(a, a)


def test_reduce_accumulate_fidelity(device):
    # ---- accuracy sweep (kernel_iters=1) + correctness gate ----
    acc = {}  # (variant, dim, fid, accum, w) -> (max_abs, mean_abs, max_ulp)
    for dim in DIMS:
        for w in WIDTHS:
            x_dev, golden = _make_input(device, dim, w)
            for fid in FIDELITIES:
                for accum in ACCUMS:
                    for variant in VARIANTS:
                        out = run_op(
                            x_dev,
                            variant=variant,
                            dim=dim,
                            num_tiles=w,
                            accum=accum,
                            kernel_iters=1,
                            math_fidelity=_FID[fid],
                        )
                        ma, me, ul = _accuracy(out, golden, dim)
                        assert ma < _MAX_ABS_TOL[accum], f"{variant}/{dim}/{fid}/{accum} N={w}: max_abs {ma} too big"
                        acc[(variant, dim, fid, accum, w)] = (ma, me, ul)

    # ---- perf sweep (kernel_iters=200) ----
    inputs = {(dim, w): _make_input(device, dim, w)[0] for dim in DIMS for w in WIDTHS}
    runners = {
        (variant, dim, fid, accum, w): (
            lambda v=variant, d=dim, f=fid, a=accum, ww=w: run_op(
                inputs[(d, ww)],
                variant=v,
                dim=d,
                num_tiles=ww,
                accum=a,
                kernel_iters=KERNEL_ITERS,
                math_fidelity=_FID[f],
            )
        )
        for dim in DIMS
        for w in WIDTHS
        for fid in FIDELITIES
        for accum in ACCUMS
        for variant in VARIANTS
    }
    samples = _measure(device, runners, TRIALS, KERNEL_ITERS)

    def med(variant, dim, fid, accum, w):
        return statistics.median(samples[(variant, dim, fid, accum, w)])

    def std_pct(variant, dim, fid, accum, w):
        vals = samples[(variant, dim, fid, accum, w)]
        m = statistics.median(vals)
        return (statistics.pstdev(vals) / m * 100) if (len(vals) > 1 and m) else 0.0

    # ---- report ----
    lines = [
        "# reduce_accumulate — math-fidelity x tiles x reduce-dim x DEST perf sweep (single core)",
        "",
        f"box={socket.gethostname()}  arch={_arch_label(device)}  cores=1  N={TRIALS} (median)  "
        f"kernel-iters={KERNEL_ITERS}",
        "input=bf16, output=fp32. DEST: fp32 = fp32_dest_acc_en ON, bf16 = OFF. fidelity in {LoFi,HiFi2,HiFi3,HiFi4}.",
        "variants: helper (reduce library, FPU matmul-with-ones) | fast (add_tiles accumulate + SFPU finalize) | "
        "dispatch (helper below the per-dim tile threshold, fast at/above).",
        "cell = median ns ±std% (x vs helper at the same dim+fidelity+tiles).",
        "",
        "## Perf — median ns per reduce (rows: variant x fidelity; one block per reduce-dim x DEST)",
        "",
    ]
    for dim in DIMS:
        for accum in ACCUMS:
            lines += [
                f"### dim={dim}, DEST={accum}",
                "",
                "| variant.fidelity | " + " | ".join(f"{w}t" for w in WIDTHS) + " |",
                "|" + "---|" * (len(WIDTHS) + 1),
            ]
            for variant in VARIANTS:
                for fid in FIDELITIES:
                    cells = []
                    for w in WIDTHS:
                        m = med(variant, dim, fid, accum, w)
                        spd = ""
                        if variant != "helper":
                            base = med("helper", dim, fid, accum, w)
                            spd = f"  ({base / m:.2f}x)" if base else ""
                        cells.append(f"{m:.0f}±{std_pct(variant, dim, fid, accum, w):.0f}%{spd}")
                    lines.append(f"| {variant}.{fid} | " + " | ".join(cells) + " |")
            lines.append("")

    # Fidelity tax: HiFi4 / LoFi ns ratio (how much fidelity costs) at the widest row, per variant x dim.
    wmax = WIDTHS[-1]
    lines += [
        f"## Fidelity tax — HiFi4 / LoFi ns ratio at {wmax}t (DEST=fp32); >1 = HiFi4 is that much slower",
        "",
        "| variant | " + " | ".join(DIMS) + " |",
        "|---|" + "---|" * len(DIMS),
    ]
    for variant in VARIANTS:
        ratios = []
        for dim in DIMS:
            lo = med(variant, dim, "LoFi", "fp32", wmax)
            hi = med(variant, dim, "HiFi4", "fp32", wmax)
            ratios.append(f"{hi / lo:.2f}x" if lo else "—")
        lines.append(f"| {variant} | " + " | ".join(ratios) + " |")
    lines.append("")

    lines += [
        "## Accuracy — error vs fp64 mean (cell = max_abs | max ULP_bf16); rows: variant x fidelity",
        "",
    ]
    for dim in DIMS:
        for accum in ACCUMS:
            lines += [
                f"### dim={dim}, DEST={accum}",
                "",
                "| variant.fidelity | " + " | ".join(f"{w}t" for w in WIDTHS) + " |",
                "|" + "---|" * (len(WIDTHS) + 1),
            ]
            for variant in VARIANTS:
                for fid in FIDELITIES:
                    cells = []
                    for w in WIDTHS:
                        ma, me, ul = acc[(variant, dim, fid, accum, w)]
                        cells.append(f"{ma:.1e} \\| {ul:.1f}u")
                    lines.append(f"| {variant}.{fid} | " + " | ".join(cells) + " |")
            lines.append("")

    report = "\n".join(lines) + "\n"
    logger.info("\n" + report)
    out_path = os.environ.get("RAF_REPORT", str(Path(__file__).parent / "reduce_accumulate_fidelity_report.md"))
    Path(out_path).write_text(report)
    logger.info(f"wrote {out_path}")
