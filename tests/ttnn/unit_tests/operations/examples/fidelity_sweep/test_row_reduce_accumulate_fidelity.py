# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fidelity x DEST sweep for the row_reduce_accumulate example (6 row-mean methods).

Extends the committed row_reduce_accumulate sweep with two new axes:
  * math_fidelity in {LoFi, HiFi2, HiFi3, HiFi4}
  * DEST mode = accumulation dtype, via the existing precision axis:
        bf16-fp32 = fp32 DEST (fp32_dest_acc_en ON)   |   bf16-bf16 = bf16 DEST (OFF)
    (input is always bf16 here; output always fp32.)

Same in-process device-profiler timing + fp64-golden accuracy as the committed test. distribution =
signal (per-row linspace+noise, large magnitudes — where fidelity/precision loss shows most).
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
    create_sharded_memory_config,
    run_op,
)

TILE = 32
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

FIDELITIES = ("LoFi", "HiFi2", "HiFi3", "HiFi4")
_FID = {f: getattr(ttnn.MathFidelity, f) for f in FIDELITIES}
# DEST mode expressed through the precision axis (input fixed bf16).
DEST = (("fp32", "bf16-fp32"), ("bf16", "bf16-bf16"))  # (label, precision)
# Representative methods (the reduce_accumulate sweep already showed the add/SFPU paths are
# fidelity-flat): reduce_fold = W matmul-reduces (strongly fidelity-sensitive); l1_accum + one
# finalize reduce (fp32-DEST-forced); dest_accum_pairs = FPU one-reduce finalize; the _sfpu twin =
# SFPU finalize (no FPU multiply at all). Trimmed widths keep the cold-compile count under the
# 300s pytest timeout (each row_reduce combo is a distinct heavy kernel compile).
SWEEP_METHODS = ("reduce_fold", "l1_accum", "dest_accum_pairs", "dest_accum_pairs_sfpu")
WIDTHS = (1, 8, 32)
TRIALS = 5
KERNEL_ITERS = 200
_MAX_ABS_TOL = 1.00  # generous correctness gate (bf16 accum, signal magnitudes)


def _gen_data(width_tiles, seed=13):
    """signal distribution: per-row linspace base + small noise (matches the committed test)."""
    torch.manual_seed(seed)
    w = width_tiles * TILE
    row_base = torch.linspace(0.25, 4.0, TILE).unsqueeze(1)
    return row_base + (torch.rand(TILE, w) - 0.5) * 0.5


def _make_input(device, width_tiles):
    data = _gen_data(width_tiles)
    golden = data.to(torch.float64).mean(dim=1)
    x_dev = ttnn.from_torch(
        data.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config(width_tiles),
    )
    return x_dev, golden


def _ulp_bf16(x):
    x = x.abs().to(torch.float64).clamp_min(2.0**-14)
    e = torch.floor(torch.log2(x))
    return torch.pow(torch.tensor(2.0, dtype=torch.float64), e - 7)


def _accuracy(output, golden):
    diff = (ttnn.to_torch(output).to(torch.float64)[:, 0] - golden).abs()
    return diff.max().item(), diff.mean().item(), (diff / _ulp_bf16(golden)).max().item()


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


def test_row_reduce_accumulate_fidelity(device):
    # ---- accuracy sweep (kernel_iters=1) + correctness gate ----
    acc = {}  # (method, dest_label, fid, w) -> (max_abs, mean_abs, max_ulp)
    for w in WIDTHS:
        x_dev, golden = _make_input(device, w)
        for dest_label, precision in DEST:
            for fid in FIDELITIES:
                for method in SWEEP_METHODS:
                    out = run_op(
                        x_dev,
                        method=method,
                        precision=precision,
                        width_tiles=w,
                        kernel_iters=1,
                        math_fidelity=_FID[fid],
                    )
                    ma, me, ul = _accuracy(out, golden)
                    assert ma < _MAX_ABS_TOL, f"{method}/{dest_label}/{fid} W={w}: max_abs {ma} too big"
                    acc[(method, dest_label, fid, w)] = (ma, me, ul)

    # ---- perf sweep (kernel_iters=200) ----
    inputs = {w: _make_input(device, w)[0] for w in WIDTHS}
    runners = {
        (method, dest_label, fid, w): (
            lambda m=method, p=precision, f=fid, ww=w: run_op(
                inputs[ww],
                method=m,
                precision=p,
                width_tiles=ww,
                kernel_iters=KERNEL_ITERS,
                math_fidelity=_FID[f],
            )
        )
        for w in WIDTHS
        for dest_label, precision in DEST
        for fid in FIDELITIES
        for method in SWEEP_METHODS
    }
    samples = _measure(device, runners, TRIALS, KERNEL_ITERS)

    def med(key):
        return statistics.median(samples[key])

    lines = [
        "# row_reduce_accumulate — math-fidelity x DEST sweep (single core, distribution=signal)",
        "",
        f"box={socket.gethostname()}  arch={_arch_label(device)}  cores=1  N={TRIALS} (median)  "
        f"kernel-iters={KERNEL_ITERS}",
        "input=bf16, output=fp32. DEST: fp32 = fp32_dest_acc_en ON (precision bf16-fp32), " "bf16 = OFF (bf16-bf16).",
        "note: l1_accum forces fp32 DEST (packer L1-acc is fp32-DEST-only); its 'bf16 DEST' column rounds "
        "only the L1 accumulator CB.",
        "",
        "## Perf — median ns per row-mean (rows: method x fidelity; one block per DEST)",
        "",
    ]
    for dest_label, _ in DEST:
        lines += [
            f"### DEST={dest_label}",
            "",
            "| method.fidelity | " + " | ".join(f"{w}t" for w in WIDTHS) + " |",
            "|" + "---|" * (len(WIDTHS) + 1),
        ]
        for method in SWEEP_METHODS:
            for fid in FIDELITIES:
                cells = [f"{med((method, dest_label, fid, w)):.0f}" for w in WIDTHS]
                lines.append(f"| {method}.{fid} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += [
        "## Accuracy — error vs fp64 mean (cell = max_abs | max ULP_bf16); rows: method x fidelity",
        "",
    ]
    for dest_label, _ in DEST:
        lines += [
            f"### DEST={dest_label}",
            "",
            "| method.fidelity | " + " | ".join(f"{w}t" for w in WIDTHS) + " |",
            "|" + "---|" * (len(WIDTHS) + 1),
        ]
        for method in SWEEP_METHODS:
            for fid in FIDELITIES:
                cells = []
                for w in WIDTHS:
                    ma, me, ul = acc[(method, dest_label, fid, w)]
                    cells.append(f"{ma:.1e} \\| {ul:.1f}u")
                lines.append(f"| {method}.{fid} | " + " | ".join(cells) + " |")
        lines.append("")

    report = "\n".join(lines) + "\n"
    logger.info("\n" + report)
    out_path = os.environ.get("RRAF_REPORT", str(Path(__file__).parent / "row_reduce_accumulate_fidelity_report.md"))
    Path(out_path).write_text(report)
    logger.info(f"wrote {out_path}")
