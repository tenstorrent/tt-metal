# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness + MATH-THREAD-ISOLATED device profiling for SFPU work-scoping.

An SFPU unary op (rsqrt | recip) applied to a 32x32 tile, scoped to a subset of the tile's four
16x16 faces instead of the whole tile — the win when the meaningful data lives on one axis after a
reduction. Scope ladder: rc (4 faces) -> r/c (2 faces) -> face (1 face) -> face_iter1 (sub-face);
`none` is an empty reps-loop (math-thread loop overhead, ~0).

Perf is measured as pure SFPU math cycles: the input is copied into DEST once and packed out once,
both OUTSIDE a DeviceZoneScopedN, and inside the zone the scoped SFPU runs `reps` times on the
MATH thread only. A compute zone records on all three TRISCs; we read TRISC_1 (math) — no unpack,
no pack, no CB handshake in the number. The zone's unpack/pack durations come back ~0, which is the
proof the SFPU is alone on the clock. Correctness is checked separately at reps=1 on each scope's
valid region. Correctness is the only pass/fail; perf is measured, never asserted.
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
from ttnn.operations.examples.sfpu_tile_scope import (
    ABLATION,
    BASELINE,
    FUNCS,
    LABEL,
    VALID_REGION,
    VARIANTS,
    ZONE_NAME,
    create_sharded_memory_config,
    run_op,
    vectors,
)

TILE = 32

# rel-err gate vs the fp64 golden of the bf16-cast input (isolates SFPU approximation, not bf16
# quantization). The SFPU rsqrt/recip land within ~0.4% on device; 2% catches wiring/scope bugs.
_REL_TOL = 0.02
_ABS_FLOOR = 1e-3


def _golden_fn(func):
    return (lambda x: torch.rsqrt(x)) if func == "rsqrt" else (lambda x: torch.reciprocal(x))


def _make_input(device, seed=13):
    """One tile, uniform in [0.5, 2.0] (safe for both rsqrt and recip)."""
    torch.manual_seed(seed)
    data = 0.5 + 1.5 * torch.rand(TILE, TILE)
    x_dev = ttnn.from_torch(
        data.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config(),
    )
    return x_dev, data.to(torch.bfloat16).to(torch.float64)  # golden input = the bf16 the device sees


def _check(output, golden_in, func, variant, label):
    """At reps=1: max relative error over the region this scope leaves valid."""
    out = ttnn.to_torch(output).to(torch.float64)
    expected = golden_in if variant == ABLATION else _golden_fn(func)(golden_in)  # `none` = identity copy
    r0, r1, c0, c1 = VALID_REGION[variant]
    got = out[r0:r1, c0:c1]
    exp = expected[r0:r1, c0:c1]
    rel = ((got - exp).abs() / exp.abs().clamp_min(_ABS_FLOOR)).max().item()
    assert rel < _REL_TOL, f"{label}: rel-err {rel:.4f} >= {_REL_TOL}"
    return rel


# =============================================================================
# Per-TRISC device-zone timing (DeviceZoneScopedN -> profile_log_device.csv)
#
# The compute-kernel zone records ZONE_START/ZONE_END on all three TRISCs. We isolate one launch by
# its (new) run-host-id and read the MATH (TRISC_1) duration; the zone brackets the whole reps-loop,
# so math_ns / reps is the per-call SFPU cost. unpack/pack come back ~0 (nothing else in the zone).
# =============================================================================
_DEVICE_CSV = os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated/profiler/.logs/profile_log_device.csv")
_RISC_LABEL = {"TRISC_0": "unpack", "TRISC_1": "math", "TRISC_2": "pack"}


def _read_csv_rows(path):
    with open(path) as f:
        lines = f.read().splitlines()
    freq_mhz = 1000.0
    for part in lines[0].split(","):
        if "CHIP_FREQ" in part:
            freq_mhz = float(part.split(":")[1])
    rows = [[x.strip() for x in ln.split(",")] for ln in lines[2:] if ln.strip()]
    return [r for r in rows if len(r) >= 12], (1000.0 / freq_mhz), freq_mhz


def _all_run_ids(path):
    if not os.path.exists(path):
        return set()
    rows, _, _ = _read_csv_rows(path)
    return {r[7] for r in rows}


def _zone_engines_for_new_run(path, seen_ids):
    """{engine: ns} for the ZONE_NAME zone whose run-host-id is new (this launch only)."""
    rows, ns_per_cycle, _ = _read_csv_rows(path)
    starts, ends = {}, {}
    for r in rows:
        risc, cyc, run_id, zone, typ = r[3], r[5], r[7], r[10], r[11]
        if run_id in seen_ids or zone != ZONE_NAME:
            continue
        (starts if typ == "ZONE_START" else ends).setdefault(risc, []).append(int(cyc))
    out = {}
    for risc, s in starts.items():
        s.sort()
        e = sorted(ends.get(risc, []))
        durs = [(ee - ss) * ns_per_cycle for ss, ee in zip(s, e)]
        if durs:
            out[_RISC_LABEL.get(risc, risc)] = statistics.median(durs)
    return out


def _measure_zone(device, run_fn, reps, trials):
    """Per-call ns for each engine, one sample per measured launch. Returns {engine: [ns/call,...]}."""
    run_fn()  # warmup so its markers land under an id we then mark 'seen'
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    samples = {}
    for _ in range(trials):
        seen = _all_run_ids(_DEVICE_CSV)
        run_fn()
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
        for engine, ns in _zone_engines_for_new_run(_DEVICE_CSV, seen).items():
            samples.setdefault(engine, []).append(ns / reps)
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


def _clock_mhz():
    if os.path.exists(_DEVICE_CSV):
        try:
            _, _, freq = _read_csv_rows(_DEVICE_CSV)
            return round(freq)
        except Exception:
            pass
    return None


def _selected(env_name, allowed):
    sel = os.environ.get(env_name)
    chosen = tuple(sel.split(",")) if sel else allowed
    unknown = set(chosen) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {env_name}: {sorted(unknown)}; valid: {allowed}")
    return tuple(v for v in allowed if v in chosen)


# =============================================================================
# Tests
# =============================================================================
def test_sfpu_tile_scope_correctness(device):
    variants = _selected("STS_VARIANTS", VARIANTS)
    funcs = _selected("STS_FUNCS", FUNCS)
    x_dev, golden_in = _make_input(device)
    for func in funcs:
        for variant in variants:
            out = run_op(x_dev, variant=variant, func=func, reps=1)
            rel = _check(out, golden_in, func, variant, f"{func}/{variant}")
            logger.info(f"{func:5s} {variant:11s}  rel_err={rel:.5f}")


def test_sfpu_tile_scope_device_perf(device):
    variants = _selected("STS_VARIANTS", VARIANTS)
    funcs = _selected("STS_FUNCS", FUNCS)
    reps = _int("STS_REPS", "2000")
    trials = _int("STS_TRIALS", "5")

    x_dev, golden_in = _make_input(device)

    # correctness gate at reps=1 for every measured cell
    for func in funcs:
        for variant in variants:
            out = run_op(x_dev, variant=variant, func=func, reps=1)
            _check(out, golden_in, func, variant, f"{func}/{variant}")

    # perf: isolated MATH-thread ns per SFPU call
    perf = {}  # (func, variant) -> {engine: [ns/call per trial]}
    for func in funcs:
        for variant in variants:
            perf[(func, variant)] = _measure_zone(
                device, lambda f=func, v=variant: run_op(x_dev, variant=v, func=f, reps=reps), reps, trials
            )

    report = _format_report(
        perf,
        funcs,
        variants,
        box=socket.gethostname(),
        arch=_arch_label(device),
        clock=_clock_mhz(),
        reps=reps,
        trials=trials,
    )
    logger.info("\n" + report)
    if report_path := os.environ.get("STS_REPORT"):
        Path(report_path).write_text(report)


# =============================================================================
# Report
# =============================================================================
def _format_report(perf, funcs, variants, *, box, arch, clock, reps, trials):
    def med(func, variant, engine):
        vals = perf[(func, variant)].get(engine, [])
        return statistics.median(vals) if vals else 0.0

    def std_pct(func, variant, engine):
        vals = perf[(func, variant)].get(engine, [])
        m = statistics.median(vals) if vals else 0.0
        return (statistics.pstdev(vals) / m * 100) if (len(vals) > 1 and m) else 0.0

    clock_str = f"{clock}MHz" if clock else "n/a"
    lines = [
        "# SFPU work-scoping — isolated MATH-thread cost, whole tile vs a scoped face subset",
        "",
        f"box={box}  arch={arch}  clock={clock_str}  cores=1  placement=single-core sharded-L1  "
        f"N={trials} (median)  reps={reps} (in-kernel math loop)",
        "metric: MATH-thread (TRISC_1) ns per SFPU call, from a DeviceZoneScopedN around a math-only loop.",
        "copy(seed) and pack are OUTSIDE the zone, so the number is pure SFPU math cycles — no unpack, no",
        "pack, no CB handshake, no per-tile copy/pack floor. Input bf16, one Tensix core, sharded L1.",
        "",
        "scopes (32-lane vector ops in []): rc (RC, whole tile [32]; BASELINE) | r (R, top half [16]) | "
        "c (C, left half [16]) | r_iter2 (R + ITERATIONS=2, ROW 0 [4]) | c_skip (C even-parity stride, COL 0 "
        "[8]) | face (None, face 0 [8]) | face_iter1 (None + IT=1, [0,0] [1]). none = empty loop (overhead).",
        "",
    ]

    for func in funcs:
        lines += [
            f"## func = {func} — MATH ns per SFPU call; (speedup vs rc); ns per vector op",
            "",
            "| scope | how | vec ops | math ns/call | speedup vs rc | ns / vector |",
            "|---|---|---|---|---|---|",
        ]
        for variant in variants:
            m = med(func, variant, "math")
            n_vec = vectors(variant)
            if variant == BASELINE:
                spd = "1.00x"
            elif variant == ABLATION:
                spd = "—"
            else:
                base = med(func, BASELINE, "math")
                spd = f"{base / m:.2f}x" if m else ""
            per_vec = f"{m / n_vec:.1f}" if n_vec else "—"
            lines.append(
                f"| {variant} | {LABEL[variant]} | {n_vec or '—'} | {m:.1f}±{std_pct(func, variant, 'math'):.0f}% | {spd} | {per_vec} |"
            )
        # proof of isolation: unpack/pack inside the zone
        up = max(med(func, v, "unpack") for v in variants)
        pk = max(med(func, v, "pack") for v in variants)
        lines += [
            "",
            f"isolation check: max unpack={up:.3f} ns/call, max pack={pk:.3f} ns/call inside the zone "
            f"(≈0 → the SFPU is alone on the math thread).",
            "",
        ]

    lines += [
        "Notes: an SFPU vector op = 4 rows x 8 stride-2 columns; a 32x32 tile is 32 vector ops (4 faces x 4 "
        "row-groups x 2 column parities). The MATH cost is ~flat per vector op (see ns/vector), so the ladder is "
        "just how many vector ops the scope runs: rc=32 -> r/c=16 -> c_skip/face=8 -> r_iter2=4 -> face_iter1=1. "
        "Two axis-optimal tricks vs the coarse half-tile modes: for a ROW-0 result the row waste is the OUTER "
        "walk axis, so ITERATIONS truncates it — r_iter2 = VectorMode::R + ITERATIONS=2 keeps the top row-group "
        "of both top faces = 4 vectors (a pure knob turn). For a COL-0 result the waste is column PARITY, the "
        "INNER walk axis (vectors alternate even/odd), so ITERATIONS can't isolate it — c_skip strides the DEST "
        "address by 2 (raw sfpi) to keep only the even-parity vectors (which hold column 0) of the two left "
        "faces = 8 vectors. That is why r_iter2 (4) beats c_skip (8), and why the row trick is a knob but the "
        "column trick needs raw sfpi. This is the SFPU cost in ISOLATION; in a full op the copy/pack/DRAM around "
        "it dilute the win, and a data-movement-bound op won't show it. recip costs more per vector than rsqrt "
        "(heavier op); its c_skip uses the Newton reciprocal body (the stock recip fast path uses SFPLOADMACRO "
        "addressing that can't be strided the same way), so its per-vector cost differs slightly from stock recip.",
    ]
    return "\n".join(lines) + "\n"
