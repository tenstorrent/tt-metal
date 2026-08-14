# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness + MATH-THREAD-ISOLATED device profiling for the rms_norm FINALIZE.

See micro.py's module docstring for what is isolated and why.  The number reported
here is the MATH (TRISC_1) duration of a DeviceZoneScopedN wrapped around a
math-only `reps`-loop, divided by `reps`: pure SFPU cycles for ONE finalize, with
the copy-into-DEST and the pack both outside the zone.  The zone's unpack/pack
durations come back ~0, which is the proof the SFPU is alone on the clock.

Correctness is the only pass/fail; perf is measured, never asserted.

    scripts/run_safe_pytest.sh --run-all \
        tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/finalize_sfpu_scope/test_micro.py
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import socket
import statistics
import struct
from pathlib import Path

import torch
import ttnn
from loguru import logger

from tests.ttnn.unit_tests.operations.rms_norm.perf_experiments.finalize_sfpu_scope.micro import (
    ABLATION,
    BASELINE,
    CANDIDATE,
    LABEL,
    VALID_REGION,
    VARIANTS,
    ZONE_NAME,
    create_sharded_memory_config,
    inits,
    run_op,
    vectors,
)

TILE = 32

# The focus case's parameters: W = 1024 (so 1/W = 2^-10) and the op's default epsilon.
W = 1024
INV_W = 1.0 / W
EPS = 1e-6


def _bits(f):
    return struct.unpack("<I", struct.pack("<f", float(f)))[0]


# rel-err gate vs the fp64 golden.  The SFPU rsqrt lands within ~0.4%; bf16 DEST
# quantization of the intermediate adds ~0.4%.  2% catches wiring/scope bugs.
_REL_TOL = 0.02


def _make_input(device, seed=13):
    """One tile of plausible Sum(x^2) values for W=1024: uniform in [512, 2048]."""
    torch.manual_seed(seed)
    data = 512.0 + 1536.0 * torch.rand(TILE, TILE)
    x_dev = ttnn.from_torch(
        data.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config(),
    )
    return x_dev, data.to(torch.bfloat16).to(torch.float64)


def _golden(x):
    return torch.rsqrt(x * INV_W + EPS)


def _check(output, golden_in, variant):
    """At reps=1: max rel error + PCC over the region this variant leaves valid."""
    out = ttnn.to_torch(output).to(torch.float64)
    if variant == ABLATION:
        expected = golden_in  # identity copy
    elif variant == "cskip_rsqrt_only":
        expected = torch.rsqrt(golden_in)  # diagnostic: rsqrt WITHOUT the 1/W and eps
    else:
        expected = _golden(golden_in)
    def _stats(r0, r1, c0, c1):
        got = out[r0:r1, c0:c1].flatten()
        exp = expected[r0:r1, c0:c1].flatten()
        rel = ((got - exp).abs() / exp.abs().clamp_min(1e-9)).max().item()
        pcc = torch.corrcoef(torch.stack([got, exp]))[0, 1].item()
        return rel, pcc

    rel, pcc = _stats(*VALID_REGION[variant])
    assert rel < _REL_TOL, f"{variant}: rel-err {rel:.5f} >= {_REL_TOL}"
    # COLUMN 0 is the only region the op ever reads and the only region EVERY
    # variant computes, so it is the apples-to-apples precision comparison.
    # (Comparing a whole-tile variant over 1024 lanes against a col-0 variant over
    # 32 would report a max-rel-err difference that is a sample-count artifact.)
    rel0, pcc0 = _stats(0, TILE, 0, 1)
    return rel, pcc, rel0, pcc0


# =============================================================================
# Per-TRISC device-zone timing (DeviceZoneScopedN -> profile_log_device.csv)
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


def _measure_zone(device, run_fn, reps):
    """ONE measured launch per variant (device kernel time has no warm-up transient).

    The first launch only exists so its profiler markers land under an id we can
    then mark 'seen'; the measurement is the second.
    """
    run_fn()
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    seen = _all_run_ids(_DEVICE_CSV)
    run_fn()
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    return {engine: ns / reps for engine, ns in _zone_engines_for_new_run(_DEVICE_CSV, seen).items()}


def _clock_mhz():
    if os.path.exists(_DEVICE_CSV):
        try:
            _, _, freq = _read_csv_rows(_DEVICE_CSV)
            return round(freq)
        except Exception:
            pass
    return None


def _arch_label(device):
    a = str(device.arch()).rsplit(".", 1)[-1]
    return {"WORMHOLE_B0": "WH_B0", "BLACKHOLE": "BH"}.get(a, a)


# =============================================================================
# Tests
# =============================================================================
def test_finalize_scope_micro(device):
    reps = int(os.environ.get("FIN_REPS", "2000"))
    inv_w_bits, eps_bits = _bits(INV_W), _bits(EPS)
    x_dev, golden_in = _make_input(device)

    # --- correctness gate at reps=1 for every measured cell ---
    acc = {}
    for variant in VARIANTS:
        out = run_op(x_dev, variant=variant, inv_w_bits=inv_w_bits, eps_bits=eps_bits, reps=1)
        acc[variant] = _check(out, golden_in, variant)
        logger.info(
            f"{variant:20s} valid-region rel_err={acc[variant][0]:.6f} pcc={acc[variant][1]:.7f} | "
            f"col0 rel_err={acc[variant][2]:.6f} pcc={acc[variant][3]:.7f}"
        )

    # --- perf: isolated MATH-thread ns per finalize, ONE measured launch each ---
    perf = {}
    for variant in VARIANTS:
        perf[variant] = _measure_zone(
            device,
            lambda v=variant: run_op(x_dev, variant=v, inv_w_bits=inv_w_bits, eps_bits=eps_bits, reps=reps),
            reps,
        )

    report = _format(perf, acc, box=socket.gethostname(), arch=_arch_label(device), clock=_clock_mhz(), reps=reps)
    logger.info("\n" + report)
    Path(os.environ.get("FIN_REPORT", "/tmp/finalize_sfpu_scope_micro.md")).write_text(report)


def _format(perf, acc, *, box, arch, clock, reps):
    def math_ns(v):
        return perf[v].get("math", 0.0)

    base = math_ns(BASELINE)
    lines = [
        "# rms_norm FINALIZE — isolated MATH-thread SFPU cost (scope x fusion)",
        "",
        f"box={box}  arch={arch}  clock={clock}MHz  cores=1  single-core sharded-L1  "
        f"reps={reps} (in-kernel math loop)  ONE measured launch per variant",
        "config (FIXED, identical for every variant): math_fidelity=HiFi2, fp32_dest_acc_en=False, "
        "math_approx_mode=False, bf16 in / fp32 out.  W=1024, eps=1e-6.",
        "metric: MATH (TRISC_1) ns for ONE finalize; copy-into-DEST and pack are OUTSIDE the zone.",
        "",
        "| variant | how | SFPU inits | vec ops | math ns/finalize | speedup vs stock | ns/vec | "
        "col-0 max rel err | col-0 PCC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for v in VARIANTS:
        m = math_ns(v)
        nv = vectors(v)
        spd = "1.00x" if v == BASELINE else ("—" if v == ABLATION else (f"{base / m:.2f}x" if m else ""))
        per_vec = f"{m / nv:.1f}" if nv else "—"
        _rel_v, _pcc_v, rel, pcc = acc[v]
        lines.append(
            f"| {v} | {LABEL[v]} | {inits(v)} | {nv or '—'} | {m:.1f} | {spd} | {per_vec} | "
            f"{rel:.5f} | {pcc:.7f} |"
        )
    up = max(perf[v].get("unpack", 0.0) for v in VARIANTS)
    pk = max(perf[v].get("pack", 0.0) for v in VARIANTS)
    lines += [
        "",
        f"isolation check: max unpack={up:.3f} ns, max pack={pk:.3f} ns inside the zone "
        f"(~0 => the SFPU is alone on the math thread).",
        "",
        f"candidate = {CANDIDATE}: {math_ns(CANDIDATE):.1f} ns vs stock {base:.1f} ns "
        f"= {base / math_ns(CANDIDATE):.2f}x" if math_ns(CANDIDATE) else "",
    ]
    return "\n".join(lines) + "\n"
