# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 6 — performance measurement + A/B crossover validation for rms_norm.

This file is the committed, reproducible companion to the Tracy / device-profiler
measurement flow documented in changelog.md (Refinement 6). It does two things:

  1. Sweeps representative shapes across both regimes x {bf16, fp32} x {+/- gamma}
     x {TILE, ROW_MAJOR}, captures per-op on-device kernel time (ns), and writes a
     baseline table to ``perf_results.md`` next to this file.

  2. Validates the *measured* A-vs-B crossover encoded in the heuristic
     (``REGIME_B_MIN_WT``): it force-runs both regimes on the same shape via the
     ``_FORCE_REGIME`` measurement hook and asserts (a) Regime A wins decisively for
     narrow rows, (b) Regime B wins decisively for wide rows, and (c) the production
     heuristic actually selects the faster regime (within a noise band).

Measurement requires a profiler-enabled build and the runtime env vars set *before*
device init (the whole module skips otherwise — so a plain ``run_safe_pytest.sh``
without the env vars is a no-op skip, not a failure):

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
    TT_METAL_PROFILER_CPP_POST_PROCESS=1 TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1 \
        scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_perf.py

The on-device timing API used here is exactly what the eval harness uses
(``eval/profiling.py``): ``ttnn.ReadDeviceProfiler(device)`` finishes the command
queue and then ``ttnn.get_latest_programs_perf_data()`` exposes per-program
``"DEVICE KERNEL DURATION [ns]"``.
"""

import os
from pathlib import Path

import pytest
import torch

import ttnn
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as desc

# Whole module is measurement-only: skip unless the profiler is on.
pytestmark = pytest.mark.skipif(
    not os.environ.get("TT_METAL_DEVICE_PROFILER"),
    reason="perf test needs TT_METAL_DEVICE_PROFILER=1 (+ MID_RUN_DUMP / CPP_POST_PROCESS)",
)

_PROFILER_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
_RESULTS_MD = Path(__file__).parent / "perf_results.md"


def _read_device_kernel_ns(device):
    """Sum DEVICE KERNEL DURATION [ns] over all programs dispatched since last read."""
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total = 0.0
    found = False
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_PROFILER_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def measure_device_kernel_ns(device, x, gamma=None, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, iters=7):
    """Best-of-`iters` per-op device kernel ns for rms_norm(x[, gamma]).

    One warmup pass (JIT + first-launch overhead), then `iters` timed passes; the
    min is the cleanest estimate of steady-state device time.
    """
    ti = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device)
    g = None
    if gamma is not None:
        g = ttnn.from_torch(gamma, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    rms_norm(ti, gamma=g)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)  # flush warmup
    samples = []
    for _ in range(iters):
        rms_norm(ti, gamma=g)
        ttnn.synchronize_device(device)
        ns = _read_device_kernel_ns(device)
        if ns is not None:
            samples.append(ns)
    return min(samples) if samples else None


# Representative shapes: Regime A (many tile-rows, narrow/moderate W) and
# Regime B (few rows, wide W). (label, shape).
_REGIME_A_SHAPES = [
    ("A: (1,1,2048,256)", (1, 1, 2048, 256)),
    ("A: (4,1,512,512)", (4, 1, 512, 512)),
    ("A: (1024,1024)", (1024, 1024)),
]
_REGIME_B_SHAPES = [
    ("B: (1,1,32,4096)", (1, 1, 32, 4096)),
    ("B: (1,1,32,8192)", (1, 1, 32, 8192)),
    ("B: (1,1,32,16384)", (1, 1, 32, 16384)),
]


def _fmt(ns):
    return "-" if ns is None else f"{ns/1000.0:8.2f}"


def test_rms_norm_perf_baseline_table(device):
    """Sweep representative shapes and write the baseline device-time table."""
    rows = []
    for label, shape in _REGIME_A_SHAPES + _REGIME_B_SHAPES:
        x = torch.randn(*shape)
        gamma = torch.randn(1, 1, 1, shape[-1])
        for dtype, dtag in [(ttnn.bfloat16, "bf16"), (ttnn.float32, "fp32")]:
            for layout, ltag in [(ttnn.TILE_LAYOUT, "TILE"), (ttnn.ROW_MAJOR_LAYOUT, "RM")]:
                no_g = measure_device_kernel_ns(device, x, None, dtype, layout)
                with_g = measure_device_kernel_ns(device, x, gamma, dtype, layout)
                rows.append((label, dtag, ltag, no_g, with_g))

    lines = [
        "# rms_norm device-kernel-time baseline (Refinement 6)",
        "",
        "Per-op on-device kernel time in microseconds (best of 7), 8x8 Wormhole grid.",
        "Measured via ttnn.ReadDeviceProfiler + get_latest_programs_perf_data",
        "(DEVICE KERNEL DURATION [ns]). Heuristic regime: "
        f"REGIME_B_MIN_WT_TILE={desc.REGIME_B_MIN_WT_TILE}, REGIME_B_MIN_WT_RM={desc.REGIME_B_MIN_WT_RM}.",
        "",
        "| shape | dtype | layout | no_gamma (us) | gamma (us) |",
        "|-------|-------|--------|---------------|------------|",
    ]
    for label, dtag, ltag, no_g, with_g in rows:
        lines.append(f"| {label} | {dtag} | {ltag} | {_fmt(no_g)} | {_fmt(with_g)} |")
    _RESULTS_MD.write_text("\n".join(lines) + "\n")

    # Every representative case must have produced a measurement.
    assert all(no_g is not None and with_g is not None for _, _, _, no_g, with_g in rows), rows


def _measure_forced(device, x, regime, **kw):
    try:
        desc._FORCE_REGIME = regime
        return measure_device_kernel_ns(device, x, **kw)
    finally:
        desc._FORCE_REGIME = None


@pytest.mark.parametrize(
    "layout, W, expect_winner",
    [
        # TILE crossover ~160: A decisive at Wt=64, B decisive at Wt=256.
        (ttnn.TILE_LAYOUT, 2048, "A"),  # Wt=64
        (ttnn.TILE_LAYOUT, 8192, "B"),  # Wt=256
        # ROW_MAJOR crossover ~96 (lower: RM Regime A tilizes the whole row on one
        # core): A at Wt=64, B already winning at Wt=128.
        (ttnn.ROW_MAJOR_LAYOUT, 2048, "A"),  # Wt=64
        (ttnn.ROW_MAJOR_LAYOUT, 4096, "B"),  # Wt=128
    ],
    ids=["TILE-Wt64", "TILE-Wt256", "RM-Wt64", "RM-Wt128"],
)
def test_rms_norm_ab_crossover(device, layout, W, expect_winner):
    """Force A and B on the same single-tile-row shape; the measured layout-aware
    crossover (REGIME_B_MIN_WT_TILE / _RM) must agree with which regime is faster."""
    x = torch.randn(1, 1, 32, W)
    a = _measure_forced(device, x, "A", layout=layout)
    b = _measure_forced(device, x, "B", layout=layout)
    assert a is not None and b is not None, (a, b)
    winner = "A" if a < b else "B"
    assert winner == expect_winner, f"W={W} {layout}: A={a/1000:.1f}us B={b/1000:.1f}us, expected {expect_winner}"

    # The production heuristic must select the faster regime within a noise band.
    chosen = measure_device_kernel_ns(device, x, layout=layout)
    best = min(a, b)
    assert chosen <= best * 1.20, f"W={W} {layout}: heuristic {chosen/1000:.1f}us, best {best/1000:.1f}us"
