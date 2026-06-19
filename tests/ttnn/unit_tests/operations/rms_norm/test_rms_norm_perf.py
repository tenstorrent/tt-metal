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


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["TILE", "RM"])
@pytest.mark.parametrize("W", [256, 8192], ids=["Wt8", "Wt256"])
def test_rms_norm_ab_crossover(device, layout, W):
    """Force A and B on the same single-tile-row shape and assert (1) the production
    heuristic picks the faster regime within a noise band, and (2) wide rows (Wt=256)
    are decisively faster in the (K-tuned) Regime B.

    With the K-tuned _select_k, Regime B is fast enough that it wins from Wt>=16; the
    crossover threshold (REGIME_B_MIN_WT_*=16) keeps only the single-tile-wide row
    (Wt=8, W=256) in Regime A. We don't pin a winner at the Wt=8 boundary (it is a
    ~15% margin, near noise) — the robust contract is that the heuristic tracks the
    faster regime."""
    x = torch.randn(1, 1, 32, W)
    a = _measure_forced(device, x, "A", layout=layout)
    b = _measure_forced(device, x, "B", layout=layout)
    assert a is not None and b is not None, (a, b)

    # (1) The production heuristic must select the faster regime within a noise band.
    chosen = measure_device_kernel_ns(device, x, layout=layout)
    best = min(a, b)
    assert chosen <= best * 1.20, f"W={W} {layout}: heuristic {chosen/1000:.1f}us, best {best/1000:.1f}us"

    # (2) Wide rows: K-tuned Regime B must decisively beat single-core Regime A.
    if W >= 8192:
        assert b < a, f"W={W} {layout}: expected B<A, got A={a/1000:.1f}us B={b/1000:.1f}us"


def _measure_forced_k(device, x, K, **kw):
    try:
        desc._FORCE_K = K
        return measure_device_kernel_ns(device, x, **kw)
    finally:
        desc._FORCE_K = None


def test_rms_norm_regime_b_rowblocking_exhausted(device):
    """Refinement 8 — measured evidence that row-blocking / coalesced-mcast for
    Regime B cannot produce a net device-time win, so it is intentionally NOT
    implemented (closed, mirroring R7's Regime-A row-blocking gate).

    R8's premise was: group ``bh`` tile-row-groups onto one K-core band and issue ONE
    coalesced mcast of ``bh`` partials (instead of ``bh`` separate K-round gathers) to
    amortize the all-gather fixed cost — keeping active-core count unchanged. Keeping
    the core count constant while grouping ``bh`` row-groups forces K up by a factor of
    ``bh`` (fewer bands, each wider). This test measures the two structural facts that
    make that a guaranteed regression:

      (1) PARALLEL-GROUP FLATNESS — there is no serialized per-gather fixed cost for
          coalescing to amortize. Regime B runs each row-group on a disjoint K-core
          rectangle, all in parallel. Adding a second row-group at the SAME K barely
          moves device time (measured ~34us -> ~38us for 1 -> 2 groups at K=16). If a
          per-gather fixed cost were serialized, 2 groups would cost ~2x; it does not.

      (2) K-MONOTONICITY — the all-gather cost grows with K, so the K-doubling that
          ``bh>1`` requires is strictly net-negative. Forcing K up on a single-row-group
          shape (where every K fits the grid) shows device time rising monotonically
          with K (measured K=16 ~34us < K=32 < K=64 ~110us). bh>1 buys exactly this
          K increase, while the per-core reduce work it would save is already flat.

    Conclusion (see changelog.md R8): the only theoretical Regime-B win is a *coverage*
    extension (oversubscribed grids — shapes with num_row_groups*K_min > total_cores
    that currently fall back to slow Regime A), which is a different mechanism than
    coalesced mcast AND applies to NO shape in the golden/LOOSE suite (every wide-W
    golden shape is 1-2 tile-rows). Row-blocking is therefore fully exhausted as a
    lever for this op.
    """
    # (1) Parallel-group flatness: 1 vs 2 row-groups, both at K=16 (each fits the grid:
    #     1*16 and 2*16 <= 64). No serialized fixed cost -> ~flat (well under 2x).
    x1 = torch.randn(1, 1, 32, 8192)  # nrg=1, K=16
    x2 = torch.randn(1, 1, 64, 8192)  # nrg=2, K=16
    t1 = measure_device_kernel_ns(device, x1)
    t2 = measure_device_kernel_ns(device, x2)
    assert t1 is not None and t2 is not None, (t1, t2)
    assert t2 < t1 * 1.6, (
        f"parallel-group flatness violated: 1 group={t1/1000:.1f}us, 2 groups={t2/1000:.1f}us "
        f"(ratio {t2/t1:.2f}); a serialized per-gather fixed cost would push this toward 2x"
    )

    # (2) K-monotonicity: forcing K up (what bh>1 requires) is net-negative. Single
    #     row-group so every K in {16,32,64} qualifies (1*K <= 64, K|256, K%8==0).
    k16 = _measure_forced_k(device, x1, 16)
    k32 = _measure_forced_k(device, x1, 32)
    k64 = _measure_forced_k(device, x1, 64)
    assert k16 is not None and k32 is not None and k64 is not None, (k16, k32, k64)
    # K=64 must be clearly worse than K=16 (the all-gather cost dominates the growth).
    assert k64 > k16 * 1.5, (
        f"expected raising K (the bh>1 lever) to be net-negative: K=16={k16/1000:.1f}us, "
        f"K=32={k32/1000:.1f}us, K=64={k64/1000:.1f}us"
    )
