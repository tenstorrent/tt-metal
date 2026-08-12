# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Runner for the I9 pipelined-combine bake-off. Correctness is the only pass/fail.

ONE fresh dispatch per (geometry, skew, variant) — DEVICE KERNEL DURATION has no
warm-up transient, so there is no trial loop.

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
    TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
    scripts/tt-probe.sh rms_norm <<'EOF'
    import sys; sys.path.insert(0, "tests/ttnn/unit_tests/operations/rms_norm/perf_experiments")
    from pipelined_combine.harness import main
    main(["focus_11x10"], ["baseline", "flag", "incr"], skews=["none", "mid", "big"])
    EOF
"""

from __future__ import annotations

import traceback

import ttnn

from .pipe_bench import GEOMS, SKEWS, VARIANTS, run

TOL = {"exact": 0.004, "distinct": 0.02}


def measure(device, geom_name, variant, partial_kind="exact", skew_name="none"):
    ns, max_rel = run(device, geom_name, variant, partial_kind, SKEWS[skew_name])
    tol = TOL[partial_kind]
    ok = max_rel <= tol
    print(
        f"BENCH {geom_name:18s} skew={skew_name:5s} {variant:9s} ns={ns} "
        f"max_rel_err={max_rel:.5f} tol={tol} {'OK' if ok else 'FAIL'}",
        flush=True,
    )
    return ns, max_rel, ok


def main(geom_names=None, variants=None, partial_kind="exact", skews=("none",)):
    geom_names = list(geom_names or GEOMS)
    variants = list(variants or VARIANTS)
    device = ttnn.open_device(device_id=0)
    results = {}
    try:
        for g in geom_names:
            for s in skews:
                for v in variants:
                    try:
                        results[(g, s, v)] = measure(device, g, v, partial_kind, s)
                    except Exception as exc:  # an inexpressible cell is DATA, not a crash
                        print(f"BENCH {g:18s} skew={s:5s} {v:9s} SKIP/ERROR {type(exc).__name__}: {exc}", flush=True)
                        traceback.print_exc()
    finally:
        ttnn.close_device(device)
    print("BENCH ---- summary ----", flush=True)
    for g in geom_names:
        for s in skews:
            base = results.get((g, s, "baseline"))
            for v in variants:
                r = results.get((g, s, v))
                if r is None or r[0] is None:
                    continue
                speedup = f"{base[0] / r[0]:.3f}x" if (base and base[0]) else "-"
                print(f"BENCH_SUMMARY {g:18s} skew={s:5s} {v:9s} ns={r[0]:.0f} vs_base={speedup} ok={r[2]}", flush=True)
    return results
