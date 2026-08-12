# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Runner for the combine bake-off. Correctness is the only pass/fail; perf is data.

ONE fresh dispatch per (geometry, variant) — DEVICE KERNEL DURATION has no warm-up
transient, so there is no trial loop.

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
    TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
    scripts/tt-probe.sh rms_norm <<'EOF'
    from ttnn.operations.rms_norm.perf_experiments.allgather_combine.harness import main
    main(["focus_11x10"], ["baseline", "allgather"])
    EOF
"""

from __future__ import annotations

import os
import shutil
import traceback

import ttnn

from .combine_bench import GEOMS, VARIANTS, run

# `exact` values are powers of two with equal addends, so the reference is
# order-independent and bf16-exact; the only error left is the bf16 rounding of the
# rsqrt OUTPUT (~2^-9). A dropped contribution moves the result by 1/(2G).
TOL = {"exact": 0.004, "distinct": 0.02}


def measure(device, geom_name, variant, partial_kind="exact", keep_zones=False):
    ns, max_rel, got, exp = run(device, geom_name, variant, partial_kind)
    tol = TOL[partial_kind]
    ok = max_rel <= tol
    print(
        f"BENCH {geom_name:22s} {variant:15s} ns={ns} max_rel_err={max_rel:.5f} " f"tol={tol} {'OK' if ok else 'FAIL'}",
        flush=True,
    )
    if keep_zones:
        logdir = os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated", "profiler", ".logs")
        src = os.path.join(logdir, "profile_log_device.csv")
        if os.path.exists(src):
            dst = os.path.join(logdir, f"agzones_{geom_name}_{variant}.csv")
            shutil.copyfile(src, dst)
            print(f"BENCH zones -> {dst}", flush=True)
    return ns, max_rel, ok


def main(geom_names=None, variants=None, partial_kind="exact", keep_zones=False):
    geom_names = list(geom_names or GEOMS)
    variants = list(variants or VARIANTS)
    device = ttnn.open_device(device_id=0)
    results = {}
    try:
        for g in geom_names:
            for v in variants:
                try:
                    results[(g, v)] = measure(device, g, v, partial_kind, keep_zones)
                except Exception as exc:  # a variant that cannot be expressed here is DATA, not a crash
                    print(f"BENCH {g:22s} {v:15s} SKIP/ERROR {type(exc).__name__}: {exc}", flush=True)
                    traceback.print_exc()
    finally:
        ttnn.close_device(device)
    print("BENCH ---- summary ----", flush=True)
    for g in geom_names:
        base = results.get((g, "baseline"))
        for v in variants:
            r = results.get((g, v))
            if r is None:
                continue
            speedup = f"{base[0] / r[0]:.3f}x" if (base and r[0]) else "-"
            print(f"BENCH_SUMMARY {g:22s} {v:15s} ns={r[0]:.0f} vs_baseline={speedup} ok={r[2]}", flush=True)
    return results
