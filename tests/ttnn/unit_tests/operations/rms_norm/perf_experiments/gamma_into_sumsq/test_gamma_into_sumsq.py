# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device bake-off: WHERE does rms_norm's apply_gamma pass belong? (see bench.py)

Correctness is the ONLY pass/fail.  Perf is measured off the device profiler
(`DEVICE KERNEL DURATION [ns]`) and never asserted.

    scripts/run_safe_pytest.sh --run-all \
      tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/gamma_into_sumsq/test_gamma_into_sumsq.py

    GIS_CASES=focus GIS_VARIANTS=baseline,gamma_first scripts/run_safe_pytest.sh --run-all <this file>

One fresh run per (case, variant): device kernel time has no warm-up transient.
"""

from __future__ import annotations

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest
import torch

import ttnn
from loguru import logger

from tests.ttnn.unit_tests.operations.rms_norm.perf_experiments.gamma_into_sumsq.bench import (
    VARIANTS,
    Geo,
    check,
    make_input,
    make_tensors,
    plan,
    run_variant,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# `focus` IS the op's plan for (1,1,8192,1024) BLOCK-sharded [1024,128] on (8,8):
# s=8, S=4, B=16, 2 blocks, 32 shard tile-rows, 8 row-groups of 8 cores.
CASES = {
    "focus": Geo(s=8, S=4, B=16, nb=2, gw=8, gh=8),
    "focus_nogamma": Geo(s=8, S=4, B=16, nb=2, gw=8, gh=8, has_gamma=False),
    "S8": Geo(s=8, S=8, B=16, nb=2, gw=8, gh=8),
    # S=16 halves the block rows: a 32-tile-row x 16-tile shard is 1 MB in + 1 MB
    # out, which does not fit L1.  Same per-core tile budget as S8.
    "S16": Geo(s=4, S=16, B=8, nb=2, gw=8, gh=8),
    "B1": Geo(s=8, S=4, B=1, nb=8, gw=8, gh=8),
    "B8": Geo(s=8, S=4, B=8, nb=4, gw=8, gh=8),
    "B32": Geo(s=8, S=4, B=32, nb=1, gw=8, gh=8),
    "s1": Geo(s=1, S=4, B=16, nb=2, gw=8, gh=8),
    # Controls for the s==1 exception: same geometries with gamma resident in L1,
    # which takes gamma's DRAM round trip out of the comparison.
    "s1_gl1": Geo(s=1, S=4, B=16, nb=2, gw=8, gh=8, gamma_in_l1=True),
    "focus_gl1": Geo(s=8, S=4, B=16, nb=2, gw=8, gh=8, gamma_in_l1=True),
    "s2": Geo(s=2, S=4, B=16, nb=2, gw=8, gh=8),
    "s16": Geo(s=16, S=4, B=16, nb=2, gw=8, gh=8),
}

pytestmark = pytest.mark.use_module_device

DEFAULT_CASES = ("focus",)
DEFAULT_VARIANTS = ("baseline", "gamma_first", "fused")


def _selected(env, default, allowed):
    names = tuple(part for part in os.environ.get(env, ",".join(default)).split(",") if part)
    unknown = set(names) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {env}: {sorted(unknown)}")
    return names


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data() or {}
    total, found = 0.0, False
    for programs in per_chip.values():
        for program in programs:
            analyses = getattr(program, "program_analyses_results", None) or {}
            entry = analyses.get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


@pytest.mark.parametrize("case_name", _selected("GIS_CASES", DEFAULT_CASES, CASES))
def test_gamma_placement_bakeoff(device, case_name):
    geo = CASES[case_name]
    variants = _selected("GIS_VARIANTS", DEFAULT_VARIANTS, VARIANTS)

    p = plan(device, geo)
    x_ref, out, gamma, expected = make_tensors(device, p)
    x_ref.deallocate()

    rows = []
    failures = []
    try:
        for variant in variants:
            # Every variant REWRITES x in place, so each run needs a fresh input.
            x = make_input(device, p)
            try:
                result = run_variant(device, p, x, out, gamma, variant=variant)
            except RuntimeError as exc:
                msg = str(exc)
                rows.append((variant, None, float("nan"), float("nan"), f"INFEASIBLE: {msg[:200]}"))
                x.deallocate()
                # An L1-capacity refusal is data; anything else (a compile error)
                # is a bug in this bench and must not pass silently.
                if "L1" not in msg and "circular buffer" not in msg.lower():
                    failures.append(f"{case_name}/{variant}: {msg[:400]}")
                continue
            ttnn.synchronize_device(device)
            ns = _read_kernel_ns(device)
            pcc, worst_rel = check(p, result, expected)
            rows.append((variant, ns, pcc, worst_rel, ""))
            x.deallocate()
            # Correctness gate — a faster wrong answer is disqualified, not a win.
            # 0.9995 is the focus case's soft PCC gate from feature_spec; the rel
            # bound catches topology bugs (which miss by orders of magnitude).
            if not (pcc > 0.9995 and worst_rel < 0.05):
                failures.append(f"{case_name}/{variant}: pcc={pcc} worst_rel={worst_rel}")
    finally:
        out.deallocate()
        if gamma is not None:
            gamma.deallocate()

    base = next((ns for name, ns, _p, _w, _n in rows if name.startswith("baseline")), None)
    logger.info(f"\n=== {case_name}: {geo.label} | cores={len(p.cores)} groups={len(p.groups)} W={geo.width} ===")
    for variant, ns, pcc, worst_rel, note in rows:
        if ns is None:
            logger.info(f"GIS {case_name:14s} {variant:18s}        n/a       -  {note}")
            continue
        speedup = f"{base / ns:.3f}x" if (base and ns) else "-"
        logger.info(f"GIS {case_name:14s} {variant:18s} {ns:10.1f} ns  {speedup:>8s}  pcc={pcc:.6f} rel={worst_rel:.2e}")
    assert not failures, "; ".join(failures)
