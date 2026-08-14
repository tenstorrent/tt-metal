# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device bake-off for COALESCING rms_norm's gather writes (see bench.py).

Correctness is the ONLY pass/fail.  Perf is measured in-process off the device
profiler (`DEVICE KERNEL DURATION [ns]`) and never asserted.

    scripts/run_safe_pytest.sh --run-all \
      tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/gather_coalesce/test_gather_coalesce.py

    GC_CASES=focus GC_VARIANTS=baseline,coalesce scripts/run_safe_pytest.sh --run-all <this file>

One fresh run per (case, variant): device kernel time has no warm-up transient.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest
import torch

import ttnn
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import VARIANTS, Geo, check, geometry, make_tensors, plan, run_variant  # noqa: E402

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# S=4 throughout (the perf-flagged BLOCK-shard profile's slice width) and
# shard_rows = nb*B = 32 (its 1024-row shard), so every case is the same total work
# and only the combine geometry moves.
#
# `focus` IS the op's plan for (1,1,8192,1024) BLOCK [1024,128] on (8,8):
# s=8, S=4, B=16, 2 blocks, num_owners=8, own_rows=2.
CASES = {
    "focus": Geo(s=8, S=4, B=16, nb=2, gw=8, gh=8),  # owners=8  own_rows=2
    "s2_B16": Geo(s=2, S=4, B=16, nb=2, gw=8, gh=8),  # owners=2  own_rows=8
    "s4_B16": Geo(s=4, S=4, B=16, nb=2, gw=8, gh=8),  # owners=4  own_rows=4
    "s16_B16": Geo(s=16, S=4, B=16, nb=2, gw=8, gh=8),  # owners=16 own_rows=1
    "s28_B16": Geo(s=28, S=4, B=16, nb=2, gw=7, gh=8),  # owners=16 own_rows=1
    "s32_B16": Geo(s=32, S=4, B=16, nb=2, gw=8, gh=8),  # owners=16 own_rows=1
    "s8_B1": Geo(s=8, S=4, B=1, nb=32, gw=8, gh=8),  # owners=1  own_rows=1
    "s8_B8": Geo(s=8, S=4, B=8, nb=4, gw=8, gh=8),  # owners=8  own_rows=1
    "s8_B32": Geo(s=8, S=4, B=32, nb=1, gw=8, gh=8),  # owners=8  own_rows=4
    "s4_B32": Geo(s=4, S=4, B=32, nb=1, gw=8, gh=8),  # owners=4  own_rows=8
}

pytestmark = pytest.mark.use_module_device

DEFAULT_CASES = tuple(CASES)
DEFAULT_VARIANTS = ("baseline", "baseline_raw", "coalesce")


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


@pytest.mark.parametrize("case_name", _selected("GC_CASES", DEFAULT_CASES, CASES))
def test_gather_coalesce(device, case_name):
    geo = CASES[case_name]
    variants = _selected("GC_VARIANTS", DEFAULT_VARIANTS, VARIANTS)

    p = plan(device, geo)
    x, out, expected = make_tensors(device, p)

    rows = []
    failures = []
    base_stats = None
    try:
        for variant in variants:
            try:
                result = run_variant(device, p, x, out, variant=variant)
            except RuntimeError as exc:
                # An INFEASIBLE variant is data too (the flat root's gather CB is
                # s*B stat pages, which does not always fit L1).
                msg = " | ".join(line.strip() for line in str(exc).splitlines() if line.strip())
                rows.append((variant, None, float("nan"), float("nan"), float("nan"), f"INFEASIBLE: {msg}"))
                continue
            ttnn.synchronize_device(device)
            ns = _read_kernel_ns(device)
            pcc, worst_rel, stats = check(p, result, expected)
            if variant == "baseline":
                base_stats = stats
            # The candidate claims BIT-IDENTICAL arithmetic (same partials, same
            # pairwise order, relabelled indices).  Measure that, do not assume it.
            drift = float("nan") if base_stats is None else (stats - base_stats).abs().max().item()
            rows.append((variant, ns, pcc, worst_rel, drift, ""))
            # Correctness gate.  RELATIVE ERROR is the gate; PCC is a coarse guard.
            # The stat is a sum of W squares accumulated in a bf16 DEST
            # (fp32_dest_acc_en=False is the USER's config, not a lever), so ~0.5-0.9%
            # against an fp32 torch reference is the arithmetic's own floor, and PCC
            # degrades with W for a reason that is not accuracy (1/rms of W standard
            # normals concentrates on 1.0 like 1/sqrt(W), so the reference's own
            # variance shrinks while the bf16 error does not).  A topology bug — a
            # missed row, a stale page, a lost partial — misses by orders of magnitude.
            if not (pcc > 0.98 and worst_rel < 0.02):
                failures.append(f"{case_name}/{variant}: pcc={pcc} worst_rel={worst_rel}")
    finally:
        x.deallocate()
        out.deallocate()

    num_owners, own_rows = geometry(geo, False)
    base = next((ns for name, ns, *_ in rows if name == "baseline"), None)
    logger.info(
        f"\n=== {case_name}: {geo.label} | cores={len(p.cores)} groups={len(p.groups)} "
        f"W={geo.width} owners={num_owners} own_rows={own_rows} "
        f"| gather txns/block/core: rowmajor={geo.B} coalesced={num_owners} ==="
    )
    for variant, ns, pcc, worst_rel, drift, note in rows:
        if ns is None:
            logger.info(f"GC {case_name:9s} {variant:20s}        n/a       -  {note}")
            continue
        speedup = f"{base / ns:.3f}x" if (base and ns) else "-"
        logger.info(
            f"GC {case_name:9s} {variant:20s} {ns:10.1f} ns  {speedup:>7s}  "
            f"pcc={pcc:.6f} rel={worst_rel:.2e} drift_vs_baseline={drift:.3e}"
        )
    assert not failures, "; ".join(failures)
