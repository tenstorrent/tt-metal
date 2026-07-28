# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pytest DRIVER for the `combine_latency_hiding` perf experiment (Perf 2
tournament — idea: fill the combine's dead time with the next row-block's
independent pass A).

All experiment code (kernels, program descriptor, bench) lives in
``tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/combine_latency_hiding/``.
Only this thin driver lives here, because ``pytest.ini`` uses
``--import-mode=importlib``: a test module under ``ttnn/ttnn/...`` would be
imported as ``ttnn.ttnn....`` and re-register every C++ op at collection time.

    scripts/run_safe_pytest.sh --profile --run-all tests/.../test_combine_latency_hiding.py -k "bench and focus"
    scripts/run_safe_pytest.sh --profile --run-all tests/.../test_combine_latency_hiding.py -k absolute
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

_EXP = (
    Path(__file__).resolve().parents[5]
    / "tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/combine_latency_hiding"
)


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_clh_{name}", _EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# The A/B bench. VARIANT and the geometry come from env / parametrize so one
# pytest node is one fresh-cache dispatch of one variant (no trial loop).
# --------------------------------------------------------------------------

_BENCH_CASES = [
    "focus",  # (1,1,8192,1024) BLOCK  shard (1024,128) grid (8,8)  nh_core=4
    "w32x1024",  # (1,1,32,1024) WIDTH  nh_core=1 (structurally inert)
    "w32x2304",
    "w32x5120",
    "w32x7168",
    "i32x5120",  # interleaved W-split decode reps
    "i32x7168",
    "i8192x1024",  # interleaved prefill reps: cw==1, no combine, DRAM-bound
    "i8192x7168",
]

_VARIANTS = ["baseline", "prefetch_a", "defer_passb"]


@pytest.mark.parametrize("case", _BENCH_CASES)
@pytest.mark.parametrize("variant", _VARIANTS)
def test_bench(device, variant, case):
    """ONE fresh dispatch. This is the TIMED node (--profile, one CSV row)."""
    _load("bench").run_case(device, case, variant, mode="random")


@pytest.mark.parametrize("case", _BENCH_CASES)
@pytest.mark.parametrize("variant", _VARIANTS)
def test_absolute(device, variant, case):
    """The mandatory ABSOLUTE all-ones element-count gate (not a timing node)."""
    _load("bench").run_case(device, case, variant, mode="ones")


@pytest.mark.parametrize("stall_wait", ["6", "4", "2", "1"])
@pytest.mark.parametrize("variant", ["baseline", "prefetch_a"])
def test_bench_stall_sensitivity(device, variant, stall_wait):
    """Sensitivity study: shrink the combine round trip (CLH_STALL_WAIT
    override, output WRONG by design) and re-measure baseline vs prefetch_a on
    the focus shape, to see how much of the candidate's win survives if a
    sibling idea shrinks the stall itself."""
    os.environ["CLH_STALL_WAIT"] = stall_wait
    try:
        _load("bench").run_case(device, "focus", variant, mode="random")
    finally:
        os.environ.pop("CLH_STALL_WAIT", None)


def test_report_blocking(device):
    """Prints the derived block factors (HT_BLOCK, nh_core, clh_eligible,
    cb_partial_out depth, L1 totals) for every case — not a timing node."""
    os.environ["CLH_REPORT"] = "1"
    try:
        for case in _BENCH_CASES:
            _load("bench").run_case(device, case, "prefetch_a", mode="random")
    finally:
        os.environ.pop("CLH_REPORT", None)
