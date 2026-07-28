# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pytest DRIVER for the `gather_payload_shrink` perf experiment.

All experiment code (kernels, program descriptor, bench) lives in
``tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/gather_payload_shrink/``.
Only this thin driver lives here, because ``pytest.ini`` uses
``--import-mode=importlib``: a test module under ``ttnn/ttnn/...`` would be
imported as ``ttnn.ttnn....`` and re-register every C++ op at collection time.

    scripts/run_safe_pytest.sh           tests/.../test_gather_payload_shrink.py -k probe
    scripts/run_safe_pytest.sh --profile tests/.../test_gather_payload_shrink.py -k "bench and focus"
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

_EXP = (
    Path(__file__).resolve().parents[5]
    / "tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/gather_payload_shrink"
)


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_gps_{name}", _EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_probe_reduce_scaler_position(device):
    _load("probe_mechanism").run_probe(device)


# --------------------------------------------------------------------------
# The A/B bench. VARIANT and the geometry come from env so one pytest node is
# one fresh-cache dispatch of one variant (no trial loop).
# --------------------------------------------------------------------------

_BENCH_CASES = [
    "focus",  # (1,1,8192,1024) BLOCK  shard (1024,128) grid (8,8)  ht_block 8
    "focus_hb4",
    "focus_hb2",
    "focus_hb1",
    "w32x1024",  # (1,1,32,1024) WIDTH shard (32,128) grid (8,1)  ht 1
    "w32x7168",  # (1,1,32,7168) WIDTH shard (32,256) grid (7,4)  ht 1
    "block8192x2304",
]


@pytest.mark.parametrize("case", _BENCH_CASES)
def test_bench(device, case):
    """ONE fresh dispatch. This is the TIMED node (--profile, one CSV row)."""
    variant = os.environ.get("GPS_VARIANT", "baseline")
    _load("bench").run_case(device, case, variant, mode="random")


@pytest.mark.parametrize("case", _BENCH_CASES)
def test_absolute(device, case):
    """The mandatory ABSOLUTE all-ones element-count gate (not a timing node)."""
    variant = os.environ.get("GPS_VARIANT", "baseline")
    _load("bench").run_case(device, case, variant, mode="ones")
