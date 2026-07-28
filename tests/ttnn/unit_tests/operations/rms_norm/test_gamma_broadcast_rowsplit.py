# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pytest DRIVER for the `gamma_broadcast_rowsplit` perf experiment.

All experiment code (the forked descriptor, the forked kernels, the bench) lives in
``tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/gamma_broadcast_rowsplit/``. Only
this thin driver lives here, because ``pytest.ini`` uses ``--import-mode=importlib``:
a test module under ``ttnn/ttnn/...`` would be imported as ``ttnn.ttnn....`` and
re-execute ``ttnn/__init__.py`` at collection time.

    # what the blocking actually derived, and the broadcast plan (no dispatch)
    scripts/run_safe_pytest.sh tests/.../test_gamma_broadcast_rowsplit.py -k describe

    # correctness gates (random / all-ones / RAMP gamma / bit-exact vs baseline)
    scripts/run_safe_pytest.sh --run-all tests/.../test_gamma_broadcast_rowsplit.py -k correct

    # the TIMED plan: one node = one fresh dispatch = one CSV row, in this order
    GBR_PLAN="prefill7168:baseline,prefill7168:ablate" \
      scripts/run_safe_pytest.sh --profile --run-all tests/.../test_gamma_broadcast_rowsplit.py -k plan
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

_EXP = (
    Path(__file__).resolve().parents[5]
    / "tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/gamma_broadcast_rowsplit"
)


def _bench():
    spec = importlib.util.spec_from_file_location("_gbr_bench", _EXP / "bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _plan():
    """The ordered (case, variant) dispatch plan from GBR_PLAN."""
    raw = os.environ.get("GBR_PLAN", "")
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        case, _, variant = item.partition(":")
        out.append((case, variant or "baseline"))
    return out


_PLAN = _plan()
_DESCRIBE = [
    "prefill7168",
    "prefill1024",
    "prefill2304",
    "prefill5120",
    "rows11x7168",
    "rows22x7168",
    "rows55x7168",
    "rows110x7168",
    "prefill1000_partialw",
    "prefill7168_fp32gamma",
    "prefill1024_rmgamma",
    "height256x512",
    "width32x7168",
]


@pytest.mark.parametrize("case", _DESCRIBE)
def test_describe(device, case):
    """No dispatch: print the derived blocking + the broadcast plan for one case."""
    info = _bench().describe(device, case)
    for k, v in info.items():
        print(f"  {k:22s} {v}")


@pytest.mark.skipif(not _PLAN, reason="set GBR_PLAN='<case>:<variant>,...' to run the timed plan")
@pytest.mark.parametrize("step", range(max(1, len(_PLAN))), ids=lambda i: f"{i:02d}")
def test_plan(device, step):
    """ONE fresh dispatch per node, in GBR_PLAN order — the timed unit."""
    case, variant = _PLAN[step]
    print(f"\n### step {step}: {case} / {variant}")
    _bench().run_case(device, case, variant, mode=os.environ.get("GBR_MODE", "random"))


# ---------------------------------------------------------------------------
# correctness gates — never timed
# ---------------------------------------------------------------------------

_CORRECT_CASES = [
    "prefill1024",
    "prefill7168",
    "prefill1000_partialw",
    "prefill7168_fp32gamma",
    "prefill1024_rmgamma",
    "rows11x7168",
    "height256x512",
    "width32x7168",
]


@pytest.mark.parametrize("case", _CORRECT_CASES)
@pytest.mark.parametrize("mode", ["random", "ones", "ramp"])
def test_correct_mcast(device, case, mode):
    _bench().run_case(device, case, "mcast", mode=mode)


@pytest.mark.parametrize("variant", ["mcast", "mcast_late"])
@pytest.mark.parametrize("case", _CORRECT_CASES)
def test_correct_bitexact(device, case, variant):
    """The strongest gate: the broadcast must be BIT-IDENTICAL to the op's read."""
    _bench().run_bitexact(device, case, variant, mode="ramp")


@pytest.mark.parametrize("case", _CORRECT_CASES)
@pytest.mark.parametrize("mode", ["random", "ramp"])
def test_correct_mcast_late(device, case, mode):
    _bench().run_case(device, case, "mcast_late", mode=mode)
