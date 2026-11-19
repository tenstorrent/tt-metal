# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Same protocol as eval/golden_tests/all_gather_rms_norm/perf_measure.py (eval/ccl_perf.run: spin + 5 traced
calls per pattern), but with a SHORT result file name: the eval script names its JSON after the full case id, which
is longer than the 255-byte filename limit for the focus case (OSError ENAMETOOLONG). Not collected by default
(no test_ prefix); run it by name:

    CCL_PERF_EXPERIMENTS=5 CCL_PERF_OUT=<dir> scripts/run_safe_pytest.sh --no-precompile \\
        tests/ttnn/unit_tests/operations/all_gather_rms_norm/perf_agrms.py -s

Writes <dir>/case<i>.json (i = LOOSE_CASES index; 0 = PERF FOCUS), read with `python -m eval.ccl_perf show|compare`.
"""

from __future__ import annotations

import os
from pathlib import Path

from eval import ccl_perf

for _k, _v in ccl_perf.PROFILER_ENV.items():  # before the mesh opens
    os.environ.setdefault(_k, _v)

import pytest  # noqa: E402

from eval.golden_tests.all_gather_rms_norm.conftest import _fabric_mesh_cache  # noqa: E402,F401  (fixture)
from eval.golden_tests.all_gather_rms_norm.feature_spec import LOOSE_CASES  # noqa: E402
from eval.golden_tests.all_gather_rms_norm.helpers import prepare_all_gather_rms_norm  # noqa: E402
from eval.golden_tests.all_gather_rms_norm.test_golden import _loose_cases  # noqa: E402
from eval.golden_tests.all_gather_rms_norm.topology import resolve_pinned_mesh, system_mesh_shape  # noqa: E402
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm  # noqa: E402

EXPERIMENTS = int(os.environ.get("CCL_PERF_EXPERIMENTS", str(ccl_perf.CHECK_EXPERIMENTS)))
OUT = Path(os.environ.get("CCL_PERF_OUT", ".eval/ccl_perf"))


def _selected():
    cases = _loose_cases()
    sel = os.environ.get("CCL_PERF_CASES", "focus")
    idx = list(range(len(cases))) if sel == "all" else [0] if sel == "focus" else [int(i) for i in sel.split(",")]
    return [pytest.param(*cases[i].values, i, id=f"case{i}") for i in idx]


@pytest.fixture
def perf_mesh(request, _fabric_mesh_cache):  # noqa: F811
    extras, axes = request.node.callspec.params["extras"], request.node.callspec.params["axes"]
    shape = system_mesh_shape()
    if extras.get("mesh_shape"):
        resolved, _ = resolve_pinned_mesh(extras["mesh_shape"], axes["topology"], axes["cluster_axis"])
        shape = resolved or shape
    return _fabric_mesh_cache.get(shape, trace_region_size=ccl_perf.TRACE_REGION_SIZE)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("inputs,axes,extras,index", _selected())
def test_perf_agrms(inputs, axes, extras, index, perf_mesh):
    call, check = prepare_all_gather_rms_norm(
        inputs, mesh_device=perf_mesh, extras=extras, op=all_gather_rms_norm, **axes
    )
    check(call())
    meta = {
        "op": "all_gather_rms_norm",
        "case": f"case{index}",
        "focus": index == 0,
        "roofline_us": extras.get("roofline_us"),
        "target_us": extras.get("target_us"),
        "experiments": EXPERIMENTS,
        "loose_cases": len(LOOSE_CASES),
    }
    ccl_perf.run(
        perf_mesh,
        call,
        experiments=EXPERIMENTS,
        out_path=OUT / f"case{index}.json",
        label=os.environ.get("CCL_PERF_LABEL", f"case{index}"),
        meta=meta,
    )
