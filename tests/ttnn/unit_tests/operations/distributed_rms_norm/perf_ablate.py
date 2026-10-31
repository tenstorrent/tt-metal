# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""MEASUREMENT ONLY (ablation copy of eval/golden_tests/distributed_rms_norm/perf_measure.py): with
DRMS_ABLATE_EXCHANGE=1 the golden check is skipped (the ablated op output is wrong by design) so the exposed transport
time t(op) - t(op, ablated) can be measured on the same harness. Headline numbers come from perf_measure.py itself.

Device time of distributed_rms_norm's perf cases, by the eval/ccl_perf.py protocol (spin + 5 calls, traced).

Not a golden test: the name keeps it out of every golden-suite run (pytest only collects test_*.py from a
directory), so it runs only when named:

    CCL_PERF_EXPERIMENTS=5 scripts/run_safe_pytest.sh --no-precompile \\
        eval/golden_tests/distributed_rms_norm/perf_measure.py -s

  CCL_PERF_EXPERIMENTS  5 (default, the check while iterating) or 100 (the final numbers)
  CCL_PERF_CASES        "focus" (default: the PERF FOCUS case, LOOSE_CASES[0]), "all", or comma-separated indices
  CCL_PERF_OUT          result directory (default .eval/ccl_perf): <case>.json per case, read with
                        `python -m eval.ccl_perf show|compare`
  CCL_PERF_LABEL        tag stored in the result (default: the case id)

Each case is checked against the golden reference once before it is timed. The optimization target is the
fastest-chip mean; the slowest chip, chip average, spread, per-chip means and the dispatch check are recorded with it.
"""

from __future__ import annotations

import os
from pathlib import Path

from eval import ccl_perf

for _k, _v in ccl_perf.PROFILER_ENV.items():  # before the mesh opens
    os.environ.setdefault(_k, _v)

import pytest  # noqa: E402

from eval.golden_tests.distributed_rms_norm.feature_spec import LOOSE_CASES  # noqa: E402
from eval.golden_tests.distributed_rms_norm.helpers import prepare_distributed_rms_norm  # noqa: E402
from eval.golden_tests.distributed_rms_norm.test_golden import _loose_cases  # noqa: E402
from eval.golden_tests.distributed_rms_norm.topology import resolve_pinned_mesh, system_mesh_shape  # noqa: E402
from eval.golden_tests.distributed_rms_norm.conftest import _fabric_mesh_cache  # noqa: E402,F401
from ttnn.operations.distributed_rms_norm import distributed_rms_norm  # type: ignore  # noqa: E402

EXPERIMENTS = int(os.environ.get("CCL_PERF_EXPERIMENTS", str(ccl_perf.CHECK_EXPERIMENTS)))
OUT = Path(os.environ.get("CCL_PERF_OUT", ".eval/ccl_perf"))


def _selected():
    cases = _loose_cases()
    sel = os.environ.get("CCL_PERF_CASES", "focus")
    if sel == "all":
        return cases
    if sel == "focus":
        return cases[:1]
    return [cases[int(i)] for i in sel.split(",")]


@pytest.fixture
def perf_mesh(request, _fabric_mesh_cache):
    extras, axes = request.node.callspec.params["extras"], request.node.callspec.params["axes"]
    shape = system_mesh_shape()
    if extras.get("mesh_shape"):
        resolved, _ = resolve_pinned_mesh(extras["mesh_shape"], axes["topology"], axes["cluster_axis"])
        shape = resolved or shape
    return _fabric_mesh_cache.get(shape, trace_region_size=ccl_perf.TRACE_REGION_SIZE)


@pytest.mark.timeout(7200)  # ~2.5 s per experiment
@pytest.mark.parametrize("inputs,axes,extras", _selected())
def test_perf_measure(inputs, axes, extras, perf_mesh, request):
    case = request.node.callspec.id
    call, check = prepare_distributed_rms_norm(
        inputs, mesh_device=perf_mesh, extras=extras, op=distributed_rms_norm, **axes
    )
    if all(
        os.environ.get(k, "0") in ("", "0")
        for k in ("DRMS_ABLATE_EXCHANGE", "DRMS_ABLATE_WRITE", "DRMS_ABLATE_FINALIZE")
    ):
        check(call())
    else:
        call()
    meta = {
        "op": "distributed_rms_norm",
        "case": case,
        "focus": case == _loose_cases()[0].id,
        "roofline_us": extras.get("roofline_us"),
        "target_us": extras.get("target_us"),
        "experiments": EXPERIMENTS,
        "loose_cases": len(LOOSE_CASES),
    }
    ccl_perf.run(
        perf_mesh,
        call,
        experiments=EXPERIMENTS,
        out_path=OUT / f"{case}.json",
        label=os.environ.get("CCL_PERF_LABEL", case),
        meta=meta,
    )
