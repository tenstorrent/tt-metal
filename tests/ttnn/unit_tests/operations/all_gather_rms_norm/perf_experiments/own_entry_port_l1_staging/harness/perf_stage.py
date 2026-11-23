# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""EXPERIMENT own_entry_port_l1_staging: eval perf_measure.py protocol on the op (AGRN_EXP_OP=base) or on the
staging copy (AGRN_EXP_OP=stage, default). Golden check always on (unless AGRN_ABLATE).

Perf ABLATION copy of eval perf_measure.py: same protocol; the golden check is skipped when AGRN_ABLATE is set.

Not a golden test: the name keeps it out of every golden-suite run (pytest only collects test_*.py from a
directory), so it runs only when named:

    CCL_PERF_EXPERIMENTS=5 scripts/run_safe_pytest.sh --no-precompile \\
        eval/golden_tests/all_gather_rms_norm/perf_measure.py -s

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
import re
from pathlib import Path

from eval import ccl_perf

for _k, _v in ccl_perf.PROFILER_ENV.items():  # before the mesh opens
    os.environ.setdefault(_k, _v)

import pytest  # noqa: E402

from eval.golden_tests.all_gather_rms_norm.feature_spec import LOOSE_CASES  # noqa: E402
from eval.golden_tests.all_gather_rms_norm.helpers import prepare_all_gather_rms_norm  # noqa: E402
from eval.golden_tests.all_gather_rms_norm.test_golden import _loose_cases  # noqa: E402
from eval.golden_tests.all_gather_rms_norm.topology import resolve_pinned_mesh, system_mesh_shape  # noqa: E402

if os.environ.get("AGRN_EXP_OP", "stage") == "base":
    from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm  # type: ignore  # noqa: E402
else:
    from ttnn.operations.all_gather_rms_norm.perf_experiments.own_entry_port_l1_staging.agrn_stage import (  # noqa: E402
        all_gather_rms_norm,
    )

EXPERIMENTS = int(os.environ.get("CCL_PERF_EXPERIMENTS", str(ccl_perf.CHECK_EXPERIMENTS)))
OUT = Path(os.environ.get("CCL_PERF_OUT", ".eval/ccl_perf"))


def _file_stem(case: str) -> str:
    """A case id with every axis spelled out can pass the 255-byte file-name limit: keep a readable head + a hash."""
    if len(case) <= 120:
        return case
    import hashlib

    return f"{case[:100]}-{hashlib.sha1(case.encode()).hexdigest()[:10]}"


def _selected():
    pats = [re.compile(p) for p in os.environ.get("AGRN_PERF_CELLS", "").split(";") if p]
    if pats:
        # golden cells (no extras) matching every regex — the domain sweep
        from eval.golden_tests.all_gather_rms_norm.test_golden import _parametrize_cases

        return [
            pytest.param(*c.values, {}, id=c.id)
            for c in _parametrize_cases()
            if all(p.search(c.id) for p in pats) and not c.marks
        ]
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
    if not extras:
        from eval.golden_tests.all_gather_rms_norm.topology import registry_mesh

        shape = registry_mesh(axes["topology"], axes["cluster_axis"])[0] or shape
    if extras.get("mesh_shape"):
        resolved, _ = resolve_pinned_mesh(extras["mesh_shape"], axes["topology"], axes["cluster_axis"])
        shape = resolved or shape
    return _fabric_mesh_cache.get(shape, trace_region_size=ccl_perf.TRACE_REGION_SIZE)


@pytest.mark.timeout(7200)  # ~2.5 s per experiment
@pytest.mark.parametrize("inputs,axes,extras", _selected())
def test_perf_stage(inputs, axes, extras, perf_mesh, request):
    case = request.node.callspec.id
    call, check = prepare_all_gather_rms_norm(
        inputs, mesh_device=perf_mesh, extras=extras, op=all_gather_rms_norm, **axes
    )
    if not os.environ.get("AGRN_ABLATE"):
        check(call())  # ablated runs (stubbed payloads) produce garbage: timing only
    meta = {
        "op": "all_gather_rms_norm",
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
        out_path=OUT / f"{_file_stem(case)}.json",
        label=os.environ.get("CCL_PERF_LABEL", case),
        meta=meta,
    )
