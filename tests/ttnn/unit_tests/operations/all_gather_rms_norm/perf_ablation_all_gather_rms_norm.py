# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf probe (implementer): the eval CCL harness protocol (eval/ccl_perf.py: spin + traced calls, fastest chip) on
the perf cases, with the op's ablation hook AGRMS_ABLATE = EMPTY | LOCAL | TRANSPORT (see the program descriptor).
Not collected by directory runs (no test_ prefix in the file name); run it by path:

    AGRMS_ABLATE=TRANSPORT CCL_PERF_EXPERIMENTS=3 scripts/run_safe_pytest.sh --no-precompile \\
        tests/ttnn/unit_tests/operations/all_gather_rms_norm/perf_ablation_all_gather_rms_norm.py -s

Sharded A/B (Refinement 1): AGRMS_PERF_SHAPE=1,1,640,384 AGRMS_PERF_MEMORY_LAYOUT=HEIGHT_SHARDED re-runs the focus
case's axes on that per-device shape / placement; AGRMS_NATIVE_SHARD=0 selects the TensorAccessor fallback.

The un-ablated run is checked against the golden reference first; ablated runs are not (their output is garbage).
Headline numbers come from eval/golden_tests/all_gather_rms_norm/perf_measure.py, not from here.
"""

from __future__ import annotations

import os

from eval import ccl_perf

for _k, _v in ccl_perf.PROFILER_ENV.items():  # before the mesh opens
    os.environ.setdefault(_k, _v)

import pytest  # noqa: E402

from eval.golden_tests.all_gather_rms_norm.conftest import _FabricMeshCache  # noqa: E402
from eval.golden_tests.all_gather_rms_norm.helpers import prepare_all_gather_rms_norm  # noqa: E402
from eval.golden_tests.all_gather_rms_norm.test_golden import _loose_cases  # noqa: E402
from eval.golden_tests.all_gather_rms_norm.topology import resolve_pinned_mesh, system_mesh_shape  # noqa: E402
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm  # noqa: E402

EXPERIMENTS = int(os.environ.get("CCL_PERF_EXPERIMENTS", "3"))
ABLATE = os.environ.get("AGRMS_ABLATE", "")


def _selected():
    cases = _loose_cases()
    sel = os.environ.get("CCL_PERF_CASES", "focus")
    if sel == "all":
        return cases
    if sel == "focus":
        return cases[:1]
    return [cases[int(i)] for i in sel.split(",")]


@pytest.fixture(scope="module")
def mesh_cache():
    cache = _FabricMeshCache()
    yield cache
    cache.close()


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("inputs,axes,extras", _selected())
def test_perf_ablation(inputs, axes, extras, mesh_cache, request):
    shape = system_mesh_shape()
    if extras.get("mesh_shape"):
        resolved, _ = resolve_pinned_mesh(extras["mesh_shape"], axes["topology"], axes["cluster_axis"])
        shape = resolved or shape
    mesh = mesh_cache.get(shape, trace_region_size=ccl_perf.TRACE_REGION_SIZE)
    if os.environ.get("AGRMS_PERF_SHAPE"):
        inputs = (tuple(int(v) for v in os.environ["AGRMS_PERF_SHAPE"].split(",")),)
    if os.environ.get("AGRMS_PERF_NUM_LINKS"):
        axes = dict(axes, num_links=int(os.environ["AGRMS_PERF_NUM_LINKS"]))
    if os.environ.get("AGRMS_PERF_MEMORY_LAYOUT"):
        import ttnn

        axes = dict(axes, memory_layout=getattr(ttnn.TensorMemoryLayout, os.environ["AGRMS_PERF_MEMORY_LAYOUT"]))
    call, check = prepare_all_gather_rms_norm(inputs, mesh_device=mesh, extras=extras, op=all_gather_rms_norm, **axes)
    out = call()
    if not ABLATE:
        check(out)
    label = f"{request.node.callspec.id}-ablate={ABLATE or 'none'}"
    out_dir = os.environ.get("CCL_PERF_OUT")
    path = None
    if out_dir:
        from pathlib import Path

        ml = os.environ.get("AGRMS_PERF_MEMORY_LAYOUT", "")
        nat = "" if os.environ.get("AGRMS_NATIVE_SHARD", "1") == "1" else "-fallback"
        path = Path(out_dir) / f"{inputs[0][-2]}x{inputs[0][-1]}{ml and '-' + ml}{nat}-{ABLATE or 'none'}.json"
    ccl_perf.run(mesh, call, experiments=EXPERIMENTS, out_path=path, label=label)
