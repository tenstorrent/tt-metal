# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""DEVICE_PRINT launcher for a perf case (no profiler: profiler + DEVICE_PRINT segfaults on the mesh): one checked
call, then a traced spin + ITERS calls so every chip starts together (the print of the traced calls is the steady
state). Not collected by default (no test_ prefix):

    TT_METAL_DEVICE_PRINT=1 TT_METAL_DPRINT_CORES=all TT_METAL_DPRINT_RISCVS=BR CCL_PERF_CASES=1 \\
        scripts/run_safe_pytest.sh --no-precompile tests/ttnn/unit_tests/operations/all_gather_rms_norm/dprint_agrms.py -s
"""

from __future__ import annotations

import os

import pytest

from eval import ccl_perf
from eval.golden_tests.all_gather_rms_norm.conftest import _fabric_mesh_cache  # noqa: F401
from eval.golden_tests.all_gather_rms_norm.helpers import prepare_all_gather_rms_norm
from eval.golden_tests.all_gather_rms_norm.test_golden import _loose_cases
from eval.golden_tests.all_gather_rms_norm.topology import resolve_pinned_mesh, system_mesh_shape
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm

ITERS = int(os.environ.get("AGRMS_DP_ITERS", "3"))


def _selected():
    cases = _loose_cases()
    idx = [int(i) for i in os.environ.get("CCL_PERF_CASES", "1").split(",")]
    return [pytest.param(*cases[i].values, i, id=f"case{i}") for i in idx]


@pytest.fixture
def perf_mesh(request, _fabric_mesh_cache):  # noqa: F811
    extras, axes = request.node.callspec.params["extras"], request.node.callspec.params["axes"]
    shape = system_mesh_shape()
    if extras.get("mesh_shape"):
        resolved, _ = resolve_pinned_mesh(extras["mesh_shape"], axes["topology"], axes["cluster_axis"])
        shape = resolved or shape
    return _fabric_mesh_cache.get(shape, trace_region_size=ccl_perf.TRACE_REGION_SIZE)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("inputs,axes,extras,index", _selected())
def test_dprint(inputs, axes, extras, index, perf_mesh):
    import ttnn

    call, check = prepare_all_gather_rms_norm(
        inputs, mesh_device=perf_mesh, extras=extras, op=all_gather_rms_norm, **axes
    )
    out = call()
    if not os.environ.get("AGRMS_ABL"):
        check(out)
    spin = ccl_perf.spin_op(perf_mesh)
    spin()
    call()
    ttnn.synchronize_device(perf_mesh)
    tid = ttnn.begin_trace_capture(perf_mesh, cq_id=0)
    spin()
    for _ in range(ITERS):
        call()
    ttnn.end_trace_capture(perf_mesh, tid, cq_id=0)
    print("=== TRACED ===", flush=True)
    ttnn.execute_trace(perf_mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(perf_mesh)
    ttnn.release_trace(perf_mesh, tid)
