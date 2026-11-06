# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host cold/hot timing of distributed_rms_norm: native C++ op versus the frozen generic_op route.

Runs on the suite's fabric mesh (`drms_mesh`, the system mesh under FABRIC_2D_TORUS_XY); the cases come from
test_acceptance.py.
"""
import json
import os

import distributed_rms_norm_generic  # frozen generic route, provided by the driver
import intree_host_comparison as hc
import pytest
import ttnn
from ttnn.operations import distributed_rms_norm as op_pkg

from tests.ttnn.unit_tests.operations.distributed_rms_norm.pre_migration import drms_pre_migration_helpers as H
from tests.ttnn.unit_tests.operations.distributed_rms_norm.pre_migration.drms_pre_migration_helpers import Case

BLOCKS = int(os.environ.get("BENCH_BLOCKS", "20"))

CASES = [
    # smallest supported call: one tile per device
    pytest.param(Case((1, 1, 32, 32)), id="single_tile"),
    # W-split over several waves, gamma as ROW_MAJOR sticks on the other axis
    pytest.param(
        Case((1, 1, 640, 1792), cluster_axis=0, gamma=(ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)),
        id="wsplit_two_waves_sticks",
    ),
    # large enough that device time matters: prefill, row-parallel, fp32 gamma
    pytest.param(Case((1, 1, 2048, 1280), gamma=(ttnn.float32, ttnn.TILE_LAYOUT)), id="prefill_row_parallel"),
    # sharded output placement independent of the interleaved input
    pytest.param(Case((1, 1, 32, 1792), out_mem="width_8x3"), id="dram_to_width_8x3"),
]


@pytest.mark.parametrize("case", CASES)
def test_host_perf(drms_mesh, case, request):
    mesh = drms_mesh
    x, gamma = H.host_data(case, tuple(mesh.shape), seed=0)
    reference = H.Call(case, x, gamma, None, None, {})  # check_output reads only case, x and gamma

    def make_input():
        return H.device_input(x, case, mesh)

    def make_kwargs():
        return H.op_kwargs(case, H.device_gamma(gamma, case, mesh))

    def validate(out):
        H.check_output(out, reference, mesh)

    def route(name, op):
        return hc.RouteAdapter(
            name, lambda _case, phase, block: hc.BoundCall(op, validate, (make_input(),), make_kwargs())
        )

    native = lambda *a, **k: op_pkg.distributed_rms_norm(*a, **k)  # noqa: E731
    case_id = request.node.callspec.id
    result = hc.HostComparison(
        route("generic", distributed_rms_norm_generic.distributed_rms_norm),
        route("native", native),
        hc.BenchmarkRuntime.from_ttnn(mesh),
    ).run(case_id, None, blocks=BLOCKS)
    out = hc.evidence_path(case_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.as_dict(), indent=1, sort_keys=True))
