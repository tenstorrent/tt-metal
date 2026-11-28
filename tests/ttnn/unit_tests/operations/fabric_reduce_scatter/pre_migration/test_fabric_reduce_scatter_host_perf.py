# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host cold/hot timing of fabric_reduce_scatter: native C++ op versus the frozen generic_op route."""

import json
import os

import fabric_reduce_scatter_generic  # frozen generic route, provided by the driver
import intree_host_comparison as hc
import pytest
import ttnn
from ttnn.bringup import fabric_reduce_scatter_ttnn as op_pkg

from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params

BLOCKS = int(os.environ.get("BENCH_BLOCKS", "20"))

_MESH = tuple(int(v) for v in (os.environ.get("BRINGUP_TEST_MESH") or "2x2").split("x"))
MESH = pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param(_MESH, fabric2d_device_params(), id="x".join(map(str, _MESH)))],
    indirect=["mesh_device", "device_params"],
)

# (tiles per output block, dtype, kwargs): the acceptance suite's smallest case (axis0_rows_one_tile), a model-size
# column reduce (axis 1, 2 links, [1024, 4096] on a 4-wide axis), and the optional arguments set (whole mesh, exact
# fp32 adds, cached descriptors, a preallocated output).
CASES = [
    pytest.param((1, 1), ttnn.bfloat16, dict(cluster_axis=0), id="min_one_tile"),
    pytest.param((32, 32), ttnn.bfloat16, dict(cluster_axis=1, dim=3, num_links=2), id="axis1_cols_model"),
    pytest.param(
        (2, 2),
        ttnn.float32,
        dict(cluster_axis=None, dim=3, num_links=2, exact_fp32_add=True, cache_descriptors=True),
        id="mesh_exact_cached_out",
    ),
]


@MESH
@pytest.mark.parametrize(("tiles", "dtype", "kwargs"), CASES)
def test_host_perf(mesh_device, device_params, rs, tiles, dtype, kwargs, request):
    rows, cols = tuple(mesh_device.shape)
    cluster_axis, dim = kwargs.get("cluster_axis", 0), kwargs.get("dim", 2)
    g = rs.group_size(mesh_device, cluster_axis)
    R, W = rs.in_shape(g, dim, tiles)
    parts = rs.int_parts(rows, cols, R, W, seed=0)  # integer partials: the sum is exact in every dtype / add path
    expected = rs.expected(parts, cluster_axis, dim)
    with_output = kwargs.get("cache_descriptors", False)  # the optional-arguments case also preallocates the output

    def make_input():
        return rs.upload(mesh_device, parts, dtype)

    def make_output():
        if not with_output:
            return {}
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(rs.out_shape([1, 1, R, W], g, dim)),
            dtype,
            ttnn.TILE_LAYOUT,
            mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        return {"output": out}

    def validate(out):
        rs.assert_exact(rs.per_chip(out), expected, "host perf output")

    def route(name, op):
        return hc.RouteAdapter(
            name,
            lambda case, phase, block: hc.BoundCall(op, validate, (make_input(),), dict(kwargs, **make_output())),
        )

    native = lambda *a, **k: op_pkg.fabric_reduce_scatter(*a, **k)  # noqa: E731
    case_id = request.node.callspec.id
    result = hc.HostComparison(
        route("generic", fabric_reduce_scatter_generic.fabric_reduce_scatter),
        route("native", native),
        hc.BenchmarkRuntime.from_ttnn(mesh_device),
    ).run(case_id, None, blocks=BLOCKS)
    out = hc.evidence_path(case_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.as_dict(), indent=1, sort_keys=True))
