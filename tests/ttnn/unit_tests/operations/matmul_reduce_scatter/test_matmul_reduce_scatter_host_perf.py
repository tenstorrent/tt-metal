# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host cold/hot timing of matmul_reduce_scatter: native C++ op versus the frozen generic_op route.

The cases are test_acceptance.py's: the smallest call (one tile per scattered block), the PERF FOCUS contract on the
balanced ring (large enough that device time matters) and the call with its optional arguments set (HiFi4, an L1
output, a positive scatter_dim). The meshes are the suite's conftest.py placements (mrs_mesh: "line" | "ring").
"""
import json
import os

import pytest
import torch
import ttnn
import intree_host_comparison as hc
import matmul_reduce_scatter_generic  # frozen generic route, provided by the driver
from ttnn.operations import matmul_reduce_scatter as op_pkg

BLOCKS = int(os.environ.get("BENCH_BLOCKS", "20"))
BF16, BF8 = ttnn.bfloat16, ttnn.bfloat8_b
L1_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
# (min PCC, max rel-RMS) per weight dtype: the golden suite's helpers.TOLERANCES, as in test_acceptance.py
TOL = {BF16: (0.999, 0.02), BF8: (0.998, 0.03)}

# placement, per-device (A, W) shapes for group size g, scatter_dim, weight dtype, kwargs (cluster_axis None: the
# ring placement's axis; compute: (math fidelity, fp32_dest_acc_en))
CASES = [
    pytest.param(
        "line",
        lambda g: ((1, 1, 32 * g, 64), (64, 32 * g)),
        -1,
        BF16,
        dict(cluster_axis=1, topology=ttnn.Topology.Linear, num_links=1),
        id="one_tile_block",
    ),
    pytest.param(
        "ring",
        lambda g: ((1, 1, 640, 2048), (2048, 7168)),
        -1,
        BF8,
        dict(cluster_axis=None, topology=ttnn.Topology.Ring, num_links=2, compute=(ttnn.MathFidelity.HiFi2, False)),
        id="ring_kimi_focus",
    ),
    pytest.param(
        "line",
        lambda g: ((1, 512, 256), (256, 768)),
        1,
        BF16,
        dict(
            cluster_axis=0,
            topology=ttnn.Topology.Linear,
            num_links=1,
            compute=(ttnn.MathFidelity.HiFi4, True),
            memory_config=L1_INTERLEAVED,
        ),
        id="rank3_hifi4_l1_out",
    ),
]


def _host_mesh_tensor(stacked, mesh, dtype):
    """Stacked (R, C, *shape) per-device data as a host tensor sharded for `mesh` (moved to the device per call)."""
    rows, cols = stacked.shape[0], stacked.shape[1]
    whole = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        whole,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=(rows, cols)),
    )


def _held_values(host_tensor, shape):
    """What the device holds for a host tensor (block-float rounding), as a stacked bf16 torch tensor."""
    return torch.stack([ttnn.to_torch(t) for t in ttnn.get_device_tensors(host_tensor)]).reshape(shape)


@pytest.mark.parametrize(("mrs_mesh", "shapes", "scatter_dim", "weight_dtype", "kwargs"), CASES, indirect=["mrs_mesh"])
def test_host_perf(request, mrs_mesh, shapes, scatter_dim, weight_dtype, kwargs):
    mesh = mrs_mesh
    kwargs = dict(kwargs)
    axis = kwargs.pop("cluster_axis")
    axis = request.getfixturevalue("mrs_ring_axis") if axis is None else axis
    compute = kwargs.pop("compute", None)
    if compute is not None:
        kwargs["compute_kernel_config"] = ttnn.ComputeConfigDescriptor(
            math_fidelity=compute[0], fp32_dest_acc_en=compute[1]
        )
    rows, cols = tuple(mesh.shape)
    g = (rows, cols)[axis]
    a_shape, w_shape = shapes(g)
    sd = scatter_dim - len(a_shape) if scatter_dim >= 0 else scatter_dim
    torch.manual_seed(0)
    a = torch.randn((rows, cols, *a_shape)).to(torch.bfloat16)
    w = (torch.randn((rows, cols, *w_shape)) * w_shape[0] ** -0.5).to(torch.bfloat16)
    host_a, host_w = _host_mesh_tensor(a, mesh, BF16), _host_mesh_tensor(w, mesh, weight_dtype)
    w = _held_values(host_w, w.shape)

    # independent fp32 reference: each device's A @ W summed over its group; group position p keeps block p
    partial = {(r, c): a[r, c].float() @ w[r, c].float() for r in range(rows) for c in range(cols)}
    expected = {}
    for r in range(rows):
        for c in range(cols):
            members = [(i, c) for i in range(rows)] if axis == 0 else [(r, j) for j in range(cols)]
            total = sum(partial[m] for m in members)
            expected[(r, c)] = torch.chunk(total, g, dim=sd)[(r, c)[axis]].double()
    out_shape = [*a_shape[:-1], w_shape[-1]]
    out_shape[sd] //= g
    pcc_min, rms_max = TOL[weight_dtype]

    def make_inputs():
        return (
            ttnn.to_device(host_a, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            ttnn.to_device(host_w, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        )

    def validate(out):
        assert list(out.shape) == out_shape and out.dtype == BF16 and out.layout == ttnn.TILE_LAYOUT
        for idx, part in enumerate(ttnn.get_device_tensors(out)):
            got = ttnn.to_torch(part).double()
            ref = expected[divmod(idx, cols)]
            assert got.shape == ref.shape and torch.isfinite(got).all()
            ac, ec = got.flatten() - got.mean(), ref.flatten() - ref.mean()
            pcc = float((ac * ec).sum() / (ac.norm() * ec.norm()))
            rms = float((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
            assert pcc >= pcc_min and rms <= rms_max, f"device {idx}: pcc {pcc:.6f} rel_rms {rms:.6f}"

    call_kwargs = dict(cluster_axis=axis, scatter_dim=scatter_dim, **kwargs)

    def route(name, op):
        return hc.RouteAdapter(
            name, lambda case, phase, block: hc.BoundCall(op, validate, make_inputs(), dict(call_kwargs))
        )

    native = lambda *a, **k: op_pkg.matmul_reduce_scatter(*a, **k)  # noqa: E731
    case_id = request.node.callspec.id
    result = hc.HostComparison(
        route("generic", matmul_reduce_scatter_generic.matmul_reduce_scatter),
        route("native", native),
        hc.BenchmarkRuntime.from_ttnn(mesh),
    ).run(case_id, None, blocks=BLOCKS)
    out = hc.evidence_path(case_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.as_dict(), indent=1, sort_keys=True))
