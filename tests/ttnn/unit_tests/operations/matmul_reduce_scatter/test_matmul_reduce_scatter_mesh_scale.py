# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mesh scale spec of matmul_reduce_scatter: both routes, call by call, on the mock topology the driver names.

Cases from test_acceptance.py: the smallest call (one tile per scattered block on an 8-device group), the call with
its optional arguments set (HiFi4, an explicit output memory config, a positive scatter_dim) and a multi-tile
scatter -2 call with a bf8b weight, whose per-core work split follows the group size. Every scattered extent is 256
elements, so it splits into whole tile blocks for groups of 2, 4 and 8. The variants are the mesh-dependent knobs:
each cluster axis, Linear and Ring (refused without a wrapping fabric, on both routes alike), 1 or 2 links and every
link a hop has (None).

No case fills L1: in one process the frozen generic copy keeps its own per-mesh staging buffer in L1 after its route
ran, so a plan sized to the whole L1 budget (MiMo o_proj, the auto chunk split) would clash with it on the next route.
No output is in L1 either: the harness keeps every output alive, and the op's staging buffer, reallocated when the
group size changes between variants, would land at a new L1 address among them (an op-internal address the
comparer cannot name). test_acceptance.py runs the L1 output on the device.
"""
import pytest
import torch
import ttnn
import intree_mesh_scale as ms
import matmul_reduce_scatter_generic  # frozen generic route, provided by the driver
from ttnn.operations import matmul_reduce_scatter as op_pkg

# per-device (A, W) shapes, weight dtype, fixed kwargs
CASES = [
    pytest.param(((1, 1, 256, 64), (64, 256)), ttnn.bfloat16, dict(scatter_dim=-1), id="one_tile_block"),
    pytest.param(
        ((1, 256, 256), (256, 768)),
        ttnn.bfloat16,
        dict(
            scatter_dim=1,
            compute_kernel_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        id="rank3_hifi4_dram_out",
    ),
    pytest.param(((1, 1, 256, 512), (512, 1024)), ttnn.bfloat8_b, dict(scatter_dim=-2), id="multi_tile_w_bf8b"),
]
VARIANTS = {
    f"axis{axis}_{topo_name}_links{links}": {"cluster_axis": axis, "topology": topology, "num_links": links}
    for axis in (0, 1)
    for topo_name, topology in (("linear", ttnn.Topology.Linear), ("ring", ttnn.Topology.Ring))
    for links in (1, 2)
}
VARIANTS.update(
    {
        f"axis{axis}_linear_all_links": {"cluster_axis": axis, "topology": ttnn.Topology.Linear, "num_links": None}
        for axis in (0, 1)
    }
)


def _router_config():
    """The production router payload the acceptance suite opens its meshes with (conftest.py)."""
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    return cfg


@pytest.fixture(scope="module")
def scale_mesh():
    mesh = ms.open_mesh(router_config=_router_config())
    yield mesh
    ms.close_mesh(mesh)


@pytest.mark.parametrize(("shapes", "weight_dtype", "kwargs"), CASES)
def test_mesh_scale(request, scale_mesh, shapes, weight_dtype, kwargs):
    mesh = scale_mesh
    a_shape, w_shape = shapes
    a_host = torch.randn(a_shape).to(torch.bfloat16)
    w_host = (torch.randn(w_shape) * w_shape[0] ** -0.5).to(torch.bfloat16)

    def make_inputs(_variant):
        def on_mesh(host, dtype):
            return ttnn.from_torch(
                host,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        return {"a": on_mesh(a_host, ttnn.bfloat16), "w": on_mesh(w_host, weight_dtype)}

    def call(op, inputs, **variant):
        return op(inputs["a"], inputs["w"], **kwargs, **variant)

    native = lambda *a, **k: op_pkg.matmul_reduce_scatter(*a, **k)  # noqa: E731
    ms.MeshScale(matmul_reduce_scatter_generic.matmul_reduce_scatter, native, mesh).run(
        request.node.callspec.id, make_inputs, call, VARIANTS
    )
