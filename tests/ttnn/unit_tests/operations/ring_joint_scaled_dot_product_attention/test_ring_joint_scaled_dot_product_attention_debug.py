# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Debug tests for ring_joint_scaled_dot_product_attention. DO NOT DELETE — documents the debugging process.

test_mesh_switch_only: open the 2D system mesh, then the 1x4 TORUS_XY ring-mock mesh, running NO op — isolates whether
the ethernet-core timeout seen when the acceptance suite switches fabric configs mid-session is caused by the op.
"""
import pytest
import ttnn

# mesh_device / _fabric_mesh_cache come from this directory's conftest.py (one session-wide fabric-mesh cache).
from eval.golden_tests.ring_joint_scaled_dot_product_attention.topology import system_mesh_shape


@pytest.mark.parametrize(
    "mesh_shape,fabric_config",
    [(system_mesh_shape(), ttnn.FabricConfig.FABRIC_2D), ((1, 4), ttnn.FabricConfig.FABRIC_2D_TORUS_XY)],
    ids=["mesh2d", "ring_mock"],
)
def test_mesh_switch_only(mesh_shape, fabric_config, mesh_device):
    assert tuple(mesh_device.shape) == tuple(mesh_shape)


@pytest.mark.parametrize(
    "mesh_shape,fabric_config", [(system_mesh_shape(), ttnn.FabricConfig.FABRIC_2D)], ids=["mesh2d"]
)
def test_reference_all_gather_teardown(mesh_shape, fabric_config, mesh_device):
    """Does the reference transport (examples/fabric_all_gather) also leave Ethernet cores active at teardown?
    (Reference comparison only — the op itself never calls the example.)"""
    import torch
    from ttnn.operations.examples.fabric_all_gather.program_descriptor_with_inline_kernels import fabric_all_gather

    rows, cols = mesh_shape
    x = ttnn.from_torch(
        torch.randn(rows * 64, cols * 64).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(rows, cols)),
    )
    fabric_all_gather(x, cluster_axis=0, topology=ttnn.Topology.Linear, num_links=2)
    ttnn.synchronize_device(mesh_device)


@pytest.mark.parametrize("num_links", [1, 2])
@pytest.mark.parametrize(
    "mesh_shape,fabric_config", [(system_mesh_shape(), ttnn.FabricConfig.FABRIC_2D)], ids=["mesh2d"]
)
def test_op_once_teardown(mesh_shape, fabric_config, num_links, mesh_device):
    """One call of the op (all-ones, single tile): bisects the Ethernet-core-stuck-at-teardown symptom."""
    import torch
    from ttnn.operations.ring_joint_scaled_dot_product_attention import ring_joint_scaled_dot_product_attention

    rows, cols = mesh_shape
    mk = lambda: ttnn.from_torch(
        torch.ones(rows, 1, 32 * cols, 32).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 2), mesh_shape=(rows, cols)),
    )
    out, _ = ring_joint_scaled_dot_product_attention(
        mk(), mk(), mk(), logical_n=rows * 32, cluster_axis=0, num_links=num_links
    )
    ttnn.synchronize_device(mesh_device)
    for t in ttnn.get_device_tensors(out):
        assert torch.allclose(ttnn.to_torch(t).float(), torch.ones(1, 1, 32, 32), atol=0.05)
