# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""high_bw_all_gather and matmul_reduce_scatter_prefill on a [RING,RING] 4x4 mesh, checked on the chips a process owns.

The ops place their kernels per chip from the chip's physical layout, so a mesh that two hosts share (rows 0-1 on
one host, rows 2-3 on the other) needs each host to learn where the other's chips put theirs. Each rank checks the
shards it holds against the torch reference for those chip positions. Run on one rank (a single Galaxy) it checks
every chip.

Launch like test_rr_prefill_mesh_allgather.py, with a binding from
models/demos/deepseek_v3_d_p/utils/rr_pipeline_binding.py.
"""

import pytest
import torch
from loguru import logger

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

SIDE = 4
NUM_LINKS = 2
PCC_BF8 = 0.998


def barrier():
    if ttnn.using_distributed_env():
        ttnn.distributed_context_barrier()


def local_shards(tensor):
    """`{(row, col): single-device host tensor}` for the shards this process holds."""
    coords = list(tensor.device_coords())
    return {(int(c[0]), int(c[1])): s for c, s in zip(coords, tensor.cpu_shards(coords))}


@pytest.fixture(scope="module")
def rr_mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D_TORUS_XY)
    try:
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(SIDE, SIDE))
        try:
            yield mesh_device
        finally:
            barrier()
            ttnn.close_mesh_device(mesh_device)
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["axis0", "axis1"])
def test_high_bw_all_gather(rr_mesh, cluster_axis):
    rows_per_device, width = 64, 256
    shape = (1, 1, rows_per_device * SIDE, width)
    torch.manual_seed(0)
    host_input = torch.rand(shape, dtype=torch.bfloat16)
    dims = (2, None) if cluster_axis == 0 else (None, 2)
    device_input = ttnn.from_torch(
        host_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(rr_mesh, dims=dims, mesh_shape=tuple(rr_mesh.shape)),
        device=rr_mesh,
    )
    output = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(rr_mesh),
        device=rr_mesh,
    )
    gathered = ttnn.experimental.high_bw_all_gather(
        device_input, dim=2, output_tensor=output, cluster_axis=cluster_axis, num_links=NUM_LINKS
    )
    ttnn.synchronize_device(rr_mesh)
    shards = local_shards(gathered)
    assert shards, "this process holds no shard"
    for coord, shard in shards.items():
        assert torch.equal(ttnn.to_torch(shard), host_input), f"chip {coord}: gathered rows differ"
    logger.info(f"RR_CHECK OK high_bw_all_gather axis {cluster_axis}: {len(shards)} local chips")
    barrier()


def to_mesh(stacked, mesh_device, dtype):
    rows, cols = stacked.shape[0], stacked.shape[1]
    full = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        full,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(rows, cols)),
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["axis0", "axis1"])
def test_matmul_reduce_scatter_prefill(rr_mesh, cluster_axis):
    m, k, n = 640, 2048, 4096
    torch.manual_seed(0)
    a = torch.randn((SIDE, SIDE, 1, 1, m, k)).to(torch.bfloat16)
    w = (torch.randn((SIDE, SIDE, k, n)) * k**-0.5).to(torch.bfloat16)
    ta = to_mesh(a.reshape(SIDE, SIDE, m, k), rr_mesh, ttnn.bfloat16)
    tw = to_mesh(w, rr_mesh, ttnn.bfloat16)
    out = ttnn.experimental.matmul_reduce_scatter_prefill(
        ta,
        tw,
        cluster_axis=cluster_axis,
        scatter_dim=-1,
        topology=ttnn.Topology.Ring,
        num_links=NUM_LINKS,
        compute_kernel_config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2),
    )
    ttnn.synchronize_device(rr_mesh)
    shards = local_shards(out)
    assert shards, "this process holds no shard"
    for (r, c), shard in shards.items():
        line = [(g, c) if cluster_axis == 0 else (r, g) for g in range(SIDE)]
        total = sum(a[p].reshape(m, k).float() @ w[p].float() for p in line)
        position = r if cluster_axis == 0 else c
        expected = torch.chunk(total, SIDE, dim=-1)[position]
        got = ttnn.to_torch(shard).float().reshape(expected.shape)
        passed, message = comp_pcc(expected, got, PCC_BF8)
        assert passed, f"chip ({r}, {c}): {message}"
    logger.info(f"RR_CHECK OK matmul_reduce_scatter_prefill axis {cluster_axis}: {len(shards)} local chips")
    barrier()
