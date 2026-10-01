# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Curated emule 2-chip (n300/p300) CCL microtests. Small, slow-dispatch, non-trace shapes that
# exercise the real ttnn fabric CCL ops on a 2-chip mesh under the emule backend. Parametrized over
# L1 and DRAM memory configs: L1 keeps the focus on the fabric/teleport path; DRAM additionally
# exercises the interleaved DRAM bank-view resolution — the host write, the kernel address
# generator, and the cross-chip fabric teleport must all agree on the DRAM backing + per-view
# address offset (Wormhole packs two views per physical channel, the odd view at +1 GB).

import pytest
import torch
import ttnn

MEM_CONFIGS = {
    "l1": ttnn.L1_MEMORY_CONFIG,
    "dram": ttnn.DRAM_MEMORY_CONFIG,
}


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("mem", ["l1", "dram"])
@pytest.mark.parametrize("dim, per_dev_width", [(3, 64), (3, 128), (3, 96), (3, 256), (3, 512)])
def test_emule_all_gather_2chip(mesh_device, mem, dim, per_dev_width):
    mem_cfg = MEM_CONFIGS[mem]
    num_devices = 2
    full_shape = (1, 1, 32, per_dev_width * num_devices)
    torch_input = torch.randn(full_shape, dtype=torch.bfloat16)

    tt_in = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=mem_cfg,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
    )

    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sems = [ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(2)]
    barrier = ttnn.create_global_semaphore(mesh_device, crs, 0)

    out = ttnn.experimental.all_gather_async(
        tt_in,
        dim,
        multi_device_global_semaphore=[sems[0], sems[1]],
        num_links=1,
        memory_config=mem_cfg,
        topology=ttnn.Topology.Linear,
        barrier_semaphore=barrier,
        num_workers_per_link=1,  # → num_workers_per_direction=1 → no fabric mux (direct teleport path)
    )
    ttnn.synchronize_device(mesh_device)

    out_torch = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    # Each device should hold the full gathered tensor; concat over the mesh axis = num_devices copies.
    expected = torch.cat([torch_input] * num_devices, dim=0)
    assert torch.allclose(out_torch.float(), expected.float(), atol=1e-2), "all_gather output mismatch"


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("mem", ["l1", "dram"])
@pytest.mark.parametrize("shape", [(1, 1, 1, 64), (1, 1, 3, 128), (1, 1, 32, 256), (1, 1, 64, 128), (1, 1, 32, 512)])
def test_emule_point_to_point_2chip(mesh_device, mem, shape):
    # point-to-point send chip0 -> chip1, exercising the fabric unicast-write + sem-inc teleport path.
    # Under DRAM it additionally exercises the interleaved DRAM bank-view resolution. Mirrors the t3000
    # p2p test but on a 2-chip mesh.
    mem_cfg = MEM_CONFIGS[mem]
    devices = 2
    multi_device_shape = (shape[0] * devices, *shape[1:])

    input_tensor_torch = torch.zeros(multi_device_shape, dtype=torch.bfloat16)
    input_tensor_torch[0 : shape[0], :, :, :] = (
        torch.linspace(1, torch.prod(torch.tensor(shape)).item(), torch.prod(torch.tensor(shape)).item())
        .reshape(shape)
        .to(dtype=torch.bfloat16)
    )
    input_tensor = ttnn.from_torch(
        input_tensor_torch,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=mem_cfg,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    coord0 = ttnn.MeshCoordinate((0, 0))
    coord1 = ttnn.MeshCoordinate((0, 1))

    sent = ttnn.point_to_point(input_tensor, coord0, coord1, topology=ttnn.Topology.Linear)
    sent_torch = ttnn.to_torch(sent, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    # chip1 should now hold what chip0 sent.
    assert torch.allclose(
        input_tensor_torch[0 : shape[0]].float(), sent_torch[shape[0] : 2 * shape[0]].float(), atol=1e-2
    ), "point_to_point send mismatch"
