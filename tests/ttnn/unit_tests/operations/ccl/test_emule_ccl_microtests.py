# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Curated emule 2-chip (n300/p300) CCL microtests. Small, slow-dispatch, non-trace shapes that
# exercise the real ttnn fabric CCL ops on a 2-chip mesh under the emule backend. L1 memory is used
# to keep the focus on the fabric/teleport path (the DRAM bank-view model is a separate concern).

import math

import pytest
import torch
import ttnn


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("dim, per_dev_width", [(3, 64), (3, 128), (3, 96), (3, 256), (3, 512)])
def test_emule_all_gather_2chip_l1(mesh_device, dim, per_dev_width):
    num_devices = 2
    full_shape = (1, 1, 32, per_dev_width * num_devices)
    torch_input = torch.randn(full_shape, dtype=torch.bfloat16)

    tt_in = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
    )

    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
    )
    sems = [ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(2)]
    barrier = ttnn.create_global_semaphore(mesh_device, crs, 0)

    out = ttnn.experimental.all_gather_async(
        tt_in,
        dim,
        multi_device_global_semaphore=[sems[0], sems[1]],
        num_links=1,
        memory_config=ttnn.L1_MEMORY_CONFIG,
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
@pytest.mark.parametrize(
    "shape", [(1, 1, 1, 64), (1, 1, 3, 128), (1, 1, 32, 256), (1, 1, 64, 128), (1, 1, 32, 512)]
)
def test_emule_point_to_point_2chip_l1(mesh_device, shape):
    # L1-resident point-to-point send chip0 -> chip1 (and back), exercising the fabric unicast-write +
    # sem-inc teleport path without touching the DRAM bank-view model. Mirrors the t3000 p2p test but on
    # a 2-chip mesh in L1.
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
        memory_config=ttnn.L1_MEMORY_CONFIG,
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


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("M, K, N", [(32, 64, 64), (32, 128, 96), (64, 64, 64), (32, 256, 128)])
def test_emule_all_gather_then_matmul_2chip_l1(mesh_device, M, K, N):
    # matmul + CCL on a 2-chip mesh (non-fused pipeline): shard the activation over its K dim across the
    # two chips, all_gather to reconstruct the full activation on each chip, then run a data-parallel
    # matmul against a replicated weight. Verifies that a CCL collective and matmul compose on the emule
    # mesh. L1-resident, slow-dispatch, no trace (matches the curated-suite policy).
    num_devices = 2
    torch.manual_seed(0)

    full_act = torch.randn((1, 1, M, K), dtype=torch.bfloat16)
    weight = torch.randn((1, 1, K, N), dtype=torch.bfloat16)
    expected = torch.matmul(full_act.float(), weight.float())

    # Activation sharded over K (dim=3): each chip holds (1,1,M,K/num_devices).
    act_tt = ttnn.from_torch(
        full_act,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    # Weight replicated on every chip.
    weight_tt = ttnn.from_torch(
        weight,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sems = [ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(2)]
    barrier = ttnn.create_global_semaphore(mesh_device, crs, 0)

    # CCL: reconstruct the full activation (1,1,M,K) on every chip.
    ag = ttnn.experimental.all_gather_async(
        act_tt,
        dim=3,
        multi_device_global_semaphore=[sems[0], sems[1]],
        num_links=1,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
        barrier_semaphore=barrier,
        num_workers_per_link=1,
    )

    # matmul: data-parallel, full activation @ replicated weight on each chip.
    mm = ttnn.matmul(ag, weight_tt, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.synchronize_device(mesh_device)

    # Both chips computed the same full result; take chip 0's copy.
    mm_torch = ttnn.to_torch(mm, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    got = mm_torch[0:1].float()
    pcc = torch.corrcoef(torch.stack([got.flatten(), expected.flatten()]))[0, 1].item()
    assert pcc > 0.99, f"all_gather+matmul PCC too low: {pcc}"


@pytest.mark.xfail(
    reason="Fused all_gather_matmul_async is not runnable on a 2-chip n300 mesh for two independent "
    "reasons, both outside emule's fabric scope: (1) it uses sub-devices to partition CCL-worker vs "
    "matmul cores, and sub-device managers require fast dispatch (emule is slow-dispatch-only) — the "
    "first TT_FATAL hit here; (2) even past that, the op asserts !(Linear && fuse_op) i.e. requires Ring, "
    "but silicon's get_usable_topology degrades Ring->Linear on a 2-chip mesh (get_boundary_mode forces "
    "NONE for a 2-chip mesh), so the fused op is only applicable on ring-capable meshes (>=3 chips). The "
    "2-chip CCL+matmul path is the non-fused all_gather->matmul pipeline (test above). Runnable once n300 "
    "scales to a ring-capable, fast-dispatch-equivalent config.",
    strict=False,
    raises=RuntimeError,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("M, K, N", [(32, 64, 64)])
def test_emule_all_gather_matmul_fused_2chip_l1(mesh_device, M, K, N):
    # FUSED matmul + CCL: ttnn.experimental.all_gather_matmul_async on a 2-chip mesh. The fused op runs
    # the all_gather on a reserved sub-device (CCL workers offset from the matmul cores) and hands the
    # gathered activation to the matmul via the writer/reader fuse_op (OpSignaler) path. Slow-dispatch,
    # no trace. Sub-devices ARE required (the op partitions CCL vs matmul cores), mirroring the t3000
    # setup; emule's slow-dispatch sub-device support makes that reachable. The op nonetheless can't run
    # on 2 chips (needs a real ring) — see the xfail reason.
    num_devices = 2
    torch.manual_seed(0)

    # Reserve the whole compute grid as the CCL worker sub-device (matches the canonical t3000 setup);
    # the matmul and all_gather workers live in disjoint core ranges within it.
    grid = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    try:
        # ag_output_shape = the FULL (gathered) activation; the per-chip input is sharded from it over K.
        ag_output_shape = (1, 1, M, K)
        full_act = torch.randn(ag_output_shape, dtype=torch.bfloat16)
        weight = torch.randn((1, 1, K, N), dtype=torch.bfloat16)
        expected = torch.matmul(full_act.float(), weight.float())

        act_tt = ttnn.from_torch(
            full_act,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        )
        weight_tt = ttnn.from_torch(
            weight,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        # Persistent all_gather output buffer (the fused op reads it as the matmul activation).
        persistent_ag = ttnn.from_torch(
            torch.zeros(ag_output_shape, dtype=torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        sems = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(2)]
        barrier = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)

        # Matmul program config (the fused op requires a 1D/2D multicast config). Single-core matmul on
        # (0,0); the all_gather workers live on the last grid row (disjoint), within the CCL sub-device.
        mm_grid = (1, 1)
        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=mm_grid,
            in0_block_w=K // 32,
            out_subblock_h=1,
            out_subblock_w=min(N // 32, 4),
            per_core_M=max(1, math.ceil(M / 32 / mm_grid[1])),
            per_core_N=max(1, math.ceil(N / 32 / mm_grid[0])),
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )

        ag_out, mm_out = ttnn.experimental.all_gather_matmul_async(
            act_tt,
            weight_tt,
            persistent_output_buffer=persistent_ag,
            dim=3,
            multi_device_global_semaphore=[sems[0], sems[1]],
            all_gather_core_grid_offset=(0, grid.y - 1),  # CCL workers on the last row; matmul on (0,0)
            barrier_semaphore=barrier,
            subdevice_id=worker_sub_device_id,
            num_links=1,
            memory_config_ag=ttnn.L1_MEMORY_CONFIG,
            memory_config_mm=ttnn.L1_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,  # fused all_gather_matmul requires Ring (Linear is asserted out)
            program_config=program_config,
            num_workers_per_link=1,
        )
        ttnn.synchronize_device(mesh_device, sub_device_ids=[worker_sub_device_id])

        mm_torch = ttnn.to_torch(mm_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        got = mm_torch[0:1].float()
        pcc = torch.corrcoef(torch.stack([got.flatten(), expected.flatten()]))[0, 1].item()
        assert pcc > 0.99, f"fused all_gather_matmul PCC too low: {pcc}"
    finally:
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()
