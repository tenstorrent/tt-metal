# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Bisect fused MM+RS correctness from the passing BH unit config toward cosmos3's
trunk usage (which produces PCC ~0.05 latents). Dimensions stepped: ring size
(cluster axis), shape/grid/workers, trace capture."""

import pytest
import torch
import ttnn

from tests.nightly.t3000.ccl.test_minimal_matmul_strided_reduce_scatter_async import (
    run_minimal_matmul_strided_reduce_scatter_impl,
)

DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize(
    "device_params, topology",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200000000}, ttnn.Topology.Ring)],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
@pytest.mark.parametrize(
    "M, K, N, mm_core_grid, num_workers_per_link, cluster_axis, enable_trace, mm_block_m, sub_h, sub_w",
    [
        # A: passing LTX shape/grid/workers, but ring size 8 (axis 1) instead of 4.
        (9472, 3456, 5120, ttnn.CoreCoord(12, 8), 5, 1, False, 256, 2, 1),
        # B: cosmos3 shape + power-clamped grid + derived workers, ring size 4 (axis 0).
        (22144, 3200, 5120, ttnn.CoreCoord(10, 8), 4, 0, False, 256, 2, 1),
        # C: cosmos3 everything, ring size 8, eager.
        (22144, 3200, 5120, ttnn.CoreCoord(10, 8), 4, 1, False, 256, 2, 1),
        # D: cosmos3 everything under trace capture+replay.
        (22144, 3200, 5120, ttnn.CoreCoord(10, 8), 4, 1, True, 256, 2, 1),
        # E/F/G: the other three trunk RowParallel shapes (to_out K=1024; und M=2720 has
        # ragged M-blocks: 85 tiles / M_block 4). Full-fused trunk latents inflate to
        # std 10 while gen-down_proj-only stays clean, so the culprit is among these.
        (22144, 1024, 5120, ttnn.CoreCoord(10, 8), 4, 1, False, 256, 2, 1),
        # E under trace: the trunk bisect isolated the fused corruption to this
        # shape; case D (K=3200) is trace-clean, so K=1024 has its own traced
        # coverage. Passes in isolation — the trunk corruption needs more context.
        (22144, 1024, 5120, ttnn.CoreCoord(10, 8), 4, 1, True, 256, 2, 1),
        (2720, 3200, 5120, ttnn.CoreCoord(10, 8), 4, 1, False, 128, 2, 2),
        (2720, 1024, 5120, ttnn.CoreCoord(10, 8), 4, 1, False, 128, 2, 2),
    ],
    ids=[
        "A_ring8_ltx",
        "B_ring4_cosmos3",
        "C_ring8_cosmos3",
        "D_ring8_cosmos3_trace",
        "E_gen_to_out",
        "E_gen_to_out_trace",
        "F_und_down_proj",
        "G_und_to_out",
    ],
)
@pytest.mark.timeout(900)
def test_mmrs_cosmos3_repro(
    mesh_device,
    topology,
    M,
    K,
    N,
    mm_core_grid,
    num_workers_per_link,
    cluster_axis,
    enable_trace,
    mm_block_m,
    sub_h,
    sub_w,
):
    _run_case(
        mesh_device,
        topology,
        M,
        K,
        N,
        mm_core_grid,
        num_workers_per_link,
        cluster_axis,
        enable_trace,
        mm_block_m=mm_block_m,
        sub_h=sub_h,
        sub_w=sub_w,
    )


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize(
    "device_params, topology",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200000000}, ttnn.Topology.Ring)],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
@pytest.mark.parametrize("enable_trace", [False, True], ids=["eager", "trace"])
@pytest.mark.timeout(900)
def test_mmrs_cosmos3_submesh(mesh_device, topology, enable_trace):
    """Cosmos3 runs the op on a 2x8 submesh of the 4x8 parent (cfg-parallel split on
    axis 0); the full-mesh cases pass while the trunk corrupts, so exercise the op's
    ring-neighbor resolution under a submesh."""
    submeshes = mesh_device.create_submeshes(ttnn.MeshShape(2, 8))
    _run_case(submeshes[0], topology, 22144, 3200, 5120, ttnn.CoreCoord(10, 8), 4, 1, enable_trace)


def _run_case(
    mesh_device,
    topology,
    M,
    K,
    N,
    mm_core_grid,
    num_workers_per_link,
    cluster_axis,
    enable_trace,
    mm_block_m=256,
    sub_h=2,
    sub_w=1,
    ops_per_trace=1,
):
    run_minimal_matmul_strided_reduce_scatter_impl(
        mesh_device,
        M=M,
        K=K,
        N=N,
        dim=3,
        num_links=2,
        input_dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mem_config_input=DRAM,
        mem_config_mm=DRAM,
        mem_config_rs=DRAM,
        topology=topology,
        enable_trace=enable_trace,
        # Production replays the captured graph 35x (one per denoise step). The
        # (now-fixed) semaphore wipe race hung at replay #4, so shallow replay
        # counts miss the regression this suite guards; see
        # minimal_ring_strided_reduce_scatter_async_{reader,writer}.cpp.
        num_iters=35 if enable_trace else 1,
        num_workers_per_link=num_workers_per_link,
        ops_per_trace=ops_per_trace,
        num_buffers_per_channel=None,
        mm_block_m=mm_block_m,
        mm_block_k=128,
        mm_block_n=256,
        subblock_h=sub_h,
        subblock_w=sub_w,
        mm_core_grid=mm_core_grid,
        chunk_width_in_mm_blocks=1,
        rs_core_grid_offset=ttnn.CoreCoord(0, 8),
        rs_mode="fused",
        cluster_axis=cluster_axis,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_acc=True,
        allowed_pcc=0.999,
    )


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize(
    "device_params, topology",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200000000}, ttnn.Topology.Ring)],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
@pytest.mark.timeout(900)
def test_mmrs_cosmos3_adjacent_trace(mesh_device, topology):
    """Two K=1024 instances back-to-back in one captured graph — the trunk
    to_out/to_add_out adjacency, where a later instance's matmul cores start
    while the earlier instance's RS cores still drain. Both instances verified.
    Passes — the trunk's K=1024 corruption needs context beyond this pairing."""
    _run_case(mesh_device, topology, 22144, 1024, 5120, ttnn.CoreCoord(10, 8), 4, 1, True, ops_per_trace=2)


def test_fused_mmrs_table_excludes_corrupting_shape():
    """(22144, 1024, 5120) passes every unit context (35-replay trace, submesh,
    two-instance adjacency) yet corrupts the 35-step trunk (visual smear, frame std
    46 vs 71 gold) — kept out of the fused table until a unit-level repro exists.
    See the E_gen_to_out_trace / adjacent_trace cases above for the exonerated contexts."""
    from models.tt_dit.utils.matmul import fused_mmrs_configs

    table = fused_mmrs_configs.get(ttnn.CoreCoord(10, 10), {})
    assert (22144, 3200, 5120) in table  # down_proj stays fused
    assert (22144, 1024, 5120) not in table


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize(
    "device_params, topology",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200000000}, ttnn.Topology.Ring)],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
@pytest.mark.timeout(900)
def test_strided_rs_batch2_trace(mesh_device, topology):
    """B=2 non-fused strided RS under 35 traced replays: exercises the per-batch
    batch_ready barrier decrement and the out_ready credit target growing across
    batches — the multi-batch half of the credit-decrement protocol, unreachable
    through the fused op (which requires B=1)."""
    from tests.nightly.t3000.ccl.test_minimal_matmul_strided_reduce_scatter_async import (
        create_global_semaphores,
    )

    B, M, N = 2, 4096, 5120
    cluster_axis = 1
    num_devices = tuple(mesh_device.shape)[cluster_axis]
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    mesh_device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
    sems = create_global_semaphores(mesh_device, all_cores, 0)
    barrier = ttnn.create_global_semaphore(mesh_device, all_cores, 0)

    torch.manual_seed(0)
    torch_input = torch.randn(B, 1, M, N, dtype=torch.float32)
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run_op():
        return ttnn.experimental.strided_reduce_scatter_async(
            tt_input,
            None,
            3,
            sems,
            barrier_semaphore=barrier,
            num_links=2,
            memory_config=DRAM,
            topology=topology,
            cluster_axis=cluster_axis,
            num_workers_per_link=4,
            num_buffers_per_channel=None,
            mm_cores_y=8,
            mm_block_ht=8,
            mm_block_wt=8,
            mm_N_full_block_wt=N // 32 // 10,
            chunk_width_in_mm_blocks=1,
        )

    run_op()
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    tt_out = run_op()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    for _ in range(35):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)

    # Inputs are replicated, so every device's scatter slice is num_devices * its
    # input chunk; batch content differs, so cross-batch credit mixups corrupt.
    slice_n = N // num_devices
    shards = ttnn.get_device_tensors(tt_out)
    rows, cols = tuple(mesh_device.shape)
    for dev in range(num_devices):
        out = ttnn.to_torch(shards[dev]).to(torch.float32)  # row 0 of the mesh
        ref = num_devices * torch_input[:, :, :, dev * slice_n : (dev + 1) * slice_n]
        pcc = torch.corrcoef(torch.stack([ref.flatten(), out.flatten()]))[0, 1].item()
        assert pcc >= 0.999, f"device {dev}: B=2 traced RS PCC {pcc:.6f}"
