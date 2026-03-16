# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""OLMo-3.1-32B QK-norm: portable tests on (8,4) Galaxy mesh.

Tests the distributed RMSNorm pipeline used for QK-norm in the OLMo model.
Runs on the full (8,4) Galaxy mesh with FABRIC_1D_RING and dispatch_core_axis=COL,
matching the exact model configuration. The all_gather uses cluster_axis=0
(row devices) with Linear topology.

Prefill: input [1, 1, seq_len, local_dim] per device, distributed norm across 8 row devices.
Decode: input [1, 1, 32, local_dim] per device, same distributed norm pipeline.
L1 sharded: per-device rms_norm with model's exact sharding/program configs.

Device lifecycle is managed manually with try/finally for guaranteed cleanup.
"""

import math
import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc

# ── OLMo-3.1-32B constants ──
HEAD_DIM = 128
N_Q_HEADS = 40
N_KV_HEADS = 8
N_TP = 8  # row devices
N_COLS = 4  # column devices
N_LOCAL_Q_HEADS = N_Q_HEADS // N_TP  # 5
N_LOCAL_Q_HEADS_PADDED = 8
N_LOCAL_KV_HEADS = N_KV_HEADS // N_TP  # 1
Q_FULL_DIM = N_Q_HEADS * HEAD_DIM  # 5120
K_FULL_DIM = N_KV_HEADS * HEAD_DIM  # 1024
Q_LOCAL_DIM = N_LOCAL_Q_HEADS * HEAD_DIM  # 640
K_LOCAL_DIM = N_LOCAL_KV_HEADS * HEAD_DIM  # 128
BATCH_PER_GROUP = 8
EPS = 1e-6
TILE = 32
CLUSTER_SHAPE = [8, 4]

HIFI2_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

# Decode core grids (from olmo_model_config.py)
DECODE_Q_NORM_CORES = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 1))})
DECODE_K_NORM_CORES = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(3, 0))})


# ── Helpers ──


def pytorch_rmsnorm(x, weight, eps=1e-6):
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    normed = x_f32 * torch.rsqrt(variance + eps)
    if weight is not None:
        normed = normed * weight.float()
    return normed.to(x.dtype)


def open_galaxy_mesh():
    """Open (8,4) Galaxy mesh with FABRIC_1D_RING and COL dispatch."""
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    dispatch_core_config = ttnn.DispatchCoreConfig(None, ttnn.DispatchCoreAxis.COL, ttnn.FabricTensixConfig.DISABLED)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(8, 4),
        dispatch_core_config=dispatch_core_config,
    )
    logger.debug(f"Opened (8,4) Galaxy mesh with {mesh_device.get_num_devices()} devices")
    return mesh_device


def close_galaxy_mesh(mesh_device):
    """Close mesh device and reset fabric. Safe if mesh_device is None."""
    if mesh_device is not None:
        for submesh in mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh_device)
        logger.debug("Closed mesh device")
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    logger.debug("Reset fabric to DISABLED")


def setup_prefill_sub_devices(mesh_device):
    """Setup sub-device manager for prefill (mirrors TtLlamaPrefetcherSetup prefill mode)."""
    all_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9))])
    worker_sub_device = ttnn.SubDevice([all_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])
    return worker_sub_device_id, all_crs


def setup_ccl_semaphores(mesh_device, sub_device_crs):
    """Create 2 double-buffered semaphore handles for all_gather_async (line topology)."""
    return [ttnn.create_global_semaphore(mesh_device, sub_device_crs, 0) for _ in range(2)]


def run_distributed_rmsnorm(mesh_device, tt_input, tt_weight, semaphores, barrier_semaphore, worker_sub_device_id):
    """pre → all_gather_async(cluster_axis=0) → post. Mirrors _olmo_qk_norm_all_gather."""
    tt_stats = ttnn.rms_norm_pre_all_gather(tt_input, compute_kernel_config=HIFI2_COMPUTE, dtype=ttnn.bfloat16)

    tt_stats_gathered = ttnn.experimental.all_gather_async(
        tt_stats,
        dim=3,
        cluster_axis=0,
        mesh_device=mesh_device,
        topology=ttnn.Topology.Linear,
        multi_device_global_semaphore=semaphores,
        barrier_semaphore=barrier_semaphore,
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        subdevice_id=worker_sub_device_id,
    )
    ttnn.deallocate(tt_stats)

    tt_output = ttnn.rms_norm_post_all_gather(
        tt_input,
        tt_stats_gathered,
        epsilon=EPS,
        weight=tt_weight,
        compute_kernel_config=HIFI2_COMPUTE,
    )
    return tt_output


# =============================================================================
# PREFILL: Distributed RMSNorm on (8,4) Galaxy
# Input [1, 1, seq_len, full_dim] → ShardTensor2dMesh(dims=(None, 3)) → local_dim per device
# Weight [1, 1, full_dim//32, 32] → ShardTensor2dMesh(dims=(2, None)) → local per row device
# all_gather on cluster_axis=0 (row devices)
# =============================================================================


@pytest.mark.parametrize("seq_len", [128, 2048])
@pytest.mark.parametrize(
    "full_dim",
    [Q_FULL_DIM, K_FULL_DIM],
    ids=["q_5120", "k_1024"],
)
def test_prefill_distributed_norm(seq_len, full_dim):
    """Prefill QK-norm on (8,4) Galaxy. Directly portable to the model."""
    mesh_device = None
    try:
        mesh_device = open_galaxy_mesh()
        torch.manual_seed(42)

        torch_input = torch.randn(1, 1, seq_len, full_dim, dtype=torch.bfloat16)
        torch_weight = torch.randn(full_dim, dtype=torch.bfloat16)
        torch_output = pytorch_rmsnorm(torch_input, torch_weight, eps=EPS)

        # Input sharded across row devices (dim 3), replicated across cols
        tt_input = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, None), mesh_shape=CLUSTER_SHAPE),
        )

        # Weight sharded across row devices (dim 2), replicated across cols
        weight_2d = torch_weight.view(1, 1, full_dim // TILE, TILE)
        tt_weight = ttnn.from_torch(
            weight_2d,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=CLUSTER_SHAPE),
        )

        worker_sub_device_id, sub_device_crs = setup_prefill_sub_devices(mesh_device)
        semaphores = setup_ccl_semaphores(mesh_device, sub_device_crs)
        barrier_semaphore = ttnn.create_global_semaphore(mesh_device, sub_device_crs, 0)
        ttnn.synchronize_device(mesh_device)

        tt_output = run_distributed_rmsnorm(
            mesh_device, tt_input, tt_weight, semaphores, barrier_semaphore, worker_sub_device_id
        )

        # Row devices each hold a shard of the output; cols are replicas
        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 1), mesh_shape=CLUSTER_SHAPE),
        )
        # Take col 0 (all cols identical), row concat gives full dim
        tt_out_col0 = tt_output_torch[:, 0:1, :, :]
        assert_with_pcc(torch_output, tt_out_col0, 0.999)
    finally:
        close_galaxy_mesh(mesh_device)


# =============================================================================
# DECODE DISTRIBUTED: Q with correction-factor weight, K without padding
# =============================================================================


Q_DECODE_CORRECTION = math.sqrt(Q_FULL_DIM / (N_LOCAL_Q_HEADS_PADDED * N_TP * HEAD_DIM))


def test_decode_q_distributed_norm_corrected():
    """Decode Q: distributed norm with padded heads and correction-factor weight on (8,4) Galaxy."""
    mesh_device = None
    try:
        mesh_device = open_galaxy_mesh()
        torch.manual_seed(42)
        padded_h = TILE
        padded_per_device = N_LOCAL_Q_HEADS_PADDED * HEAD_DIM  # 1024
        total_padded_dim = padded_per_device * N_TP  # 8192

        # Build full input: per row-device has 5 real + 3 zero-padded heads
        full_input = torch.zeros(1, 1, padded_h, total_padded_dim, dtype=torch.bfloat16)
        for dev in range(N_TP):
            dev_start = dev * padded_per_device
            real_data = torch.randn(1, 1, BATCH_PER_GROUP, Q_LOCAL_DIM, dtype=torch.bfloat16)
            full_input[:, :, :BATCH_PER_GROUP, dev_start : dev_start + Q_LOCAL_DIM] = real_data

        # Build corrected weight (matches llama_attention.py lines 401-434)
        base_weight = torch.randn(Q_FULL_DIM, dtype=torch.bfloat16)
        full_weight = torch.ones(total_padded_dim, dtype=torch.bfloat16)
        for dev in range(N_TP):
            dev_offset = dev * padded_per_device
            for h in range(N_LOCAL_Q_HEADS):
                global_head = dev * N_LOCAL_Q_HEADS + h
                w_start = global_head * HEAD_DIM
                local_start = dev_offset + h * HEAD_DIM
                full_weight[local_start : local_start + HEAD_DIM] = (
                    base_weight[w_start : w_start + HEAD_DIM] * Q_DECODE_CORRECTION
                )

        # PyTorch reference: variance over real dims only, apply corrected weight
        x_f32 = full_input.float()
        x_real = x_f32[..., :Q_FULL_DIM]
        variance = x_real.pow(2).mean(-1, keepdim=True)
        normed = x_f32 * torch.rsqrt(variance + EPS) * full_weight.float()
        torch_output = normed.to(torch.bfloat16)

        # Input sharded across row devices (dim 3), replicated across cols
        tt_input = ttnn.from_torch(
            full_input,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, None), mesh_shape=CLUSTER_SHAPE),
        )
        # Weight sharded across row devices (dim 2), replicated across cols
        weight_2d = full_weight.view(1, 1, total_padded_dim // TILE, TILE)
        tt_weight = ttnn.from_torch(
            weight_2d,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=CLUSTER_SHAPE),
        )

        worker_sub_device_id, sub_device_crs = setup_prefill_sub_devices(mesh_device)
        semaphores = setup_ccl_semaphores(mesh_device, sub_device_crs)
        barrier_semaphore = ttnn.create_global_semaphore(mesh_device, sub_device_crs, 0)
        ttnn.synchronize_device(mesh_device)

        tt_output = run_distributed_rmsnorm(
            mesh_device, tt_input, tt_weight, semaphores, barrier_semaphore, worker_sub_device_id
        )

        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 1), mesh_shape=CLUSTER_SHAPE),
        )
        tt_out_col0 = tt_output_torch[:, 0:1, :, :]
        assert_with_pcc(torch_output, tt_out_col0, 0.999)
    finally:
        close_galaxy_mesh(mesh_device)


def test_decode_k_distributed_norm():
    """Decode K: distributed norm, no padding (1 KV head × 128 per device) on (8,4) Galaxy."""
    mesh_device = None
    try:
        mesh_device = open_galaxy_mesh()
        torch.manual_seed(42)
        padded_h = TILE

        full_input = torch.zeros(1, 1, padded_h, K_FULL_DIM, dtype=torch.bfloat16)
        full_input[:, :, :BATCH_PER_GROUP, :] = torch.randn(1, 1, BATCH_PER_GROUP, K_FULL_DIM, dtype=torch.bfloat16)
        torch_weight = torch.randn(K_FULL_DIM, dtype=torch.bfloat16)
        torch_output = pytorch_rmsnorm(full_input, torch_weight, eps=EPS)

        # Input sharded across row devices (dim 3), replicated across cols
        tt_input = ttnn.from_torch(
            full_input,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, None), mesh_shape=CLUSTER_SHAPE),
        )
        # Weight sharded across row devices (dim 2), replicated across cols
        weight_2d = torch_weight.view(1, 1, K_FULL_DIM // TILE, TILE)
        tt_weight = ttnn.from_torch(
            weight_2d,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=CLUSTER_SHAPE),
        )

        worker_sub_device_id, sub_device_crs = setup_prefill_sub_devices(mesh_device)
        semaphores = setup_ccl_semaphores(mesh_device, sub_device_crs)
        barrier_semaphore = ttnn.create_global_semaphore(mesh_device, sub_device_crs, 0)
        ttnn.synchronize_device(mesh_device)

        tt_output = run_distributed_rmsnorm(
            mesh_device, tt_input, tt_weight, semaphores, barrier_semaphore, worker_sub_device_id
        )

        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 1), mesh_shape=CLUSTER_SHAPE),
        )
        tt_out_col0 = tt_output_torch[:, 0:1, :, :]
        assert_with_pcc(torch_output, tt_out_col0, 0.999)
    finally:
        close_galaxy_mesh(mesh_device)


# =============================================================================
# DECODE L1 SHARDED: per-device rms_norm with model's exact sharding configs
# Uses (8,4) mesh with ReplicateTensorToMesh (no fabric needed).
# =============================================================================


@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
def test_decode_q_norm_l1_sharded(mesh_device):
    """Decode Q local norm: L1 WIDTH_SHARDED on 4 cores, all 32 devices."""
    torch.manual_seed(42)

    n_hqd = N_LOCAL_Q_HEADS_PADDED * BATCH_PER_GROUP  # 64
    torch_input = torch.randn(1, 1, n_hqd, HEAD_DIM, dtype=torch.bfloat16)
    torch_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16)
    torch_output = pytorch_rmsnorm(torch_input, torch_weight, eps=EPS)

    q_norm_mem_cfg = ttnn.create_sharded_memory_config(
        shape=(n_hqd, HEAD_DIM // 4),
        core_grid=DECODE_Q_NORM_CORES,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    q_norm_prog_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[2, 2],
        subblock_w=1,
        block_h=n_hqd // TILE,
        block_w=1,
        inplace=False,
    )

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_input_sharded = ttnn.to_memory_config(tt_input, q_norm_mem_cfg)
    tt_weight = ttnn.from_torch(
        torch_weight.view(1, 1, HEAD_DIM // TILE, TILE),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_output = ttnn.rms_norm(
        tt_input_sharded,
        epsilon=EPS,
        weight=tt_weight,
        program_config=q_norm_prog_cfg,
        memory_config=q_norm_mem_cfg,
        compute_kernel_config=HIFI2_COMPUTE,
    )

    tt_output_dram = ttnn.to_memory_config(tt_output, ttnn.DRAM_MEMORY_CONFIG)
    tt_output_torch = ttnn.to_torch(tt_output_dram, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    assert_with_pcc(torch_output, tt_output_torch[:1], 0.999)


@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
def test_decode_k_norm_l1_sharded(mesh_device):
    """Decode K local norm: L1 WIDTH_SHARDED on 1 core, all 32 devices."""
    torch.manual_seed(42)

    padded_h = TILE
    torch_input = torch.zeros(1, 1, padded_h, HEAD_DIM, dtype=torch.bfloat16)
    torch_input[:, :, :BATCH_PER_GROUP, :] = torch.randn(1, 1, BATCH_PER_GROUP, HEAD_DIM, dtype=torch.bfloat16)
    torch_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16)
    torch_output = pytorch_rmsnorm(torch_input, torch_weight, eps=EPS)

    k_norm_mem_cfg = ttnn.create_sharded_memory_config(
        shape=(TILE, HEAD_DIM),
        core_grid=DECODE_K_NORM_CORES,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    k_norm_prog_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[1, 1],
        subblock_w=4,
        block_h=1,
        block_w=4,
        inplace=False,
    )

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_input_sharded = ttnn.to_memory_config(tt_input, k_norm_mem_cfg)
    tt_weight = ttnn.from_torch(
        torch_weight.view(1, 1, HEAD_DIM // TILE, TILE),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_output = ttnn.rms_norm(
        tt_input_sharded,
        epsilon=EPS,
        weight=tt_weight,
        program_config=k_norm_prog_cfg,
        memory_config=k_norm_mem_cfg,
        compute_kernel_config=HIFI2_COMPUTE,
    )

    tt_output_dram = ttnn.to_memory_config(tt_output, ttnn.DRAM_MEMORY_CONFIG)
    tt_output_torch = ttnn.to_torch(tt_output_dram, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    assert_with_pcc(torch_output, tt_output_torch[:1], 0.999)
