# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
import torch
import torch.nn as nn
from loguru import logger
from models.demos.llama3_70b_galaxy.tt.model_config import (
    get_core_ranges,
)
from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.common.rmsnorm import RMSNorm
from models.common.utility_functions import (
    torch_random,
    comp_pcc,
    comp_allclose,
)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_qwen3_tg_qk_norm(
    mesh_device,
):
    """
    Test qk_norm operation for Qwen3 model.
    """
    torch.manual_seed(0)
    galaxy_type = "6U" if ttnn.GetNumPCIeDevices() == 32 else "4U"

    # Setup prefetcher
    prefetcher_setup = TtLlamaPrefetcherSetup(
        mesh_device,
        n_tensors=2,
        n_layers=1,
    )
    mesh_device.set_sub_device_stall_group(
        [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
    )

    sub_core_grids = ttnn.CoreRangeSet(  # 50 cores total
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )

    qkv_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            sub_core_grids,
            [32, 128],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    _, _, _, pf_receiver_cores_list, _, _, _, _ = get_core_ranges(12, 2, False)
    pf_mm_out_core_range_set = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(x, y),
                ttnn.CoreCoord(x, y),
            )
            for x, y in pf_receiver_cores_list
        ]
    )

    SHARDED_QKV_OUT_RING_MEMCFG = ttnn.create_sharded_memory_config(
        shape=(32, 12288 // 8 // 24),  # (32, 64)
        core_grid=pf_mm_out_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Create random input tensors
    torch_xqkv_tensor = torch_random([1, 1, 32, 1280], -1, 1, dtype=torch.bfloat16)
    # Convert to ttnn tensors with appropriate memory configuration
    xqkv_tensor = ttnn.from_torch(
        torch_xqkv_tensor,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=SHARDED_QKV_OUT_RING_MEMCFG,
    )

    # Set up persistent buffers to mimic tt_ccl settings
    RS_CREATE_HEADS_PACKET_WORKER_CRS = ttnn.CoreRangeSet(
        [  # 5 cores total
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 0)),
            ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 1)),
        ]
    )
    RS_CREATE_HEADS_INTERIM_MEMCFG = ttnn.create_sharded_memory_config(
        shape=(32, 512),
        core_grid=RS_CREATE_HEADS_PACKET_WORKER_CRS,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    cluster_shape = (8, 4)
    num_pages_per_packet = 4
    shard_height = 32

    # (*cluster_shape, shard_height, cluster_shape[1] * num_pages_per_packet * 32 * 5)
    persistent_buffers = ttnn.from_torch(
        torch.zeros((*cluster_shape, 32, 512 * 5)),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=RS_CREATE_HEADS_INTERIM_MEMCFG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(8, 4)),
    )

    global_semaphore = ttnn.create_global_semaphore(mesh_device, sub_core_grids, 0)

    # Perform rs_create_heads using ttnn
    (
        q_heads_pre_rot_1BQD,
        k_heads_pre_rot_1BKD,
        v_heads_1BKD,
    ) = ttnn.experimental.llama_rs_create_heads(
        input_tensor=xqkv_tensor,
        intermediate_packet_buffer=persistent_buffers,
        dim=3,
        cross_device_semaphore=global_semaphore,
        subdevice_id=prefetcher_setup.worker_sub_device_id,
        cluster_axis=1,
        mesh_device=mesh_device,
        topology={"6U": ttnn.Topology.Ring, "4U": ttnn.Topology.Linear}.get(galaxy_type),
        num_links=1,
        num_heads=8,
        num_kv_heads=1,
        memory_config=qkv_memory_config,
        qkv_memory_config=qkv_memory_config,
        use_noc1_only=False,
        use_optimal_ccl_for_llama=False,
    )

    q_norm_weights = torch.randn([1, 128])  # [1, 128] ==> [1 (32), 32 x 4]
    k_norm_weights = torch.randn([1, 128])
    state_dict = {"q_norm.weight": q_norm_weights, "k_norm.weight": k_norm_weights}

    norm_weight_mem_cfg = ttnn.create_sharded_memory_config(
        shape=(32, 32),
        core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 1))]),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    reshape_intermediate_mem_cfg = ttnn.create_sharded_memory_config(
        shape=(64, 128),  # [1, 8, 8 (32), 128] ==> *[1, 1, 64, 128]* ==> [1, 1, 64, 32 * 4 = 128]
        core_grid=ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))
            ]  # This captures the fact that we are using 1 core (height sharded)
        ),  # resharding tensor to 1 core
        strategy=ttnn.ShardStrategy.HEIGHT,  # Literally stating to the device to perform width sharding
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    reshape_output_mem_cfg = ttnn.create_sharded_memory_config(
        shape=(64, 32),  # [1, 8, 8, 128] ==> [1, 1, 64, 128] ==> *[1, 1, 64, 32 * 4 = 128]*
        core_grid=ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 1))]
        ),  # resharding tensor to 1 core
        strategy=ttnn.ShardStrategy.WIDTH,  # Literally stating to the device to perform width sharding
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    block_w = 128 // 4 // 32
    # Find largest value <= 4 that evenly divides block_w
    subblock_w = 1
    while subblock_w > 0:
        if block_w % subblock_w == 0:
            break
        subblock_w -= 1
    norm_program_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[2, 2],
        subblock_w=subblock_w,
        block_h=2,  # 64 // 32
        block_w=block_w,
        inplace=False,
    )

    q_norm = RMSNorm(
        device=mesh_device,
        dim=128,
        state_dict=state_dict,
        state_dict_prefix=None,
        weight_dtype=ttnn.bfloat16,
        weight_key="q_norm",
        sharded_program_config=norm_program_cfg,
        sharded_output_config=reshape_output_mem_cfg,
    )
    k_norm = RMSNorm(
        device=mesh_device,
        dim=128,
        state_dict=state_dict,
        state_dict_prefix=None,
        weight_dtype=ttnn.bfloat16,
        weight_key="k_norm",
        sharded_program_config=norm_program_cfg,
        sharded_output_config=reshape_output_mem_cfg,
    )

    # [1, 8, 8, 128] ==> [1, 1, 64, 128] ==> [1, 1, 64, 32 x 4]
    rm_mem_cfg_q = q_heads_pre_rot_1BQD.memory_config()
    rm_mem_cfg_k = k_heads_pre_rot_1BKD.memory_config()

    q_heads_pre_rot_1BQD = ttnn.to_memory_config(q_heads_pre_rot_1BQD, memory_config=reshape_intermediate_mem_cfg)
    k_heads_pre_rot_1BKD = ttnn.to_memory_config(k_heads_pre_rot_1BKD, memory_config=reshape_intermediate_mem_cfg)

    q_heads_pre_rot_1BQD = ttnn.reshape(q_heads_pre_rot_1BQD, [1, 1, 64, 128])
    k_heads_pre_rot_1BKD = ttnn.reshape(k_heads_pre_rot_1BKD, [1, 1, 64, 128])

    q_heads_pre_rot_1BQD = ttnn.to_layout(q_heads_pre_rot_1BQD, ttnn.TILE_LAYOUT)
    k_heads_pre_rot_1BKD = ttnn.to_layout(k_heads_pre_rot_1BKD, ttnn.TILE_LAYOUT)

    q_heads_intermediate_after_reshape_mem_cfg = q_heads_pre_rot_1BQD.memory_config()
    k_heads_intermediate_after_reshape_mem_cfg = k_heads_pre_rot_1BKD.memory_config()

    q_heads_pre_rot_1BQD = ttnn.to_memory_config(q_heads_pre_rot_1BQD, memory_config=reshape_output_mem_cfg)
    k_heads_pre_rot_1BKD = ttnn.to_memory_config(k_heads_pre_rot_1BKD, memory_config=reshape_output_mem_cfg)

    # [1, 1, 64, 32 x 4]

    q_heads_pre_rot_1BQD = q_norm(q_heads_pre_rot_1BQD, mode="decode", in_sharded=True, out_sharded=True)
    k_heads_pre_rot_1BKD = k_norm(k_heads_pre_rot_1BKD, mode="decode", in_sharded=True, out_sharded=True)

    q_heads_pre_rot_1BQD = ttnn.to_memory_config(
        q_heads_pre_rot_1BQD, memory_config=q_heads_intermediate_after_reshape_mem_cfg
    )
    k_heads_pre_rot_1BKD = ttnn.to_memory_config(
        k_heads_pre_rot_1BKD, memory_config=k_heads_intermediate_after_reshape_mem_cfg
    )

    q_heads_pre_rot_1BQD = ttnn.to_layout(q_heads_pre_rot_1BQD, ttnn.ROW_MAJOR_LAYOUT)
    k_heads_pre_rot_1BKD = ttnn.to_layout(k_heads_pre_rot_1BKD, ttnn.ROW_MAJOR_LAYOUT)

    q_heads_pre_rot_1BQD = ttnn.reshape(q_heads_pre_rot_1BQD, [1, 8, 8, 128])
    k_heads_pre_rot_1BKD = ttnn.reshape(k_heads_pre_rot_1BKD, [1, 8, 8, 128])

    q_heads_pre_rot_1BQD = ttnn.to_memory_config(q_heads_pre_rot_1BQD, memory_config=rm_mem_cfg_q)
    k_heads_pre_rot_1BKD = ttnn.to_memory_config(k_heads_pre_rot_1BKD, memory_config=rm_mem_cfg_k)

    # Convert ttnn results back to torch for comparison
    ttnn_q_heads_normalized = ttnn.to_torch(
        q_heads_pre_rot_1BQD, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=(8, 4))
    )
    ttnn_k_heads_normalized = ttnn.to_torch(
        k_heads_pre_rot_1BKD, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 2), mesh_shape=(8, 4))
    )

    ttnn_q_heads_normalized = ttnn_q_heads_normalized[:1, :, :, :]
    ttnn_k_heads_normalized = ttnn_k_heads_normalized[:1, :, ::8, :].reshape(1, 32, 1, 128)

    # ===== TORCH REFERENCE IMPLEMENTATION =====

    # Create torch RMSNorm modules matching the ttnn implementation
    class TorchRMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def _norm(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        def forward(self, x):
            output = self._norm(x.float()).type_as(x)
            return output * self.weight

    # Initialize torch RMSNorm with the same weights as ttnn
    torch_q_norm = TorchRMSNorm(128)
    torch_k_norm = TorchRMSNorm(128)

    # Set the weights to match the ttnn implementation
    with torch.no_grad():
        torch_q_norm.weight.copy_(q_norm_weights.squeeze())
        torch_k_norm.weight.copy_(k_norm_weights.squeeze())

    torch_xq = torch_xqkv_tensor[:, :, :, : 8 * 128].reshape(1, 32, 8, 128)  # [1, 32, 8, 128]
    torch_xk = torch_xqkv_tensor[:, :, :, 8 * 128 : (9 * 128)]  # [1, 1, 32, 128]

    # Apply QK normalization in torch (same as in Attention.forward)
    torch_q_heads_normalized = torch_q_norm(torch_xq)  # [1, 8, 8, 128]
    torch_k_heads_normalized = torch_k_norm(torch_xk)  # [1, 8, 8, 128]
    idx = torch.arange(32).reshape(4, 8).T.flatten()
    torch_k_heads_normalized = torch_k_heads_normalized[:, :, idx, :].view(1, 32, 1, 128)

    # Compare results
    logger.info("Comparing Q heads normalization results")
    q_pcc = comp_pcc(torch_q_heads_normalized, ttnn_q_heads_normalized)
    q_allclose = comp_allclose(torch_q_heads_normalized, ttnn_q_heads_normalized)
    logger.info(f"Q heads PCC: {q_pcc}, AllClose: {q_allclose}")

    logger.info("Comparing K heads normalization results")
    k_pcc = comp_pcc(torch_k_heads_normalized, ttnn_k_heads_normalized)
    k_allclose = comp_allclose(torch_k_heads_normalized, ttnn_k_heads_normalized)
    logger.info(f"K heads PCC: {k_pcc}, AllClose: {k_allclose}")

    # Assert that results are close enough
    assert q_pcc[1] > 0.99, f"Q heads PCC {q_pcc[1]} is too low"
    assert k_pcc[1] > 0.99, f"K heads PCC {k_pcc[1]} is too low"

    logger.info("QK norm test passed successfully!")
