# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.demo.model_pipeline import ModelPipeline
from models.demos.deepseek_v3_b1.demo.pipeline import create_fabric_router_config
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import FlashMLADecode
from models.demos.deepseek_v3_b1.tests.unit_tests.test_pre_sdpa import deinterleave_kv_cache


@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "position_id",
    [4096],
)
def test_decode_kv_cache_metadata(mesh_device, device_params, position_id):
    torch.manual_seed(0)
    mesh_rows = mesh_device.shape[0]
    mesh_cols = mesh_device.shape[1]
    model_pipeline = ModelPipeline(mesh_device, weights_mode="synthetic")
    flash_mla_program_config = FlashMLADecode.ProgramConfig(k_chunk_size=128)
    kvpe_dim = 576
    cache_shape = (flash_mla_program_config.max_kv_cache_slots, 1, flash_mla_program_config.max_seq_len, kvpe_dim)

    dcs = flash_mla_program_config.device_chunk_size
    num_sp = flash_mla_program_config.sp_dim

    torch_kv_cache = torch.zeros(cache_shape, dtype=torch.bfloat16)
    torch_kv_cache[:, :, :position_id, :] = torch.randn(1, 1, position_id, kvpe_dim, dtype=torch.bfloat16)
    torch_kv_cache_shuffled = deinterleave_kv_cache(torch_kv_cache, dcs, num_sp)

    grid = flash_mla_program_config.grid
    kv_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, flash_mla_program_config.k_chunk_size, kvpe_dim],
        grid=grid.optimal_dram_grid(),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    kv_mem_config = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=kv_nd_shard_spec,
    )
    num_chunks = flash_mla_program_config.max_seq_len // flash_mla_program_config.k_chunk_size
    num_banks = len(grid.OPTIMAL_DRAM_BANK_ORDER)
    logger.info(
        f"KV cache: ND sharded, DRAM banks: {num_banks} (optimal order: {grid.OPTIMAL_DRAM_BANK_ORDER}), chunks: {num_chunks}, shard_shape: [1, 1, {flash_mla_program_config.k_chunk_size}, {kvpe_dim}]"
    )

    ttnn_kv_cache = ttnn.from_torch(
        torch_kv_cache_shuffled,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=kv_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(mesh_rows, mesh_cols), dims=(2, None)),
    )

    kv_cache_metadata = model_pipeline.get_kv_cache_metadata(0, 0, 0, ttnn_kv_cache)
    print(kv_cache_metadata)
