# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Minimal Llama 70B Attention decode test for comparison with OLMo"""

import torch
import pytest
import os
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.llama_attention import TtLlamaAttention
from models.demos.llama3_70b_galaxy.tt.model_config import TtModelArgs
from models.demos.llama3_70b_galaxy.tt.llama_common import get_rot_transformation_mat
from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL
from models.demos.llama3_70b_galaxy.tt.distributed_norm import DistributedNorm
from models.common.rmsnorm import RMSNorm


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size",
    (32,),
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
def test_llama_attention_decode(batch_size, mesh_device, reset_seeds, ensure_gc):
    """Test Llama 70B Attention decode path with LayerNorm - minimal test for comparison"""
    dtype = ttnn.bfloat8_b

    # Set environment variable to use dummy weights
    os.environ["HF_MODEL"] = "meta-llama/Llama-3.1-70B"

    logger.info("=" * 60)
    logger.info("Llama 70B Attention Decode Test (with LayerNorm)")
    logger.info("=" * 60)

    # Initialize Llama model args with dummy weights
    model_args = TtModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        dummy_weights=True,
        max_seq_len=2048,
    )
    model_args.n_layers = 1
    model_config = model_args.get_model_config()

    logger.info(
        f"Llama dimensions: dim={model_args.dim}, n_heads={model_args.n_heads}, n_kv_heads={model_args.n_kv_heads}"
    )
    logger.info(f"head_dim={model_args.head_dim}, batch_size={batch_size}")

    state_dict = model_args.load_state_dict()

    # Setup prefetcher and CCL for decode mode
    prefetcher_setup = TtLlamaPrefetcherSetup(mesh_device, n_tensors=0, n_layers=1, mode="decode")
    prefetcher_setup.create_global_cb()
    model_args.use_prefetcher = True
    mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])
    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id, mode="decode")

    paged_attention_config = None

    # Create transformation matrices for decode
    transformation_mat_torch = get_rot_transformation_mat(model_args.head_dim)
    transformation_mats_decode = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    transformation_mats = {"decode": transformation_mats_decode}

    # Create attention norm (RMSNorm)
    logger.info("Creating DistributedNorm (attention_norm)...")
    # Debug: print memory configs
    ring_memcfg = model_config.get("SHARDED_ATTN_INPUT_RING_MEMCFG")
    logger.info(f"SHARDED_ATTN_INPUT_RING_MEMCFG: {ring_memcfg}")
    if ring_memcfg is not None:
        logger.info(f"  memory_layout: {ring_memcfg.memory_layout}")
        logger.info(f"  shard_spec: {ring_memcfg.shard_spec}")
    logger.info(f"SHARDED_ATTN_INPUT_MEMCFG: {model_config.get('SHARDED_ATTN_INPUT_MEMCFG')}")
    logger.info(f"SHARDED_NORM_ATTN_PRGM_CFG: {model_config.get('SHARDED_NORM_ATTN_PRGM_CFG')}")

    attention_norm = DistributedNorm(
        RMSNorm(
            device=mesh_device,
            dim=model_args.dim,
            state_dict=state_dict,
            state_dict_prefix=model_args.get_state_dict_prefix("", 0),
            weight_cache_path=None,
            weight_dtype=ttnn.bfloat16,
            weight_key="attention_norm",
            is_distributed=model_args.is_distributed_norm,
            sharded_program_config=model_config["SHARDED_NORM_ATTN_PRGM_CFG"],
            sharded_output_config=model_config["SHARDED_ATTN_INPUT_MEMCFG"],
            output_mem_config=model_config["SHARDED_ATTN_INPUT_RING_MEMCFG"],
        ),
        model_args,
        tt_ccl=tt_ccl,
        ccl_topology=model_config["CCL_TOPOLOGY"],
    )

    # Debug: print norm and buffer configs
    logger.info(f"RMSNorm.output_mem_config: {attention_norm.norm.output_mem_config}")
    logger.info(f"DistributedNorm.gather_in_mem_cfg: {attention_norm.gather_in_mem_cfg}")
    logger.info(f"DistributedNorm.ln_prg_cfg: {attention_norm.ln_prg_cfg}")
    logger.info(f"model_args.decode_ln_core_grid: {getattr(model_args, 'decode_ln_core_grid', 'NOT SET')}")
    logger.info(f"tt_ccl.all_gather_buffers keys: {list(tt_ccl.all_gather_buffers.keys())}")
    ln_buffer = tt_ccl.all_gather_buffers.get("LAYERNORM", None)
    logger.info(f"LAYERNORM buffer exists: {ln_buffer is not None}")
    if ln_buffer is not None:
        logger.info(f"LAYERNORM buffer memory_config: {ln_buffer.memory_config()}")

    logger.info("Creating TtLlamaAttention...")
    tt_attention = TtLlamaAttention(
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        transformation_mats=transformation_mats,
        configuration=model_args,
        paged_attention_config=paged_attention_config,
        use_paged_kv_cache=False,
        prefetcher_setup=prefetcher_setup,
        tt_ccl=tt_ccl,
    )

    # Create random input tensor for decode
    seq_len = 1
    pt_decode_input = (torch.rand(batch_size, seq_len, model_args.dim) * 2) - 1

    logger.info(f"Input shape: {pt_decode_input.shape}")

    logger.info("Running Llama Attention decode...")
    for i in range(3):
        decode_input = model_args.prepare_residual_tensor_decode(
            pt_decode_input.clone(),
            model_config["DECODE_RESIDUAL_MEMCFG"],
        )
        logger.info(f"decode_input memory_config: {decode_input.memory_config()}")

        # Apply LayerNorm
        attn_input, _ = attention_norm(decode_input, None, mode="decode")

        # Run attention decode
        tt_output = tt_attention.forward(
            attn_input,
            current_pos=i,
            rot_mats=None,
            user_id=0,
            mode="decode",
            page_table=None,
        )

        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )

        logger.info(f"Iteration {i+1}: Output shape = {tt_output_torch.shape}")

    tt_ccl.close()
    logger.info("=" * 60)
    logger.info("Llama 70B Attention Decode Test PASSED!")
    logger.info("=" * 60)
