# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
OLMo-3.1-32B Decoder Prefill Performance Test.

Tests decoder blocks directly without full model wrapper to measure prefill performance.

Run with:
    export HF_MODEL=~/models/models--allenai--Olmo-3.1-32B-Think
    pytest models/demos/llama3_70b_galaxy/tests/test_olmo_decoder_prefill.py -v
"""

import time
import torch
import pytest
from loguru import logger
import ttnn
from tqdm import tqdm
from models.demos.llama3_70b_galaxy.tt.llama_common import (
    precompute_freqs_yarn,
    gather_cos_sin,
    get_rot_transformation_mat,
)
from models.demos.llama3_70b_galaxy.tt.olmo_model_config import TtOlmoModelArgs
from models.demos.llama3_70b_galaxy.tt.llama_decoder import TtTransformerBlock
from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],  # Galaxy TG mesh shape
    indirect=True,
)
@pytest.mark.parametrize(
    "paged_attention",
    (False,),  # Use standard KV cache for simplicity
    ids=("standard_kv",),
)
@pytest.mark.parametrize(
    "page_params",
    [{}],
)
@pytest.mark.parametrize(
    "seq_len",
    (128, 512, 1024, 2048, 4096),
    ids=["128", "512", "1k", "2k", "4k"],
)
@pytest.mark.parametrize(
    "num_layers",
    (1, 8, 64),
    ids=["1layer", "8layers", "64layers"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
def test_olmo_decoder_prefill(
    paged_attention,
    page_params,
    seq_len,
    num_layers,
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    """Test OLMo decoder prefill performance."""
    dtype = ttnn.bfloat8_b
    batch_size = 1
    max_seq_len = 8192

    # Load OLMo model config
    model_args = TtOlmoModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
    )
    model_args.n_layers = num_layers
    model_args.use_prefetcher = False

    logger.info(f"OLMo Config: layers={num_layers}, seq_len={seq_len}, batch={batch_size}")
    logger.info(f"  dim={model_args.dim}, n_heads={model_args.n_heads}, n_kv_heads={model_args.n_kv_heads}")

    # Load state dict
    state_dict = model_args.load_state_dict()

    # Setup paged attention (disabled for simplicity)
    paged_attention_config = None

    # Setup prefetcher and CCL for prefill mode
    prefetcher_setup = TtLlamaPrefetcherSetup(mesh_device, n_tensors=0, n_layers=num_layers, mode="prefill")
    mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])
    tt_ccl = TT_CCL(
        mesh_device,
        model_args,
        prefetcher_setup.worker_sub_device_id,
        mode="prefill",
        is_qwen=False,
        is_olmo=True,
    )

    # Setup RoPE transformation matrices
    head_dim = model_args.head_dim
    prefill_trans_mat_torch = get_rot_transformation_mat(dhead=head_dim)
    transformation_mat_prefill = ttnn.from_torch(
        prefill_trans_mat_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    transformation_mats = {"decode": transformation_mat_prefill, "prefill": transformation_mat_prefill}

    # Create decoder blocks
    logger.info(f"Loading {num_layers} decoder layers...")
    layers = [
        TtTransformerBlock(
            args=model_args,
            mesh_device=mesh_device,
            dtype=dtype,
            state_dict=state_dict,
            layer_num=i,
            n_layers=num_layers,
            weight_cache_path=model_args.weight_cache_path(dtype),
            transformation_mats=transformation_mats,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=False,  # Use standard KV cache
            prefetcher_setup=prefetcher_setup,
            tt_ccl=tt_ccl,
        )
        for i in tqdm(range(num_layers))
    ]
    logger.info("Finished loading decoder layers.")

    # Prepare YaRN RoPE
    ttnn_cos, ttnn_sin, _ = precompute_freqs_yarn(
        dim=model_args.head_dim,
        end=max_seq_len * 2,
        theta=model_args.rope_theta,
        scaling_factor=model_args.rope_scaling_factor,
        original_max_position_embeddings=model_args.original_max_position_embeddings,
        beta_fast=model_args.yarn_beta_fast,
        beta_slow=model_args.yarn_beta_slow,
        attention_factor=model_args.yarn_attention_factor,
    )
    position_ids = torch.arange(seq_len)
    cos_gathered, sin_gathered = gather_cos_sin(position_ids, ttnn_cos, ttnn_sin)

    rot_mats = [
        ttnn.from_torch(
            cos_gathered,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
        ttnn.from_torch(
            sin_gathered,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
    ]

    # Create random input
    torch.manual_seed(42)
    pt_prefill_input = torch.randn(batch_size, seq_len, model_args.dim)
    tt_prefill_input = model_args.prepare_residual_tensor_prefill(pt_prefill_input)

    # Debug: Check input tensor properties
    logger.info(f"Input tensor shape: {tt_prefill_input.shape}")
    logger.info(f"Input tensor dtype: {tt_prefill_input.dtype}")
    logger.info(f"Input tensor memory_config: {tt_prefill_input.memory_config()}")
    logger.info(f"Input tensor layout: {tt_prefill_input.layout}")

    # Debug: Check if norm weight_distributed exists
    layer = layers[0]
    if hasattr(layer.attention_norm.norm, "weight_distributed"):
        logger.info(
            f"attention_norm.norm.weight_distributed exists: {layer.attention_norm.norm.weight_distributed.shape}"
        )
    else:
        logger.error("attention_norm.norm.weight_distributed MISSING!")
    logger.info(f"attention_norm.norm.is_distributed: {layer.attention_norm.norm.is_distributed}")
    logger.info(f"attention_norm.tt_ccl: {layer.attention_norm.tt_ccl}")

    # Warmup run
    logger.info("Running warmup...")
    x = tt_prefill_input
    h = None  # h=None for first layer

    # Test direct call to tt_distributed_rmsnorm with the same input
    from models.demos.llama3_70b_galaxy.tt.llama_ccl import tt_distributed_rmsnorm

    logger.info("Testing direct tt_distributed_rmsnorm call...")
    try:
        direct_out, _ = tt_distributed_rmsnorm(
            x,
            epsilon=1e-6,
            gamma=layers[0].attention_norm.norm.weight_distributed,
            mesh_device=mesh_device,
            compute_kernel_config=layers[0].attention_norm.ln_cfg,
            tt_ccl=tt_ccl,
        )
        logger.info(f"Direct tt_distributed_rmsnorm SUCCESS! Output shape: {direct_out.shape}")
        direct_out.deallocate(True)
    except Exception as e:
        logger.error(f"Direct tt_distributed_rmsnorm FAILED: {e}")

    # Now test through the layer
    logger.info("Testing through layer.attention_norm...")
    try:
        norm_out, _ = layers[0].attention_norm(x, None, "prefill")
        logger.info(f"layer.attention_norm SUCCESS! Output shape: {norm_out.shape}")
    except Exception as e:
        logger.error(f"layer.attention_norm FAILED: {e}")
        raise

    # Step-by-step layer forward for debugging
    # Re-create input to avoid reusing deallocated tensor
    tt_step_input = model_args.prepare_residual_tensor_prefill(pt_prefill_input)
    logger.info("Step-by-step layer forward...")
    layer = layers[0]
    mode = "prefill"

    # Step 1: attention_norm
    logger.info("Step 1: attention_norm...")
    attn_in_sharded, _ = layer.attention_norm(tt_step_input, None, mode)
    logger.info(f"  attn_in_sharded shape: {attn_in_sharded.shape}")
    h = tt_step_input

    # Step 2: attention forward
    logger.info("Step 2: attention.forward...")
    try:
        attn_out = layer.attention.forward(
            attn_in_sharded,
            current_pos=None,
            rot_mats=rot_mats,
            user_id=0,
            mode=mode,
            page_table=None,
            chunk_page_table=None,
            chunk_start_idx=None,
            kv_cache=None,
            batch_size=1,
        )
        logger.info(f"  attn_out shape: {attn_out.shape}")
    except Exception as e:
        logger.error(f"  attention.forward FAILED: {e}")
        raise

    # Step 3: add residual for prefill
    logger.info("Step 3: h = ttnn.add(tt_step_input, attn_out)...")
    h = ttnn.add(tt_step_input, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_step_input.deallocate(True)
    logger.info(f"  h shape: {h.shape}")

    # Step 4: ff_norm
    logger.info("Step 4: ff_norm...")
    try:
        ff_in_sharded, _ = layer.ff_norm(h, None, mode)
        logger.info(f"  ff_in_sharded shape: {ff_in_sharded.shape}")
    except Exception as e:
        logger.error(f"  ff_norm FAILED: {e}")
        raise

    # Step 5: feed_forward
    logger.info("Step 5: feed_forward.forward...")
    try:
        ff_out = layer.feed_forward.forward(ff_in_sharded, mode, batch_size=1)
        logger.info(f"  ff_out shape: {ff_out.shape}")
    except Exception as e:
        logger.error(f"  feed_forward.forward FAILED: {e}")
        raise

    # Step 6: final add
    logger.info("Step 6: ttnn.add(ff_out, h)...")
    out = ttnn.add(ff_out, h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    h.deallocate(True)
    logger.info(f"  out shape: {out.shape}")

    x = out
    ttnn.synchronize_device(mesh_device)
    logger.info("Step-by-step layer forward COMPLETE!")

    # Performance measurement
    num_iterations = 5
    logger.info(f"Running {num_iterations} iterations for performance measurement...")

    start_time = time.perf_counter()
    for iter_idx in range(num_iterations):
        # Re-create input tensor for each iteration (forward deallocates it)
        x = model_args.prepare_residual_tensor_prefill(pt_prefill_input)
        h = None
        for layer in layers:
            x, h = layer(x, h, current_pos=None, rot_mats=rot_mats, user_id=0, mode="prefill")
        ttnn.synchronize_device(mesh_device)
        if iter_idx < num_iterations - 1:
            x.deallocate(True)  # Deallocate intermediate outputs
    end_time = time.perf_counter()

    # Calculate metrics
    total_time = end_time - start_time
    avg_time_per_iter = total_time / num_iterations
    tokens_per_second = seq_len / avg_time_per_iter

    logger.info(f"\n{'='*60}")
    logger.info(f"OLMo Decoder Prefill Performance Results")
    logger.info(f"{'='*60}")
    logger.info(f"  Sequence Length: {seq_len}")
    logger.info(f"  Number of Layers: {num_layers}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Average Latency: {avg_time_per_iter*1000:.2f} ms")
    logger.info(f"  Tokens/Second: {tokens_per_second:.2f}")
    logger.info(f"{'='*60}\n")

    # Basic sanity check
    tt_output_torch = ttnn.to_torch(
        x,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )
    assert not torch.isnan(tt_output_torch).any(), "Output contains NaN"
    assert not torch.isinf(tt_output_torch).any(), "Output contains Inf"

    tt_ccl.close()

    logger.info("OLMo Decoder Prefill Performance Test PASSED!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
