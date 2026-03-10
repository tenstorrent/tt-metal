# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
OLMo-3.1-32B Prefill Performance Test.

Run with:
    export HF_MODEL=~/models/models--allenai--Olmo-3.1-32B-Think
    pytest models/demos/llama3_70b_galaxy/tests/test_olmo_prefill_perf.py -v
"""

import time
import torch
import pytest
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.llama_common import (
    PagedAttentionConfig,
    precompute_freqs_yarn,
    gather_cos_sin,
)
from models.demos.llama3_70b_galaxy.tt.olmo_model_config import TtOlmoModelArgs
from models.demos.llama3_70b_galaxy.tt.llama_model import TtTransformer


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],  # Galaxy TG mesh shape
    indirect=True,
)
@pytest.mark.parametrize(
    "paged_attention",
    (True,),
    ids=("paged_attention",),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 64, "page_max_num_blocks": 4096}],
)
@pytest.mark.parametrize(
    "seq_len",
    (128, 512, 1024, 2048),
    ids=["128", "512", "1k", "2k"],
)
@pytest.mark.parametrize(
    "num_layers",
    (1, 64),
    ids=["1layer", "64layers"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
def test_olmo_prefill_perf(
    paged_attention,
    page_params,
    seq_len,
    num_layers,
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    """Test OLMo prefill performance."""
    dtype = ttnn.bfloat8_b
    batch_size = 1
    max_seq_len = 8192  # OLMo supports up to 64k but use 8k for testing

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

    # Setup paged attention
    paged_attention_config = None
    page_table_tt = None
    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    # Load TT model
    logger.info("Loading TT model...")
    tt_model = TtTransformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
        mode="prefill",
        allocate_prefill_buffers=True,
    )
    logger.info("Finished loading TT model.")

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

    # Warmup run
    logger.info("Running warmup...")
    _ = tt_model(
        tt_prefill_input,
        current_pos=None,
        rot_mats=rot_mats,
        user_id=0,
        mode="prefill",
        page_table=page_table_tt,
    )
    ttnn.synchronize_device(mesh_device)

    # Performance measurement
    num_iterations = 5
    logger.info(f"Running {num_iterations} iterations for performance measurement...")

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        tt_out = tt_model(
            tt_prefill_input,
            current_pos=None,
            rot_mats=rot_mats,
            user_id=0,
            mode="prefill",
            page_table=page_table_tt,
        )
        ttnn.synchronize_device(mesh_device)
    end_time = time.perf_counter()

    # Calculate metrics
    total_time = end_time - start_time
    avg_time_per_iter = total_time / num_iterations
    tokens_per_second = seq_len / avg_time_per_iter

    logger.info(f"\n{'='*60}")
    logger.info(f"OLMo Prefill Performance Results")
    logger.info(f"{'='*60}")
    logger.info(f"  Sequence Length: {seq_len}")
    logger.info(f"  Number of Layers: {num_layers}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Average Latency: {avg_time_per_iter*1000:.2f} ms")
    logger.info(f"  Tokens/Second: {tokens_per_second:.2f}")
    logger.info(f"{'='*60}\n")

    # Basic sanity check - ensure output is valid
    tt_output_torch = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )
    assert not torch.isnan(tt_output_torch).any(), "Output contains NaN"
    assert not torch.isinf(tt_output_torch).any(), "Output contains Inf"

    logger.info("OLMo Prefill Performance Test PASSED!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
