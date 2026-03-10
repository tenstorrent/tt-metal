# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
OLMo-3.1-32B Decode Test.

Tests decode mode with KV cache and sliding window attention.

Run with:
    export HF_MODEL=~/models/models--allenai--Olmo-3.1-32B-Think
    export LINE_RS=1
    pytest models/demos/llama3_70b_galaxy/tests/test_olmo_decode.py -v
"""

import torch
import pytest
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.llama_common import (
    precompute_freqs_yarn,
    PagedAttentionConfig,
)
from models.demos.llama3_70b_galaxy.tt.olmo_model_config import TtOlmoModelArgs
from models.demos.llama3_70b_galaxy.tt.llama_decoder import TtTransformerBlock
from models.demos.llama3_70b_galaxy.tt.llama_rope import TtLlamaRotarySetup
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
    (False,),  # Start with standard KV cache
    ids=("standard_kv",),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 64, "page_max_num_blocks": 4096}],
)
@pytest.mark.parametrize(
    "batch_size",
    (32,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (256,),  # For decode unit test, no need for large sequences
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "trace_region_size": 165136000,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
def test_olmo_decoder_decode(
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    """Test OLMo decoder block in decode mode with sliding window attention."""
    dtype = ttnn.bfloat8_b
    num_layers = 1  # Test single layer

    # Load OLMo model config
    model_args = TtOlmoModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
    )
    model_args.n_layers = num_layers

    logger.info(f"OLMo Config: layers={num_layers}, max_seq_len={max_seq_len}, batch={batch_size}")
    logger.info(f"  dim={model_args.dim}, n_heads={model_args.n_heads}, n_kv_heads={model_args.n_kv_heads}")

    # Load state dict
    state_dict = model_args.load_state_dict()

    # Setup prefetcher for decode mode (only if enabled in model config)
    if model_args.use_prefetcher:
        prefetcher_setup = TtLlamaPrefetcherSetup(
            mesh_device,
            n_tensors=5,
            n_layers=num_layers,
        )
        mesh_device.set_sub_device_stall_group(
            [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
        )
        worker_sub_device_id = prefetcher_setup.worker_sub_device_id
    else:
        prefetcher_setup = None
        # Even without prefetcher, decode CCL ops need a valid sub_device_id
        # Create a minimal sub_device setup covering all compute cores
        all_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9))])
        all_sub_device = ttnn.SubDevice([all_core_range_set])
        worker_sub_device_id = ttnn.SubDeviceId(0)
        sub_device_manager = mesh_device.create_sub_device_manager([all_sub_device], 0)
        mesh_device.load_sub_device_manager(sub_device_manager)
        mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    # Setup CCL for decode mode
    tt_ccl = TT_CCL(
        mesh_device,
        model_args,
        worker_sub_device_id,
        mode="decode",
        is_qwen=False,
        is_olmo=True,
    )

    # Setup paged attention
    paged_attention_config = None
    page_table_tt = None

    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        # Create page table
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.batch_size_per_device_group,
            paged_attention_config.max_num_blocks // model_args.batch_size_per_device_group,
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

    # Setup RoPE transformation matrices for YaRN
    rope_setup = TtLlamaRotarySetup(
        mesh_device,
        model_args.max_batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.use_scaled_rope,
        model_args.rope_scaling_factor,
    )
    transformation_mats = rope_setup.get_both_trans_mats()

    # Create decoder block
    logger.info("Loading decoder layer...")
    layer_id = 0
    tt_model = TtTransformerBlock(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        layer_num=layer_id,
        n_layers=num_layers,
        weight_cache_path=model_args.weight_cache_path(dtype),
        transformation_mats=transformation_mats,
        paged_attention_config=paged_attention_config,
        prefetcher_setup=prefetcher_setup,
        tt_ccl=tt_ccl,
    )

    # Check sliding window configuration
    sliding_window = model_args.get_sliding_window_size(layer_id)
    layer_type = model_args.get_layer_type(layer_id)
    logger.info(f"Layer {layer_id}: type={layer_type}, sliding_window={sliding_window}")

    # Prepare YaRN RoPE for reference
    ttnn_cos, ttnn_sin, mscale = precompute_freqs_yarn(
        dim=model_args.head_dim,
        end=max_seq_len * 2,
        theta=model_args.rope_theta,
        scaling_factor=model_args.rope_scaling_factor,
        original_max_position_embeddings=model_args.original_max_position_embeddings,
        beta_fast=model_args.yarn_beta_fast,
        beta_slow=model_args.yarn_beta_slow,
        attention_factor=model_args.yarn_attention_factor,
    )

    # Decode test parameters
    seqlen = 1
    generation_start_pos = 127
    generation_length = 10
    all_tests_pass = True

    # Initial positions
    current_pos = torch.tensor([generation_start_pos for _ in range(batch_size)])
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    # Explicitly allocate global CB (only if prefetcher is enabled)
    if prefetcher_setup is not None:
        prefetcher_setup.create_global_cb()

    for i in range(generation_length):
        logger.info(f"[Decode] Generating token {i}")

        # Create random input (single token)
        pt_decode_input = (torch.rand(batch_size, seqlen, model_args.dim) * 2) - 1
        tt_decode_input = pt_decode_input.clone()

        # Prepare input tensor
        decode_input = model_args.prepare_residual_tensor_decode(
            tt_decode_input,
            model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
        )

        # Get rotation matrices for current position
        rot_mats = rope_setup.get_rm_rot_mats(current_pos)

        # Setup prefetcher (only if enabled)
        if prefetcher_setup is not None:
            tt_pf = prefetcher_setup.get_input_tensors()
            ttnn.dram_prefetcher(
                tt_pf,
                num_layers=1,
                global_cb=prefetcher_setup.global_circular_buffer,
            )
            mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])

        # Run TT model in decode mode
        res = None
        tt_out, res = tt_model(
            decode_input,
            res,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )

        # Convert output to torch
        tt_output_torch = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )[:, 0:1, : model_args.max_batch_size, : model_args.dim].view(-1, 1, model_args.dim)

        # For now, just check output is valid (not NaN/Inf)
        assert not torch.isnan(tt_output_torch).any(), "Output contains NaN"
        assert not torch.isinf(tt_output_torch).any(), "Output contains Inf"

        logger.info(f"  Output shape: {tt_output_torch.shape}")
        logger.info(f"  Output mean: {tt_output_torch.mean().item():.4f}")
        logger.info(f"  Output std: {tt_output_torch.std().item():.4f}")

        # Increment position
        current_pos = torch.tensor([generation_start_pos + i + 1 for _ in range(batch_size)])
        current_pos_tensor = ttnn.from_torch(
            current_pos,
            device=mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    tt_ccl.close()

    logger.info(f"All {generation_length} OLMo decode iterations passed!")


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],  # Galaxy TG mesh shape
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
def test_olmo_sliding_window_decode_layers(mesh_device, reset_seeds, ensure_gc):
    """Test that sliding window is correctly applied to different layer types in decode mode."""
    batch_size = 32
    max_seq_len = 256
    dtype = ttnn.bfloat8_b

    # Load OLMo model config
    model_args = TtOlmoModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
    )

    # Test layer type patterns (3 sliding + 1 full)
    layer_types = []
    for layer_id in range(8):
        layer_type = model_args.get_layer_type(layer_id)
        sliding_window = model_args.get_sliding_window_size(layer_id)
        layer_types.append((layer_id, layer_type, sliding_window))
        logger.info(f"Layer {layer_id}: type={layer_type}, sliding_window={sliding_window}")

    # Verify pattern: 3 sliding, 1 full, repeat
    expected_pattern = [
        (0, "sliding_attention", 4096),
        (1, "sliding_attention", 4096),
        (2, "sliding_attention", 4096),
        (3, "full_attention", None),
        (4, "sliding_attention", 4096),
        (5, "sliding_attention", 4096),
        (6, "sliding_attention", 4096),
        (7, "full_attention", None),
    ]

    for layer_id, layer_type, sliding_window in layer_types:
        exp_layer_id, exp_type, exp_window = expected_pattern[layer_id]
        assert layer_type == exp_type, f"Layer {layer_id}: expected {exp_type}, got {layer_type}"
        assert (
            sliding_window == exp_window
        ), f"Layer {layer_id}: expected sliding_window={exp_window}, got {sliding_window}"

    logger.info("Sliding window pattern verification PASSED!")
