# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Test for instantiating both reference CPU and TT device MLA modules with the same weights.
This test verifies that both modules can be created and weights are loaded correctly.
"""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers.cache_utils import DynamicCache

import ttnn
from models.demos.deepseek_v3_d_p.reference.mla_reference import create_mla_reference
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.mla.utils import (
    create_balanced_chunk_order,
    reorder_tensor_chunks,
    reverse_reorder_tensor_chunks,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.fixture
def random_weights(config_only):
    """
    Generate random weights for testing using the config.

    Args:
        config_only: HuggingFace config (only downloads config files, not weight shards)

    Returns:
        Tuple of (config, weights_dict) in bfloat16
    """
    config = config_only

    torch.manual_seed(42)

    # Use proper initialization scale from config (typically 0.02)
    std = config.initializer_range

    # Generate random weights matching MLA architecture using actual config
    # Generate in float32 first, then convert to bfloat16 for better numerical properties
    weights = {
        "q_a_proj.weight": (torch.randn(config.q_lora_rank, config.hidden_size) * std).to(torch.bfloat16),
        "q_a_layernorm.weight": torch.ones(config.q_lora_rank, dtype=torch.bfloat16),
        "q_b_proj.weight": (
            torch.randn(
                config.num_attention_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim),
                config.q_lora_rank,
            )
            * std
        ).to(torch.bfloat16),
        "kv_a_proj_with_mqa.weight": (
            torch.randn(
                config.kv_lora_rank + config.qk_rope_head_dim,
                config.hidden_size,
            )
            * std
        ).to(torch.bfloat16),
        "kv_a_layernorm.weight": torch.ones(config.kv_lora_rank, dtype=torch.bfloat16),
        "kv_b_proj.weight": (
            torch.randn(
                config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
                config.kv_lora_rank,
            )
            * std
        ).to(torch.bfloat16),
        "o_proj.weight": (
            torch.randn(
                config.hidden_size,
                config.num_attention_heads * config.v_head_dim,
            )
            * std
        ).to(torch.bfloat16),
    }

    logger.info(f"Generated {len(weights)} random weight tensors using config dimensions")
    return config, weights


# sp x tp
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4), (2, 4)],
    ids=["8x4", "2x4"],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"])
@pytest.mark.parametrize("scale_down_sl", [False, True], ids=["max_sl", "scaled_sl"])
@pytest.mark.parametrize("seq_len", [128 * 1024, 100 * 1024], ids=["seq128k", "seq100k"])
@pytest.mark.parametrize("skip_host_comparison", [False, True], ids=["check_pcc", "skip_check"])
@pytest.mark.parametrize("is_balanced", [False, True], ids=["sequential", "balanced"])
@pytest.mark.timeout(0)  # Disable timeout — first run computes and caches CPU reference for large seq lengths
def test_mla(use_pretrained, request, mesh_device, seq_len, skip_host_comparison, scale_down_sl, is_balanced):
    """
    Test comparing reference and TT MLA modules with same weights.

    Args:
        use_pretrained: Whether to use pretrained weights
        request: Pytest request object for conditional fixture loading
        mesh_device: Mesh device fixture
        seq_len: Sequence length
    """
    weight_type = "Pretrained" if use_pretrained else "Random"
    logger.info("=" * 80)
    logger.info(f"Test: Reference vs TT Comparison ({weight_type} Weights)")
    logger.info("=" * 80)

    # Conditionally load fixtures - only load what we need!
    if use_pretrained:
        config, weights = request.getfixturevalue("pretrained_weights")
    else:
        config, weights = request.getfixturevalue("random_weights")

    production_mesh = [32, 4]
    sp_axis = 0
    tp_axis = 1

    mesh_shape = list(mesh_device.shape)
    if scale_down_sl:
        seq_len = (seq_len // production_mesh[sp_axis]) * mesh_shape[sp_axis]

    # temp hack
    config.max_seq_len = seq_len

    # Create reference MLA
    if use_pretrained:
        # For pretrained, create from weights
        logger.info("Creating reference MLA with pretrained weights...")
        mla_ref = create_mla_reference(
            config=config,
            state_dict={"model.layers.0.self_attn." + k: v for k, v in weights.items()},
            layer_idx=0,
            module_path="model.layers.0.self_attn",
        )
    else:
        # For random, use same weights
        logger.info("Creating reference MLA with random weights...")
        mla_ref = create_mla_reference(
            config=config,
            state_dict={"model.layers.0.self_attn." + k: v for k, v in weights.items()},
            layer_idx=0,
            module_path="model.layers.0.self_attn",
        )

    # Create TT MLA
    logger.info("Creating TT MLA...")
    mla_tt = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=is_balanced,
    )
    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=is_balanced)
    rope_tensors = rope_setup.get_rope_tensors(seq_len)

    # Verify both exist
    assert mla_ref is not None, "Reference MLA should exist"
    assert mla_tt is not None, "TT MLA should exist"

    # Test forward pass comparison
    logger.info("=" * 80)
    logger.info(f"Testing forward pass comparison (seq_len={seq_len})")
    logger.info("=" * 80)

    # Create test inputs
    batch_size = 1
    hidden_size = config.hidden_size

    logger.info(f"Creating test inputs: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")

    # Create random input tensor (generate in float32, then convert to bfloat16)
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(torch.bfloat16)

    if skip_host_comparison == False:
        # Check for cached reference results to avoid expensive host attention computation
        cache_dir = Path(os.environ.get("DEEPSEEK_V3_MLA_REF_CACHE", "/tmp/deepseek_v3_mla_ref_cache"))
        cache_key = f"{weight_type.lower()}_seq{seq_len}"
        cache_path = cache_dir / f"{cache_key}.pt"

        if cache_path.exists():
            logger.info(f"Loading cached reference results from {cache_path}")
            cached = torch.load(cache_path, weights_only=True)
            ref_output = cached["ref_output"]
            ref_kvpe = cached["ref_kvpe"]
            logger.info(f"✓ Loaded cached reference results")
            logger.info(f"  Output shape: {ref_output.shape}")
        else:
            # Create position IDs
            position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)

            # Run reference forward pass with cache to capture KVPE
            # Uses F.scaled_dot_product_attention with is_causal=True (no explicit mask needed)
            logger.info("Running reference CPU forward pass...")
            mla_ref = mla_ref.eval().to(torch.bfloat16)
            ref_cache = DynamicCache()
            with torch.no_grad():
                ref_output, _, ref_cache = mla_ref(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    past_key_value=ref_cache,
                    use_cache=True,
                )

            ref_kvpe = ref_cache.key_cache[0]  # layer 0

            # Save to cache for future runs
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"ref_output": ref_output, "ref_kvpe": ref_kvpe}, cache_path)
            logger.info(f"✓ Saved reference results to {cache_path}")

            logger.info(f"✓ Reference forward pass complete")
            logger.info(f"  Input shape:  {hidden_states.shape}")
            logger.info(f"  Output shape: {ref_output.shape}")
            logger.info(f"  Output dtype: {ref_output.dtype}")
            logger.info(f"  Output mean:  {ref_output.mean().item():.4f}")
            logger.info(f"  Output std:   {ref_output.std().item():.4f}")

    # Reorder hidden_states for balanced ring attention
    sp_factor = mesh_shape[sp_axis]
    chunk_order = create_balanced_chunk_order(sp_factor) if is_balanced else None
    tt_input = hidden_states.unsqueeze(0)  # [1, batch, seq, hidden]
    if is_balanced:
        tt_input = reorder_tensor_chunks(tt_input, chunk_order, seq_dim=2)

    tt_hidden_states = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(-2, -1)),
    )
    tt_output = mla_tt.forward(
        hidden_states=tt_hidden_states,
        rope_tensors=rope_tensors,
    )

    if skip_host_comparison == False:
        tt_output_cpu = ttnn.to_torch(
            tt_output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=mesh_device.shape)
        ).to(torch.bfloat16)

        if is_balanced:
            tt_output_cpu = reverse_reorder_tensor_chunks(tt_output_cpu, chunk_order, seq_dim=2)

        _, pcc_message = assert_with_pcc(ref_output.unsqueeze(0), tt_output_cpu, 0.98)
        logger.info(f"Output PCC is {pcc_message}")

        # Validate KVPE cache contents
        # Reference KVPE: [batch, 1, seq_len, kv_lora_rank + qk_rope_head_dim]
        # ref_kvpe is already available (loaded from cache or computed above)

        # Read back KVPE cache from device
        # Cache is replicated across TP, so concat TP replicas on dim 1 (unused) and discard extras
        tt_kvpe_cache = ttnn.to_torch(
            mla_tt.kvpe_cache,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
        ).to(torch.bfloat16)
        tt_kvpe_cache = tt_kvpe_cache[:, :1, :, :]

        if is_balanced:
            tt_kvpe_cache = reverse_reorder_tensor_chunks(tt_kvpe_cache, chunk_order, seq_dim=2)

        # Check PCC separately for KV (latent) and PE (rope) parts
        kv_lora_rank = config.kv_lora_rank
        _, kv_pcc_message = assert_with_pcc(
            ref_kvpe[:, :, :, :kv_lora_rank], tt_kvpe_cache[:, :, :, :kv_lora_rank], 0.99
        )
        logger.info(f"KVPE cache KV part PCC is {kv_pcc_message}")
        _, pe_pcc_message = assert_with_pcc(
            ref_kvpe[:, :, :, kv_lora_rank:], tt_kvpe_cache[:, :, :, kv_lora_rank:], 0.99
        )
        logger.info(f"KVPE cache PE part PCC is {pe_pcc_message}")
    else:
        ttnn.synchronize_device(mesh_device)

    logger.success(f"✓ Reference and TT comparison with {weight_type} weights successful")
