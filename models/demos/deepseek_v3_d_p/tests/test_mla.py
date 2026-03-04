# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Test for instantiating both reference CPU and TT device MLA modules with the same weights.
This test verifies that both modules can be created and weights are loaded correctly.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.mla_reference import create_mla_reference
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.utils.utils import get_rope_tensors
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
    [(8, 4)],
    ids=["4x2"],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"])
@pytest.mark.parametrize("seq_len", [128 * 1024], ids=["seq8k"])
@pytest.mark.parametrize("skip_check", [False, True], ids=["pcc_check", "skip_check"])
# @pytest.mark.timeout(900)  # Increase timeout to 15 minutes for large sequence lengths
def test_mla(use_pretrained, request, mesh_device, seq_len, skip_check):
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

    # temp hack
    config.max_seq_len = seq_len

    # Create reference MLA
    mla_ref = None
    if not skip_check:
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
    mla_tt = ttMLA(config, weights, mesh_device, layer_idx=0, seq_len=seq_len, sp_axis=0, tp_axis=1)
    rope_tensors = get_rope_tensors(config, seq_len, mesh_device, sp_axis=0)

    # Verify both exist
    assert not (mla_ref is None and not skip_check), "Reference MLA should exist"
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

    # Create causal attention mask
    attention_mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1).to(torch.bfloat16)
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)

    # Create position IDs
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)

    # Run reference forward pass
    if not skip_check:
        logger.info("Running reference CPU forward pass...")
        mla_ref = mla_ref.eval().to(torch.bfloat16)
        with torch.no_grad():
            ref_output, _, _ = mla_ref(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )

        logger.info(f"✓ Reference forward pass complete")
        logger.info(f"  Input shape:  {hidden_states.shape}")
        logger.info(f"  Output shape: {ref_output.shape}")
        logger.info(f"  Output dtype: {ref_output.dtype}")
        logger.info(f"  Output mean:  {ref_output.mean().item():.4f}")
        logger.info(f"  Output std:   {ref_output.std().item():.4f}")

    tt_hidden_states = ttnn.from_torch(
        hidden_states.unsqueeze(0),
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

    if tt_output is not None:
        tt_output_cpu = ttnn.to_torch(
            tt_output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=mesh_device.shape)
        ).to(torch.bfloat16)

        if not skip_check:
            _, pcc_message = assert_with_pcc(ref_output.unsqueeze(0), tt_output_cpu, 0.98)
            logger.info(f"PCC is {pcc_message}")

    logger.success(f"✓ Reference and TT comparison with {weight_type} weights successful")
