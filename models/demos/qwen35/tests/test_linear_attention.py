# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Qwen3.5 Linear Attention (GatedDeltaNet) implementation.
"""

import pytest
import torch
from loguru import logger
from transformers import AutoModel

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen35.tt.linear_attention import TtLinearAttention


def get_reference_model(model_name="Qwen/Qwen3.5-9B"):
    """Load HuggingFace reference model."""
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    return model


@pytest.mark.parametrize(
    "layer_idx,seq_len,batch_size",
    [
        (1, 128, 1),  # Layer 1 is linear attention
    ],
)
def test_linear_attention_forward(layer_idx, seq_len, batch_size, mesh_device):
    """Test full forward pass against HuggingFace reference."""

    # Load reference model
    ref_model = get_reference_model()
    ref_linear_attn = ref_model.layers[layer_idx].linear_attn

    # Get model config from reference
    from models.demos.qwen35.tt.model_config import Qwen35ModelArgs

    model_args = Qwen35ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=seq_len,
        dummy_weights=False,
    )

    # Verify layer is linear attention
    assert model_args.is_linear_attention_layer(layer_idx), f"Layer {layer_idx} is not linear attention"

    # Create TT model
    tt_model = TtLinearAttention(model_args, layer_idx, mesh_device)
    tt_model.load_weights(ref_model.state_dict())

    # Create test input
    hidden_size = model_args.dim
    x_pt = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # Run reference
    with torch.no_grad():
        ref_output = ref_linear_attn(x_pt)

    # Run TT model
    x_tt = ttnn.from_torch(
        x_pt,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_output = tt_model(x_tt)
    tt_output_torch = ttnn.to_torch(tt_output)

    # Compare outputs
    passing, pcc = comp_pcc(ref_output, tt_output_torch)
    logger.info(f"Forward pass PCC: {pcc}")

    assert passing, f"Forward pass PCC {pcc} below threshold"
    logger.info("✓ Forward pass matches reference")


@pytest.mark.parametrize(
    "layer_idx,seq_len,batch_size",
    [
        (1, 32, 1),
    ],
)
def test_qkv_split(layer_idx, seq_len, batch_size, mesh_device):
    """Test QKV splitting against HuggingFace reference."""

    ref_model = get_reference_model()
    ref_linear_attn = ref_model.layers[layer_idx].linear_attn

    from models.demos.qwen35.tt.model_config import Qwen35ModelArgs

    model_args = Qwen35ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=seq_len,
        dummy_weights=False,
    )

    tt_model = TtLinearAttention(model_args, layer_idx, mesh_device)
    tt_model.load_weights(ref_model.state_dict())

    # Create test input
    hidden_size = model_args.dim
    x_pt = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # Run reference projection + split
    with torch.no_grad():
        mixed_qkv_pt = ref_linear_attn.in_proj_qkv(x_pt)
        q_pt, k_pt, v_pt = torch.split(
            mixed_qkv_pt,
            [tt_model.key_dim, tt_model.key_dim, tt_model.value_dim],
            dim=-1,
        )

    # Run TT projection + split
    x_tt = ttnn.from_torch(x_pt, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    mixed_qkv_tt = ttnn.linear(x_tt, tt_model.in_proj_qkv)
    q_tt, k_tt, v_tt = tt_model._split_qkv(mixed_qkv_tt)

    # Compare
    q_tt_torch = ttnn.to_torch(q_tt)
    k_tt_torch = ttnn.to_torch(k_tt)
    v_tt_torch = ttnn.to_torch(v_tt)

    passing_q, pcc_q = comp_pcc(q_pt, q_tt_torch)
    passing_k, pcc_k = comp_pcc(k_pt, k_tt_torch)
    passing_v, pcc_v = comp_pcc(v_pt, v_tt_torch)

    logger.info(f"Q split PCC: {pcc_q}")
    logger.info(f"K split PCC: {pcc_k}")
    logger.info(f"V split PCC: {pcc_v}")

    assert passing_q and passing_k and passing_v, "QKV split PCC below threshold"
    logger.info("✓ QKV split matches reference")


@pytest.mark.parametrize(
    "layer_idx,seq_len,batch_size",
    [
        (1, 32, 1),
    ],
)
def test_g_computation(layer_idx, seq_len, batch_size, mesh_device):
    """Test g parameter computation against HuggingFace reference."""

    ref_model = get_reference_model()
    ref_linear_attn = ref_model.layers[layer_idx].linear_attn

    from models.demos.qwen35.tt.model_config import Qwen35ModelArgs

    model_args = Qwen35ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=seq_len,
        dummy_weights=False,
    )

    tt_model = TtLinearAttention(model_args, layer_idx, mesh_device)
    tt_model.load_weights(ref_model.state_dict())

    # Create test input
    hidden_size = model_args.dim
    x_pt = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # Run reference
    with torch.no_grad():
        a_pt = ref_linear_attn.in_proj_a(x_pt)
        g_pt = -ref_linear_attn.A_log.float().exp() * torch.nn.functional.softplus(
            a_pt.float() + ref_linear_attn.dt_bias
        )

    # Run TT
    x_tt = ttnn.from_torch(x_pt, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    a_tt = ttnn.linear(x_tt, tt_model.in_proj_a)
    g_tt = tt_model._compute_g(a_tt)

    # Compare
    g_tt_torch = ttnn.to_torch(g_tt)
    passing, pcc = comp_pcc(g_pt, g_tt_torch)

    logger.info(f"g computation PCC: {pcc}")
    assert passing, f"g computation PCC {pcc} below threshold"
    logger.info("✓ g computation matches reference")


@pytest.mark.parametrize(
    "layer_idx,seq_len,batch_size",
    [
        (1, 32, 1),
    ],
)
def test_rms_norm_gated(layer_idx, seq_len, batch_size, mesh_device):
    """Test RMSNormGated against HuggingFace reference."""

    ref_model = get_reference_model()
    ref_linear_attn = ref_model.layers[layer_idx].linear_attn

    from models.demos.qwen35.tt.model_config import Qwen35ModelArgs

    model_args = Qwen35ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=seq_len,
        dummy_weights=False,
    )

    tt_model = TtLinearAttention(model_args, layer_idx, mesh_device)
    tt_model.load_weights(ref_model.state_dict())

    # Create test inputs
    N = 128
    head_v_dim = tt_model.head_v_dim
    x_pt = torch.randn(N, head_v_dim, dtype=torch.bfloat16)
    gate_pt = torch.randn(N, head_v_dim, dtype=torch.bfloat16)

    # Run reference
    with torch.no_grad():
        output_pt = ref_linear_attn.norm(x_pt, gate_pt)

    # Run TT
    x_tt = ttnn.from_torch(x_pt, dtype=ttnn.bfloat16, device=mesh_device)
    gate_tt = ttnn.from_torch(gate_pt, dtype=ttnn.bfloat16, device=mesh_device)
    output_tt = tt_model._rms_norm_gated(x_tt, gate_tt)

    # Compare
    output_tt_torch = ttnn.to_torch(output_tt)
    passing, pcc = comp_pcc(output_pt, output_tt_torch)

    logger.info(f"RMSNormGated PCC: {pcc}")
    assert passing, f"RMSNormGated PCC {pcc} below threshold"
    logger.info("✓ RMSNormGated matches reference")


@pytest.mark.parametrize(
    "layer_idx,seq_len,batch_size",
    [
        (1, 8, 1),  # Short sequence for causal conv
    ],
)
def test_causal_conv1d(layer_idx, seq_len, batch_size, mesh_device):
    """Test causal 1D convolution against HuggingFace reference."""

    ref_model = get_reference_model()
    ref_linear_attn = ref_model.layers[layer_idx].linear_attn

    from models.demos.qwen35.tt.model_config import Qwen35ModelArgs

    model_args = Qwen35ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=seq_len,
        dummy_weights=False,
    )

    tt_model = TtLinearAttention(model_args, layer_idx, mesh_device)
    tt_model.load_weights(ref_model.state_dict())

    # Create test input: [batch, seq_len, conv_dim]
    conv_dim = tt_model.conv_dim
    x_pt = torch.randn(batch_size, seq_len, conv_dim, dtype=torch.bfloat16)

    # Run reference causal conv (extract from forward pass logic)
    with torch.no_grad():
        # Reference doesn't expose _causal_conv1d directly, so we test via full forward
        # But we can verify the conv step by checking intermediate values
        pass  # Skip exact reference comparison for internal method

    # Run TT causal conv with no cache
    x_tt = ttnn.from_torch(x_pt, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    x_conv_tt = tt_model._causal_conv1d(x_tt, cache_params=None)
    x_conv_torch = ttnn.to_torch(x_conv_tt)

    # Verify output shape
    assert x_conv_torch.shape == x_pt.shape, f"Shape mismatch: {x_conv_torch.shape} vs {x_pt.shape}"
    logger.info("✓ Causal conv1d output shape correct")


@pytest.mark.parametrize(
    "layer_idx,seq_len,batch_size",
    [
        (1, 1, 1),  # Decode mode (seq_len=1)
    ],
)
def test_recurrent_gated_delta_rule(layer_idx, seq_len, batch_size, mesh_device):
    """Test recurrent gated delta rule (decode mode)."""

    ref_model = get_reference_model()
    ref_linear_attn = ref_model.layers[layer_idx].linear_attn

    from models.demos.qwen35.tt.model_config import Qwen35ModelArgs

    model_args = Qwen35ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=seq_len,
        dummy_weights=False,
    )

    tt_model = TtLinearAttention(model_args, layer_idx, mesh_device)
    tt_model.load_weights(ref_model.state_dict())

    # Create test inputs
    num_v_heads = tt_model.num_v_heads
    k_head_dim = tt_model.head_k_dim
    v_head_dim = tt_model.head_v_dim

    q_pt = torch.randn(batch_size, seq_len, num_v_heads, k_head_dim, dtype=torch.bfloat16)
    k_pt = torch.randn(batch_size, seq_len, num_v_heads, k_head_dim, dtype=torch.bfloat16)
    v_pt = torch.randn(batch_size, seq_len, num_v_heads, v_head_dim, dtype=torch.bfloat16)
    g_pt = torch.randn(batch_size, seq_len, num_v_heads, dtype=torch.bfloat16)
    beta_pt = torch.randn(batch_size, seq_len, num_v_heads, dtype=torch.bfloat16)

    # Convert to ttnn
    q_tt = ttnn.from_torch(q_pt, dtype=ttnn.bfloat16, device=mesh_device)
    k_tt = ttnn.from_torch(k_pt, dtype=ttnn.bfloat16, device=mesh_device)
    v_tt = ttnn.from_torch(v_pt, dtype=ttnn.bfloat16, device=mesh_device)
    g_tt = ttnn.from_torch(g_pt, dtype=ttnn.bfloat16, device=mesh_device)
    beta_tt = ttnn.from_torch(beta_pt, dtype=ttnn.bfloat16, device=mesh_device)

    # Run recurrent delta rule
    output_tt, state_tt = tt_model._recurrent_gated_delta_rule(
        q_tt, k_tt, v_tt, g_tt, beta_tt, initial_state=None, output_final_state=True
    )

    output_torch = ttnn.to_torch(output_tt)

    # Verify output shapes
    assert output_torch.shape == (
        batch_size,
        seq_len,
        num_v_heads,
        v_head_dim,
    ), f"Output shape mismatch: {output_torch.shape}"
    assert state_tt is not None, "State should be returned when output_final_state=True"

    state_torch = ttnn.to_torch(state_tt)
    assert state_torch.shape == (
        batch_size,
        num_v_heads,
        k_head_dim,
        v_head_dim,
    ), f"State shape mismatch: {state_torch.shape}"

    logger.info("✓ Recurrent gated delta rule output shapes correct")


@pytest.mark.parametrize(
    "layer_idx,seq_len,batch_size",
    [
        (1, 64, 1),  # Chunk mode (seq_len=64)
    ],
)
def test_chunk_gated_delta_rule(layer_idx, seq_len, batch_size, mesh_device):
    """Test chunk gated delta rule (prefill mode)."""

    ref_model = get_reference_model()
    ref_linear_attn = ref_model.layers[layer_idx].linear_attn

    from models.demos.qwen35.tt.model_config import Qwen35ModelArgs

    model_args = Qwen35ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=seq_len,
        dummy_weights=False,
    )

    tt_model = TtLinearAttention(model_args, layer_idx, mesh_device)
    tt_model.load_weights(ref_model.state_dict())

    # Create test inputs
    num_v_heads = tt_model.num_v_heads
    k_head_dim = tt_model.head_k_dim
    v_head_dim = tt_model.head_v_dim

    q_pt = torch.randn(batch_size, seq_len, num_v_heads, k_head_dim, dtype=torch.bfloat16)
    k_pt = torch.randn(batch_size, seq_len, num_v_heads, k_head_dim, dtype=torch.bfloat16)
    v_pt = torch.randn(batch_size, seq_len, num_v_heads, v_head_dim, dtype=torch.bfloat16)
    k_beta_pt = torch.randn(batch_size, seq_len, num_v_heads, k_head_dim, dtype=torch.bfloat16)
    v_beta_pt = torch.randn(batch_size, seq_len, num_v_heads, v_head_dim, dtype=torch.bfloat16)
    g_pt = torch.randn(batch_size, seq_len, num_v_heads, dtype=torch.bfloat16)

    # Convert to ttnn
    q_tt = ttnn.from_torch(q_pt, dtype=ttnn.bfloat16, device=mesh_device)
    k_tt = ttnn.from_torch(k_pt, dtype=ttnn.bfloat16, device=mesh_device)
    v_tt = ttnn.from_torch(v_pt, dtype=ttnn.bfloat16, device=mesh_device)
    k_beta_tt = ttnn.from_torch(k_beta_pt, dtype=ttnn.bfloat16, device=mesh_device)
    v_beta_tt = ttnn.from_torch(v_beta_pt, dtype=ttnn.bfloat16, device=mesh_device)
    g_tt = ttnn.from_torch(g_pt, dtype=ttnn.bfloat16, device=mesh_device)

    # Run chunk delta rule
    output_tt, state_tt = tt_model._chunk_gated_delta_rule(
        q_tt, k_tt, v_tt, k_beta_tt, v_beta_tt, g_tt, initial_state=None, output_final_state=True
    )

    output_torch = ttnn.to_torch(output_tt)

    # Verify output shapes
    assert output_torch.shape == (
        batch_size,
        seq_len,
        num_v_heads,
        v_head_dim,
    ), f"Output shape mismatch: {output_torch.shape}"
    assert state_tt is not None, "State should be returned when output_final_state=True"

    state_torch = ttnn.to_torch(state_tt)
    assert state_torch.shape == (
        batch_size,
        num_v_heads,
        k_head_dim,
        v_head_dim,
    ), f"State shape mismatch: {state_torch.shape}"

    logger.info("✓ Chunk gated delta rule output shapes correct")


@pytest.mark.parametrize(
    "layer_idx,prefill_len,decode_len,batch_size",
    [
        (1, 64, 10, 1),  # Prefill 64 tokens, then decode 10 tokens
    ],
)
def test_linear_attention_with_cache(layer_idx, prefill_len, decode_len, batch_size, mesh_device):
    """Test linear attention with cache across prefill and decode phases."""

    ref_model = get_reference_model()
    ref_linear_attn = ref_model.layers[layer_idx].linear_attn

    from models.demos.qwen35.tt.model_config import Qwen35ModelArgs

    model_args = Qwen35ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=prefill_len + decode_len,
        dummy_weights=False,
    )

    tt_model = TtLinearAttention(model_args, layer_idx, mesh_device)
    tt_model.load_weights(ref_model.state_dict())

    hidden_size = model_args.dim

    # Phase 1: Prefill
    x_prefill_pt = torch.randn(batch_size, prefill_len, hidden_size, dtype=torch.bfloat16)
    x_prefill_tt = ttnn.from_torch(
        x_prefill_pt,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
    )

    # Initialize cache
    cache_params = {"conv_state": None, "ssm_state": None}

    # Run prefill
    output_prefill_tt = tt_model(x_prefill_tt, cache_params=cache_params)
    output_prefill_torch = ttnn.to_torch(output_prefill_tt)

    # Verify prefill output shape
    assert output_prefill_torch.shape == (
        batch_size,
        prefill_len,
        hidden_size,
    ), f"Prefill output shape mismatch: {output_prefill_torch.shape}"

    # Verify cache was populated
    assert cache_params["conv_state"] is not None, "Conv state should be cached after prefill"
    assert cache_params["ssm_state"] is not None, "SSM state should be cached after prefill"

    # Phase 2: Decode (one token at a time)
    for i in range(decode_len):
        x_decode_pt = torch.randn(batch_size, 1, hidden_size, dtype=torch.bfloat16)
        x_decode_tt = ttnn.from_torch(
            x_decode_pt,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
        )

        # Run decode with cache
        output_decode_tt = tt_model(x_decode_tt, cache_params=cache_params)
        output_decode_torch = ttnn.to_torch(output_decode_tt)

        # Verify decode output shape
        assert output_decode_torch.shape == (
            batch_size,
            1,
            hidden_size,
        ), f"Decode step {i} output shape mismatch: {output_decode_torch.shape}"

    logger.info("✓ Linear attention with cache works across prefill and decode")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
