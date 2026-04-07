# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC comparison tests for CPU bottleneck fixes in Qwen3.5-27B TTNN modules.

Tests verify that the device-side implementations match PyTorch reference outputs:
1. GDN prefill on device vs PyTorch reference (PCC >= 0.95)
2. Full attention with paged KV cache vs PyTorch reference (PCC >= 0.90)
"""

import pytest
import torch

from .conftest import (
    assert_with_pcc,
    get_config_attr,
    get_layer_type,
    skip_no_ttnn,
    skip_no_transformers,
    skip_no_symbiote,
    TTNN_AVAILABLE,
    TT_SYMBIOTE_AVAILABLE,
)

if TTNN_AVAILABLE:
    import ttnn

if TT_SYMBIOTE_AVAILABLE:
    from models.experimental.tt_symbiote.modules.qwen35_gated_deltanet import TTNNQwen35GatedDeltaNet
    from models.experimental.tt_symbiote.modules.qwen35_attention import TTNNQwen35FullAttention
    from models.experimental.tt_symbiote.modules.attention import PagedAttentionConfig
    from models.experimental.tt_symbiote.modules.qwen_attention import TTNNQwenPagedAttentionKVCache
    from models.experimental.tt_symbiote.utils.device_management import set_device


# ──────────────────────────────────────────────────────────────────────
# GDN Prefill Tests
# ──────────────────────────────────────────────────────────────────────


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_gdn_prefill_device_pcc(device, model_1_layer):
    """GDN prefill on device vs PyTorch reference. PCC >= 0.95.

    Runs the full GDN layer (projections + conv1d + recurrence + gate + out_proj)
    with seq_len=32 on device and compares against the PyTorch GDN module.
    """
    model, config = model_1_layer
    layer_0 = model.model.layers[0]

    # Layer 0 should be linear attention (GDN)
    assert get_layer_type(layer_0) == "linear_attention", "Layer 0 is not linear attention; cannot test GDN prefill"
    torch_gdn = layer_0.linear_attn

    # Create input
    hidden_size = get_config_attr(config, "hidden_size")
    batch_size, seq_len = 1, 32
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # PyTorch reference
    with torch.no_grad():
        torch_out = torch_gdn(x)[0]  # GDN returns (output, ...) tuple

    # TTNN on-device prefill
    ttnn_gdn = TTNNQwen35GatedDeltaNet.from_torch(torch_gdn)
    set_device(ttnn_gdn, device)
    ttnn_gdn.preprocess_weights()
    ttnn_gdn.move_weights_to_device()

    tt_input = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_out = ttnn_gdn.forward(tt_input)

    pcc_val = assert_with_pcc(torch_out, ttnn_out, 0.95)
    print(f"  GDN prefill device PCC = {pcc_val:.6f}")


# ──────────────────────────────────────────────────────────────────────
# KV Cache Tests
# ──────────────────────────────────────────────────────────────────────


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_full_attn_paged_kv_cache_pcc(device, model_4_layers):
    """Full attention with paged KV cache vs PyTorch reference. PCC >= 0.90.

    Tests the paged attention path for prefill.
    Layer 3 is the first full attention layer in the [linear, linear, linear, full]
    pattern.
    """
    model, config = model_4_layers

    # Layer 3 is the first full attention layer
    torch_attn = model.model.layers[3].self_attn

    hidden_size = get_config_attr(config, "hidden_size")
    num_kv_heads = get_config_attr(config, "num_key_value_heads")
    head_dim = get_config_attr(config, "head_dim")
    batch_size, seq_len = 1, 32

    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # Generate position embeddings using the model's rotary embedding
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = model.model.rotary_emb(x, position_ids)

    # PyTorch reference (no cache, just prefill)
    with torch.no_grad():
        torch_out = torch_attn(
            x,
            attention_mask=None,
            position_embeddings=(cos, sin),
        )[0]

    # TTNN with paged attention KV cache
    paged_config = PagedAttentionConfig(block_size=64, max_num_blocks=32, batch_size=batch_size)
    paged_cache = TTNNQwenPagedAttentionKVCache(
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        config=paged_config,
        device=None,
        layer_indices=[3],  # Map layer_idx=3 -> cache_idx=0
    ).to_device(device)

    ttnn_attn = TTNNQwen35FullAttention.from_torch(torch_attn)
    set_device(ttnn_attn, device)
    ttnn_attn.preprocess_weights()
    ttnn_attn.move_weights_to_device()

    tt_input = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_cos = ttnn.from_torch(
        cos,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_sin = ttnn.from_torch(
        sin,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Use paged attention KV cache
    ttnn_out, _ = ttnn_attn.forward(
        tt_input,
        position_embeddings=(tt_cos, tt_sin),
        past_key_values=paged_cache,
    )

    pcc_val = assert_with_pcc(torch_out, ttnn_out, 0.90)
    print(f"  Full attention paged KV cache PCC = {pcc_val:.6f}")

    # Verify paged cache was populated
    seq_after = paged_cache.get_seq_length(3)
    assert seq_after == seq_len, f"Expected seq_length={seq_len}, got {seq_after}"
    print(f"  Paged cache seq_length after prefill: {seq_after}")


@pytest.fixture(scope="module")
def model_4_layers():
    """Load Qwen3.5-27B-FP8 with 4 hidden layers (module-scoped for reuse)."""
    from .conftest import load_model

    model, config = load_model(num_hidden_layers=4)
    return model, config
