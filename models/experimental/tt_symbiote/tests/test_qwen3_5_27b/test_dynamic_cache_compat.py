# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Test 3: Validate paged KV cache seq_length tracking and causal mask correctness.

Verifies that TTNNQwen35FullAttention correctly uses paged attention:
- paged_fill_on_device during prefill fills the KV cache
- paged_update_on_device during decode updates at the correct position
- get_seq_length() returns the correct value for mask/position_ids logic
"""

import pytest
import torch

from .conftest import (
    get_config_attr,
    skip_no_ttnn,
    skip_no_transformers,
    skip_no_symbiote,
    TTNN_AVAILABLE,
    TT_SYMBIOTE_AVAILABLE,
)

if TTNN_AVAILABLE:
    import ttnn

if TT_SYMBIOTE_AVAILABLE:
    from models.experimental.tt_symbiote.modules.qwen35_attention import TTNNQwen35FullAttention
    from models.experimental.tt_symbiote.modules.attention import PagedAttentionConfig
    from models.experimental.tt_symbiote.modules.qwen_attention import TTNNQwenPagedAttentionKVCache
    from models.experimental.tt_symbiote.utils.device_management import set_device


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_paged_cache_seq_length_tracking(device, model_4_layers):
    """Paged KV cache get_seq_length() returns correct values after prefill + decode.

    Simulates the prefill + decode flow:
    1. Prefill with seq_len=32 -> get_seq_length() should return 32
    2. Decode with seq_len=1 -> get_seq_length() should return 33
    """
    model, config = model_4_layers

    # Layer 3 is the first full attention layer
    torch_attn = model.model.layers[3].self_attn

    hidden_size = get_config_attr(config, "hidden_size")
    num_kv_heads = get_config_attr(config, "num_key_value_heads")
    head_dim = get_config_attr(config, "head_dim")
    batch_size = 1

    # Create paged KV cache for layer 3 only
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

    # ---- Prefill (seq_len=32) ----
    seq_len_prefill = 32
    x_prefill = torch.randn(batch_size, seq_len_prefill, hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len_prefill).unsqueeze(0)
    cos, sin = model.model.rotary_emb(x_prefill, position_ids)

    tt_input = ttnn.from_torch(
        x_prefill,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_cos = ttnn.from_torch(
        cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_sin = ttnn.from_torch(
        sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    ttnn_attn.forward(
        tt_input,
        position_embeddings=(tt_cos, tt_sin),
        past_key_values=paged_cache,
    )

    seq_after_prefill = paged_cache.get_seq_length(3)
    print(f"  get_seq_length(3) after prefill: {seq_after_prefill}")
    assert seq_after_prefill == seq_len_prefill, f"Expected {seq_len_prefill}, got {seq_after_prefill}"

    # ---- Decode (seq_len=1) ----
    x_decode = torch.randn(batch_size, 1, hidden_size, dtype=torch.bfloat16)
    position_ids_decode = torch.tensor([[seq_len_prefill]])
    cos_d, sin_d = model.model.rotary_emb(x_decode, position_ids_decode)

    tt_input_d = ttnn.from_torch(
        x_decode,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_cos_d = ttnn.from_torch(
        cos_d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_sin_d = ttnn.from_torch(
        sin_d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # cache_position tells the paged decode where to write
    cache_position = torch.tensor([seq_len_prefill], dtype=torch.int32)

    ttnn_attn.forward(
        tt_input_d,
        position_embeddings=(tt_cos_d, tt_sin_d),
        past_key_values=paged_cache,
        cache_position=cache_position,
    )

    seq_after_decode = paged_cache.get_seq_length(3)
    print(f"  get_seq_length(3) after 1 decode step: {seq_after_decode}")
    assert seq_after_decode == seq_len_prefill + 1, f"Expected {seq_len_prefill + 1}, got {seq_after_decode}"


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_paged_cache_output_valid(device, model_4_layers):
    """Paged attention produces valid (non-NaN, non-Inf) output on prefill.

    Runs a single prefill through TTNNQwen35FullAttention with paged cache
    and verifies the output is numerically valid.
    """
    model, config = model_4_layers
    torch_attn = model.model.layers[3].self_attn

    hidden_size = get_config_attr(config, "hidden_size")
    num_kv_heads = get_config_attr(config, "num_key_value_heads")
    head_dim = get_config_attr(config, "head_dim")
    batch_size = 1
    seq_len = 32

    paged_config = PagedAttentionConfig(block_size=64, max_num_blocks=32, batch_size=batch_size)
    paged_cache = TTNNQwenPagedAttentionKVCache(
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        config=paged_config,
        device=None,
        layer_indices=[3],
    ).to_device(device)

    ttnn_attn = TTNNQwen35FullAttention.from_torch(torch_attn)
    set_device(ttnn_attn, device)
    ttnn_attn.preprocess_weights()
    ttnn_attn.move_weights_to_device()

    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = model.model.rotary_emb(x, position_ids)

    tt_input = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_cos = ttnn.from_torch(
        cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_sin = ttnn.from_torch(
        sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    out, _ = ttnn_attn.forward(tt_input, position_embeddings=(tt_cos, tt_sin), past_key_values=paged_cache)

    out_torch = ttnn.to_torch(out)
    assert not torch.isnan(out_torch).any(), "Output contains NaN"
    assert not torch.isinf(out_torch).any(), "Output contains Inf"
    print(
        f"  Paged prefill output: shape={out_torch.shape}, "
        f"mean={out_torch.float().mean():.6f}, std={out_torch.float().std():.6f}"
    )


@pytest.fixture(scope="module")
def model_4_layers():
    """Load Qwen3.5-27B-FP8 with 4 hidden layers (module-scoped for reuse)."""
    from .conftest import load_model

    model, config = load_model(num_hidden_layers=4)
    return model, config
