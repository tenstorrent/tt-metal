# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 Attention — uses HF Gemma4TextAttention as reference.

Tests prefill (PCC comparison) and decode (sanity) for both sliding and global layers.
Also includes TP=2 prefill test for N300.
"""

import pytest
import torch

import ttnn
from models.demos.gemma4.tt.attention import Gemma4Attention, Gemma4AttentionConfig

from ...tests.test_factory import TestFactory, compare_tensors

# ── Prefill PCC Test ──────────────────────────────────────────────────────


@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding", "global"])
@pytest.mark.parametrize("seq_len", [32, 128], ids=["seq32", "seq128"])
def test_attention_prefill(layer_idx, seq_len, device):
    """
    Test prefill attention against HF Gemma4TextAttention with PCC >= 0.95.
    """
    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    hf_attn = hf_layer.self_attn
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    hidden_size = config.hidden_size

    # Extract HF attention state_dict
    state_dict = {
        k: v.clone() for k, v in hf_attn.state_dict().items() if not k.startswith("v_norm")
    }  # v_norm has no weight

    # Create TT attention
    mesh_config = TestFactory.create_mesh_config((1, 1))
    tt_attn = Gemma4Attention(
        mesh_device=device,
        config=config,
        state_dict=state_dict,
        ccl_manager=None,
        mesh_config=mesh_config,
        program_config=None,
        layer_idx=layer_idx,
    )

    # Input
    x_torch = torch.randn(1, seq_len, hidden_size, dtype=torch.float32)

    # HF reference forward
    hf_rope = TestFactory.create_hf_rope(hf_text_config, seq_len, layer_idx)
    causal_mask = torch.triu(torch.full((1, 1, seq_len, seq_len), float("-inf")), diagonal=1)
    with torch.no_grad():
        ref_output, _ = hf_attn(x_torch, position_embeddings=hf_rope, attention_mask=causal_mask)

    # TT forward
    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(device, hf_text_config, max(seq_len, 128), layer_idx)
    x_tt = ttnn.from_torch(
        x_torch.unsqueeze(0).to(torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    tt_output = tt_attn(x_tt, rope_mats=(cos_tt, sin_tt), is_decode=False)
    tt_output_torch = ttnn.to_torch(tt_output).squeeze(0).float()

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=0.95)
    assert passing, f"Attention prefill (layer_idx={layer_idx}, seq={seq_len}) PCC too low: {pcc_msg}"


# ── Decode PCC Test ───────────────────────────────────────────────────────


@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding", "global"])
def test_attention_decode(layer_idx, device):
    """
    Test decode attention against HF reference with random KV cache and PCC >= 0.95.

    Sets up identical random KV cache in both HF (DynamicCache) and TT (kv_cache),
    then decodes one token at the next position and compares outputs.
    """
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    hf_attn = hf_layer.self_attn
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    hidden_size = config.hidden_size
    cache_len = 32  # pre-existing KV cache length

    state_dict = {k: v.clone() for k, v in hf_attn.state_dict().items() if not k.startswith("v_norm")}

    mesh_config = TestFactory.create_mesh_config((1, 1))
    tt_attn = Gemma4Attention(
        mesh_device=device,
        config=config,
        state_dict=state_dict,
        ccl_manager=None,
        mesh_config=mesh_config,
        program_config=None,
        layer_idx=layer_idx,
        create_kv_cache=True,
        max_batch_size=1,
        max_seq_len=cache_len + 32,
    )

    # Create random KV cache data [1, num_kv_heads, cache_len, head_dim]
    k_data = torch.randn(1, config.num_key_value_heads, cache_len, config.head_dim)
    v_data = torch.randn(1, config.num_key_value_heads, cache_len, config.head_dim)

    # Set HF cache
    hf_cache = DynamicCache()
    hf_cache.update(k_data.clone(), v_data.clone(), layer_idx=layer_idx)

    # Set TT cache — write k_data/v_data into the TT KV cache at positions 0..cache_len-1
    k_cache_tt, v_cache_tt = tt_attn.kv_cache
    k_for_fill = k_data.to(torch.bfloat16)
    v_for_fill = v_data.to(torch.bfloat16)
    k_fill_tt = ttnn.from_torch(k_for_fill, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    v_fill_tt = ttnn.from_torch(v_for_fill, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ttnn.fill_cache(k_cache_tt, k_fill_tt, batch_idx=0)
    ttnn.fill_cache(v_cache_tt, v_fill_tt, batch_idx=0)

    # Decode input (one new token at position cache_len)
    x_torch = torch.randn(1, 1, hidden_size, dtype=torch.float32)

    # HF reference decode
    rope = Gemma4TextRotaryEmbedding(hf_text_config)
    layer_type = hf_text_config.layer_types[layer_idx]
    cos, sin = rope(x_torch, torch.tensor([[cache_len]]), layer_type=layer_type)
    mask = torch.zeros(1, 1, 1, cache_len + 1)  # no masking for decode
    with torch.no_grad():
        ref_output, _ = hf_attn(x_torch, position_embeddings=(cos, sin), past_key_values=hf_cache, attention_mask=mask)

    # TT decode
    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(device, hf_text_config, cache_len + 32, layer_idx)
    x_tt = ttnn.from_torch(
        x_torch.unsqueeze(0).to(torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    position_idx_tt = ttnn.from_torch(
        torch.tensor([[cache_len]], dtype=torch.int32),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
    )
    tt_output = tt_attn(
        x_tt,
        rope_mats=(cos_tt, sin_tt),
        position_idx=position_idx_tt,
        is_decode=True,
        token_index=cache_len,
    )
    tt_output_torch = ttnn.to_torch(tt_output).squeeze(0).float()  # [1, 1, H]

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=0.95)
    assert passing, f"Attention decode (layer_idx={layer_idx}) PCC too low: {pcc_msg}"


# ── TP=2 Prefill Test ────────────────────────────────────────────────────


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding", "global"])
def test_attention_prefill_tp(layer_idx, mesh_device, device_params):
    """
    Test TP=2 prefill attention against HF reference with PCC >= 0.95.
    """
    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tt.ccl import CCLManager

    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    hf_attn = hf_layer.self_attn
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    hidden_size = config.hidden_size
    seq_len = 32

    state_dict = {k: v.clone() for k, v in hf_attn.state_dict().items() if not k.startswith("v_norm")}

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1]))
    ccl_manager = CCLManager(mesh_device, num_links=1)

    tt_attn = Gemma4Attention(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=None,
        layer_idx=layer_idx,
    )

    # Input (replicated)
    x_torch = torch.randn(1, seq_len, hidden_size, dtype=torch.float32)
    x_tt = ttnn.from_torch(
        x_torch.unsqueeze(0).to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, max(seq_len, 128), layer_idx)

    # TT forward
    tt_output = tt_attn(x_tt, rope_mats=(cos_tt, sin_tt), is_decode=False)
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0]).squeeze(0).float()

    # HF reference
    hf_rope = TestFactory.create_hf_rope(hf_text_config, seq_len, layer_idx)
    causal_mask = torch.triu(torch.full((1, 1, seq_len, seq_len), float("-inf")), diagonal=1)
    with torch.no_grad():
        ref_output, _ = hf_attn(x_torch, position_embeddings=hf_rope, attention_mask=causal_mask)

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=0.95)
    assert passing, f"Attention TP prefill (layer_idx={layer_idx}) PCC too low: {pcc_msg}"


# ── TP=2 Decode PCC Test ─────────────────────────────────────────────────


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding", "global"])
def test_attention_decode_tp(layer_idx, mesh_device, device_params):
    """
    Test TP=2 decode attention against HF reference with random KV cache and PCC >= 0.95.
    """
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tt.ccl import CCLManager

    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    hf_attn = hf_layer.self_attn
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    hidden_size = config.hidden_size
    cache_len = 32

    state_dict = {k: v.clone() for k, v in hf_attn.state_dict().items() if not k.startswith("v_norm")}

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1]))
    ccl_manager = CCLManager(mesh_device, num_links=1)

    tt_attn = Gemma4Attention(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=None,
        layer_idx=layer_idx,
        create_kv_cache=True,
        max_batch_size=1,
        max_seq_len=cache_len + 32,
    )

    # Random KV cache
    k_data = torch.randn(1, config.num_key_value_heads, cache_len, config.head_dim)
    v_data = torch.randn(1, config.num_key_value_heads, cache_len, config.head_dim)

    # HF cache (full heads)
    hf_cache = DynamicCache()
    hf_cache.update(k_data.clone(), v_data.clone(), layer_idx=layer_idx)

    # TT cache — each device has local_kv_heads. Fill each device's cache separately.
    k_cache_tt, v_cache_tt = tt_attn.kv_cache
    local_kv = config.num_key_value_heads // mesh_config.tp
    for dev_idx in range(mesh_config.tp):
        k_local = k_data[:, dev_idx * local_kv : (dev_idx + 1) * local_kv].to(torch.bfloat16)
        v_local = v_data[:, dev_idx * local_kv : (dev_idx + 1) * local_kv].to(torch.bfloat16)
        dev_k_cache = ttnn.get_device_tensors(k_cache_tt)[dev_idx]
        dev_v_cache = ttnn.get_device_tensors(v_cache_tt)[dev_idx]
        k_fill = ttnn.from_torch(k_local, device=dev_k_cache.device(), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        v_fill = ttnn.from_torch(v_local, device=dev_v_cache.device(), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        ttnn.fill_cache(dev_k_cache, k_fill, batch_idx=0)
        ttnn.fill_cache(dev_v_cache, v_fill, batch_idx=0)

    # Decode input
    x_torch = torch.randn(1, 1, hidden_size, dtype=torch.float32)

    # HF reference
    rope = Gemma4TextRotaryEmbedding(hf_text_config)
    layer_type = hf_text_config.layer_types[layer_idx]
    cos, sin = rope(x_torch, torch.tensor([[cache_len]]), layer_type=layer_type)
    mask = torch.zeros(1, 1, 1, cache_len + 1)
    with torch.no_grad():
        ref_output, _ = hf_attn(x_torch, position_embeddings=(cos, sin), past_key_values=hf_cache, attention_mask=mask)

    # TT decode
    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, cache_len + 32, layer_idx)
    x_tt = ttnn.from_torch(
        x_torch.unsqueeze(0).to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    position_idx_tt = ttnn.from_torch(
        torch.tensor([[cache_len]], dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_output = tt_attn(
        x_tt,
        rope_mats=(cos_tt, sin_tt),
        position_idx=position_idx_tt,
        is_decode=True,
        token_index=cache_len,
    )
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0]).squeeze(0).float()

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=0.95)
    assert passing, f"Attention TP decode (layer_idx={layer_idx}) PCC too low: {pcc_msg}"
