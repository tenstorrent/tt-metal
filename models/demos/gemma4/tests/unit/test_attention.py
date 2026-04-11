# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 Attention — uses HF Gemma4TextAttention as reference.

Tests prefill and decode for both sliding and global layers, across all TP factors.

    pytest -k "1x1"              # single card
    pytest -k "1x8"              # T3K
    pytest -k "sliding"          # sliding attention only
    pytest -k "global"           # global attention only
    pytest -k "prefill"          # prefill only
    pytest -k "decode"           # decode only
"""

import pytest
import torch

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.attention import Gemma4Attention, Gemma4AttentionConfig
from models.demos.gemma4.tt.ccl import CCLManager

from ...tests.test_factory import TestFactory, compare_tensors, parametrize_mesh_with_fabric


def _skip_if_l1_overflow(config, mesh_device):
    """Skip if global attention head_dim overflows L1 on this mesh config."""
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    # Global layers with large head_dim (512) overflow L1 when hidden_size > 4096 on single device
    if not config.is_sliding and config.head_dim >= 512 and tp == 1:
        hf_config = TestFactory.create_hf_config()
        if hf_config.hidden_size > 4096:
            pytest.skip("Global attention head_dim=512 overflows L1 on single device for large models")


def _setup_attention(mesh_device, layer_idx, create_kv_cache=False, max_seq_len=128):
    """Create HF reference and TT attention module for a given mesh."""
    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    hf_attn = hf_layer.self_attn
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)

    state_dict = {k: v.clone() for k, v in hf_attn.state_dict().items() if not k.startswith("v_norm")}

    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    tt_attn = Gemma4Attention(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=None,
        layer_idx=layer_idx,
        create_kv_cache=create_kv_cache,
        max_batch_size=1,
        max_seq_len=max_seq_len,
    )

    return hf_text_config, hf_attn, config, tt_attn, mesh_config


def _to_device(tensor, mesh_device):
    """Send tensor to mesh device with appropriate mapper."""
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )


def _from_device(tensor, mesh_device):
    """Read tensor back from device 0."""
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    if is_mesh:
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])
    return ttnn.to_torch(tensor)


# ── Prefill PCC Test ──────────────────────────────────────────────────────


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding", "global"])
@pytest.mark.parametrize("seq_len", [32], ids=["seq32"])
def test_attention_prefill(layer_idx, seq_len, mesh_device):
    """Test prefill attention against HF reference with PCC >= 0.95."""
    hf_text_config, hf_attn, config, tt_attn, mesh_config = _setup_attention(mesh_device, layer_idx)
    _skip_if_l1_overflow(config, mesh_device)

    x_torch = torch.randn(1, seq_len, config.hidden_size, dtype=torch.float32)

    # HF reference
    hf_rope = TestFactory.create_hf_rope(hf_text_config, seq_len, layer_idx)
    causal_mask = torch.triu(torch.full((1, 1, seq_len, seq_len), float("-inf")), diagonal=1)
    with torch.no_grad():
        ref_output, _ = hf_attn(x_torch, position_embeddings=hf_rope, attention_mask=causal_mask)

    # TT forward
    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, max(seq_len, 128), layer_idx)
    x_tt = _to_device(x_torch.unsqueeze(0).to(torch.bfloat16), mesh_device)
    tt_output = tt_attn(x_tt, rope_mats=(cos_tt, sin_tt), is_decode=False)
    tt_output_torch = _from_device(tt_output, mesh_device).squeeze(0).float()

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=0.95)
    assert passing, f"Attention prefill (layer={layer_idx}, seq={seq_len}, tp={mesh_config.tp}) PCC too low: {pcc_msg}"


# ── Decode PCC Test ───────────────────────────────────────────────────────


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding", "global"])
def test_attention_decode(layer_idx, mesh_device):
    """Test decode attention against HF reference with random KV cache and PCC >= 0.95."""
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    cache_len = 32
    hf_text_config, hf_attn, config, tt_attn, mesh_config = _setup_attention(
        mesh_device, layer_idx, create_kv_cache=True, max_seq_len=cache_len + 32
    )
    _skip_if_l1_overflow(config, mesh_device)

    tp = mesh_config.tp

    # Random KV cache [1, num_kv_heads, cache_len, head_dim]
    k_data = torch.randn(1, config.num_key_value_heads, cache_len, config.head_dim)
    v_data = torch.randn(1, config.num_key_value_heads, cache_len, config.head_dim)

    # HF cache
    hf_cache = DynamicCache()
    hf_cache.update(k_data.clone(), v_data.clone(), layer_idx=layer_idx)

    # TT cache — fill each device with its local KV heads
    k_cache_tt, v_cache_tt = tt_attn.kv_cache
    kv_replicated = config.num_key_value_heads < tp
    q_per_device = config.num_attention_heads // tp if tp > 1 else config.num_attention_heads

    if tp > 1:
        for dev_idx in range(tp):
            if kv_replicated:
                kv_idx = (dev_idx * q_per_device) * config.num_key_value_heads // config.num_attention_heads
                k_local = k_data[:, kv_idx : kv_idx + 1].to(torch.bfloat16)
                v_local = v_data[:, kv_idx : kv_idx + 1].to(torch.bfloat16)
            else:
                local_kv = config.num_key_value_heads // tp
                k_local = k_data[:, dev_idx * local_kv : (dev_idx + 1) * local_kv].to(torch.bfloat16)
                v_local = v_data[:, dev_idx * local_kv : (dev_idx + 1) * local_kv].to(torch.bfloat16)
            dev_k = ttnn.get_device_tensors(k_cache_tt)[dev_idx]
            dev_v = ttnn.get_device_tensors(v_cache_tt)[dev_idx]
            ttnn.fill_cache(
                dev_k,
                ttnn.from_torch(k_local, device=dev_k.device(), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
                batch_idx=0,
            )
            ttnn.fill_cache(
                dev_v,
                ttnn.from_torch(v_local, device=dev_v.device(), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
                batch_idx=0,
            )
    else:
        k_fill = ttnn.from_torch(
            k_data.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        v_fill = ttnn.from_torch(
            v_data.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        ttnn.fill_cache(k_cache_tt, k_fill, batch_idx=0)
        ttnn.fill_cache(v_cache_tt, v_fill, batch_idx=0)

    # Decode input
    x_torch = torch.randn(1, 1, config.hidden_size, dtype=torch.float32)

    # HF reference
    rope = Gemma4TextRotaryEmbedding(hf_text_config)
    layer_type = hf_text_config.layer_types[layer_idx]
    cos, sin = rope(x_torch, torch.tensor([[cache_len]]), layer_type=layer_type)
    mask = torch.zeros(1, 1, 1, cache_len + 1)
    with torch.no_grad():
        ref_output, _ = hf_attn(x_torch, position_embeddings=(cos, sin), past_key_values=hf_cache, attention_mask=mask)

    # TT decode
    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, cache_len + 32, layer_idx)
    x_tt = _to_device(x_torch.unsqueeze(0).to(torch.bfloat16), mesh_device)
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    position_idx_tt = ttnn.from_torch(
        torch.tensor([[cache_len]], dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    tt_output = tt_attn(
        x_tt,
        rope_mats=(cos_tt, sin_tt),
        position_idx=position_idx_tt,
        is_decode=True,
        token_index=cache_len,
    )
    tt_output_torch = _from_device(tt_output, mesh_device).squeeze(0).float()

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=0.95)
    assert passing, f"Attention decode (layer={layer_idx}, tp={tp}) PCC too low: {pcc_msg}"


# ── Paged Attention Test ─────────────────────────────────────────────────


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("layer_idx", [0], ids=["sliding"])
def test_attention_decode_paged(layer_idx, mesh_device):
    """Test decode attention with paged KV cache against HF reference."""
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    from models.tt_transformers.tt.common import PagedAttentionConfig

    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    hf_attn = hf_layer.self_attn
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    _skip_if_l1_overflow(config, mesh_device)

    hidden_size = config.hidden_size
    cache_len = 32
    state_dict = {k: v.clone() for k, v in hf_attn.state_dict().items() if not k.startswith("v_norm")}

    block_size = 32
    max_num_blocks = 4
    paged_attention_config = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=1))
    from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache

    kv_cache = init_kv_cache(
        mesh_device=mesh_device, config=config, paged_attention_config=paged_attention_config, cache_dtype=ttnn.bfloat16
    )

    tt_attn = Gemma4Attention(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        ccl_manager=None,
        mesh_config=mesh_config,
        program_config=None,
        layer_idx=layer_idx,
    )
    tt_attn.kv_cache = kv_cache

    k_data = torch.randn(1, config.num_key_value_heads, cache_len, config.head_dim)
    v_data = torch.randn(1, config.num_key_value_heads, cache_len, config.head_dim)

    hf_cache = DynamicCache()
    hf_cache.update(k_data.clone(), v_data.clone(), layer_idx=layer_idx)

    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(1, max_num_blocks)
    page_table_tt = ttnn.from_torch(page_table, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)

    k_cache_tt, v_cache_tt = kv_cache
    k_fill = ttnn.from_torch(
        k_data.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    v_fill = ttnn.from_torch(
        v_data.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    ttnn.experimental.paged_fill_cache(k_cache_tt, k_fill, page_table_tt, batch_idx=0)
    ttnn.experimental.paged_fill_cache(v_cache_tt, v_fill, page_table_tt, batch_idx=0)

    x_torch = torch.randn(1, 1, hidden_size, dtype=torch.float32)

    rope = Gemma4TextRotaryEmbedding(hf_text_config)
    layer_type = hf_text_config.layer_types[layer_idx]
    cos, sin = rope(x_torch, torch.tensor([[cache_len]]), layer_type=layer_type)
    mask = torch.zeros(1, 1, 1, cache_len + 1)
    with torch.no_grad():
        ref_output, _ = hf_attn(x_torch, position_embeddings=(cos, sin), past_key_values=hf_cache, attention_mask=mask)

    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, 128, layer_idx)
    x_tt = ttnn.from_torch(
        x_torch.unsqueeze(0).to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    position_idx_tt = ttnn.from_torch(
        torch.tensor([[cache_len]], dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
    )
    tt_output = tt_attn(
        x_tt,
        rope_mats=(cos_tt, sin_tt),
        position_idx=position_idx_tt,
        is_decode=True,
        token_index=cache_len,
        page_table=page_table_tt,
    )
    tt_output_torch = ttnn.to_torch(tt_output).squeeze(0).float()

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=0.95)
    assert passing, f"Attention paged decode (layer={layer_idx}) PCC too low: {pcc_msg}"
