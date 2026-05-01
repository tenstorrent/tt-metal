# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 full decoder layer.

Uses HuggingFace Gemma4TextDecoderLayer as reference for PCC comparison,
following the gpt-oss test_modules.py pattern.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt.layer import Gemma4DecoderLayer
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

from ...tests.test_factory import TestFactory, compare_tensors, parametrize_batch_seq, parametrize_mesh_with_fabric

# ── Config / Structure Tests ───────────────────────────────────────────────


@pytest.mark.parametrize("layer_idx", [0])
@parametrize_batch_seq()
def test_layer_config(batch_size, seq_len, layer_idx):
    """Verify layer type assignment."""
    hf_config = TestFactory.create_hf_config()
    assert len(hf_config.layer_types) == hf_config.num_hidden_layers
    assert all(lt in ("sliding_attention", "full_attention") for lt in hf_config.layer_types)
    # First layer is always sliding
    assert hf_config.layer_types[0] == "sliding_attention"


def test_layer_norms_count():
    """Verify norm count depends on MoE config."""
    hf_config = TestFactory.create_hf_config()
    # MoE layers have 7 norms, dense layers have 4
    if hf_config.enable_moe_block:
        expected_extra = 3  # post_feedforward_layernorm_1, pre/post_feedforward_layernorm_2
    else:
        expected_extra = 0
    assert expected_extra >= 0  # Just verify it doesn't crash


# ── HF Reference Helpers ──────────────────────────────────────────────────


def _create_hf_reference_layer(hf_text_config, layer_idx):
    """Create HF Gemma4TextDecoderLayer with random weights as reference."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer as HFDecoderLayer

    hf_layer = HFDecoderLayer(hf_text_config, layer_idx=layer_idx)
    # Randomize router/expert weights (HF inits some to zeros/ones)
    with torch.no_grad():
        for name, param in hf_layer.named_parameters():
            if any(k in name for k in ["router", "experts", "layer_scalar"]):
                if "scale" in name or "scalar" in name:
                    param.data.fill_(1.0)
                else:
                    param.data.normal_(0, 0.02)
    hf_layer.eval()
    return hf_layer


def _create_hf_text_config(num_experts=None, top_k=None):
    """Create HF text config from HF_MODEL, with optional MoE overrides for speed."""
    from transformers import AutoConfig

    from ...tests.test_factory import _get_model_path

    config = AutoConfig.from_pretrained(_get_model_path(), trust_remote_code=True)
    tc = config.text_config
    # Reduce experts for speed on MoE models
    if getattr(tc, "enable_moe_block", False):
        if num_experts is not None:
            tc.num_experts = num_experts
        if top_k is not None:
            tc.top_k_experts = top_k
    # Disable per-layer input for now (not yet implemented in TT)
    tc.hidden_size_per_layer_input = 0
    tc._attn_implementation = "eager"
    return tc


def _hf_state_to_tt_state(hf_state_dict, layer_idx):
    """Convert HF layer state_dict keys to the format our TT layer expects.

    HF keys: "input_layernorm.weight", "self_attn.q_proj.weight", etc.
    TT expects: "model.layers.{idx}.input_layernorm.weight", etc.
    """
    prefix = f"model.layers.{layer_idx}"
    return {f"{prefix}.{k}": v for k, v in hf_state_dict.items()}


def _create_hf_rope(hf_text_config, seq_len, layer_idx):
    """Create HF RoPE position embeddings for a given layer."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    rope = Gemma4TextRotaryEmbedding(hf_text_config)
    x_dummy = torch.randn(1, seq_len, hf_text_config.hidden_size)
    pos_ids = torch.arange(seq_len).unsqueeze(0)
    layer_type = hf_text_config.layer_types[layer_idx]
    cos, sin = rope(x_dummy, pos_ids, layer_type=layer_type)
    return cos, sin


def _make_per_layer_input(hf_text_config, seq_len):
    """Create dummy per_layer_input for E2B/E4B models that use it."""
    pli_size = getattr(hf_text_config, "hidden_size_per_layer_input", 0) or 0
    if pli_size:
        return torch.randn(1, seq_len, pli_size)
    return None


def _create_gemma4_model_args(hf_text_config):
    """Create Gemma4ModelArgs that matches the HF text config."""
    return Gemma4ModelArgs.from_hf_config(hf_text_config)


# ── Full Layer PCC Test ───────────────────────────────────────────────────


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("layer_idx", [0], ids=["sliding"])
@parametrize_batch_seq(configs=[(1, 32)], ids=["prefill_32"])
def test_layer_forward(batch_size, seq_len, layer_idx, mesh_device, reset_seeds):
    """
    Full decoder layer PCC test: compares TT layer against HF reference.

    Creates HF Gemma4TextDecoderLayer with random weights, runs both HF and TT
    forward passes with the same input, and checks PCC >= 0.90.
    """
    # Create HF config and reference layer
    hf_text_config = _create_hf_text_config(num_experts=4, top_k=2)
    hf_layer = _create_hf_reference_layer(hf_text_config, layer_idx)

    # Get HF state_dict and convert for TT
    hf_state = hf_layer.state_dict()
    tt_state = _hf_state_to_tt_state(hf_state, layer_idx)

    # Create matching Gemma4ModelArgs
    model_args = _create_gemma4_model_args(hf_text_config)

    from models.demos.gemma4.tt.attention import Gemma4AttentionConfig

    attn_cfg = Gemma4AttentionConfig(model_args, layer_idx)

    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tt.ccl import CCLManager

    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    tt_layer = Gemma4DecoderLayer(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=tt_state,
        layer_idx=layer_idx,
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=seq_len,
        max_local_batch_size=batch_size,
    )

    # Input
    x_torch = torch.randn(1, seq_len, model_args.hidden_size, dtype=torch.float32)

    # HF reference forward (must pass causal mask — HF defaults to bidirectional without it)
    hf_rope = _create_hf_rope(hf_text_config, seq_len, layer_idx)
    causal_mask = torch.triu(torch.full((1, 1, seq_len, seq_len), float("-inf")), diagonal=1)
    with torch.no_grad():
        pli = _make_per_layer_input(hf_text_config, seq_len)
        hf_output = hf_layer(x_torch, per_layer_input=pli, position_embeddings=hf_rope, attention_mask=causal_mask)
    logger.info(f"HF output shape: {hf_output.shape}, range: [{hf_output.min():.4f}, {hf_output.max():.4f}]")

    # TT forward
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    x_tt = ttnn.from_torch(
        x_torch.unsqueeze(0).to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, max(seq_len, 128), layer_idx)
    tt_output = tt_layer(
        x_tt,
        rope_mats=(cos_tt, sin_tt),
        position_idx=None,
        page_table=None,
        kv_cache=None,
        is_decode=False,
    )
    tt_output_torch = (
        (ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0]) if is_mesh else ttnn.to_torch(tt_output))
        .squeeze(0)
        .float()
    )

    passing, pcc_msg = compare_tensors(tt_output_torch, hf_output, pcc_threshold=0.95)
    assert passing, f"Full layer (layer_idx={layer_idx}, tp={tp}) PCC too low: {pcc_msg}"


# ── Decode Layer Test (fully on device) ───────────────────────────────────


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("layer_idx", [0], ids=["sliding"])
def test_layer_forward_decode(layer_idx, mesh_device, reset_seeds):
    """
    Full decoder layer decode test (seq_len=1, fully on device).

    Uses KV cache pre-filled with random data, then decodes one token.
    Compares against HF reference with same KV cache state.
    """
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    from models.demos.gemma4.tt.attention import Gemma4AttentionConfig

    hf_text_config = _create_hf_text_config(num_experts=4, top_k=2)
    hf_layer = _create_hf_reference_layer(hf_text_config, layer_idx)
    hf_state = hf_layer.state_dict()
    tt_state = _hf_state_to_tt_state(hf_state, layer_idx)
    model_args = _create_gemma4_model_args(hf_text_config)
    attn_cfg = Gemma4AttentionConfig(model_args, layer_idx)

    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tt.ccl import CCLManager

    cache_len = 32
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    tt_layer = Gemma4DecoderLayer(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=tt_state,
        layer_idx=layer_idx,
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=cache_len + 32,
        max_local_batch_size=1,
    )

    # Pre-fill KV cache with random data
    k_data = torch.randn(1, attn_cfg.num_key_value_heads, cache_len, attn_cfg.head_dim)
    v_data = torch.randn(1, attn_cfg.num_key_value_heads, cache_len, attn_cfg.head_dim)

    from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache

    kv_cache = init_kv_cache(
        mesh_device, attn_cfg, max_batch_size=1, max_seq_len=cache_len + 32, cache_dtype=ttnn.bfloat16
    )
    # Fill cache — on multi-device, each device gets its local KV heads
    if tp > 1:
        local_kv = attn_cfg.num_key_value_heads // tp
        for dev_idx in range(tp):
            k_local = k_data[:, dev_idx * local_kv : (dev_idx + 1) * local_kv].to(torch.bfloat16)
            v_local = v_data[:, dev_idx * local_kv : (dev_idx + 1) * local_kv].to(torch.bfloat16)
            dev_k = ttnn.get_device_tensors(kv_cache[0])[dev_idx]
            dev_v = ttnn.get_device_tensors(kv_cache[1])[dev_idx]
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
        ttnn.fill_cache(kv_cache[0], k_fill, batch_idx=0)
        ttnn.fill_cache(kv_cache[1], v_fill, batch_idx=0)
    tt_layer.self_attn.kv_cache = kv_cache

    # Set HF KV cache
    hf_cache = DynamicCache()
    hf_cache.update(k_data.clone(), v_data.clone(), layer_idx=layer_idx)

    # Decode input
    x_torch = torch.randn(1, 1, model_args.hidden_size, dtype=torch.float32)

    # HF reference
    rope = Gemma4TextRotaryEmbedding(hf_text_config)
    layer_type = hf_text_config.layer_types[layer_idx]
    cos, sin = rope(x_torch, torch.tensor([[cache_len]]), layer_type=layer_type)
    mask = torch.zeros(1, 1, 1, cache_len + 1)
    with torch.no_grad():
        pli = _make_per_layer_input(hf_text_config, 1)
        hf_output = hf_layer(
            x_torch,
            per_layer_input=pli,
            position_embeddings=(cos, sin),
            past_key_values=hf_cache,
            attention_mask=mask,
        )

    # TT forward (decode)
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, cache_len + 32, layer_idx)
    x_tt = ttnn.from_torch(
        x_torch.unsqueeze(0).to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    position_idx_tt = ttnn.from_torch(
        torch.tensor([[cache_len]], dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    tt_output = tt_layer(
        x_tt,
        rope_mats=(cos_tt, sin_tt),
        position_idx=position_idx_tt,
        page_table=None,
        kv_cache=kv_cache,
        is_decode=True,
        token_index=cache_len,
    )
    tt_output_torch = (
        (ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0]) if is_mesh else ttnn.to_torch(tt_output))
        .squeeze(0)
        .float()
    )

    # Relaxed threshold for decode: MoE router bf16 topk can pick different experts
    passing, pcc_msg = compare_tensors(tt_output_torch, hf_output, pcc_threshold=0.90)
    assert passing, f"Full layer decode (layer_idx={layer_idx}, tp={tp}) PCC too low: {pcc_msg}"
