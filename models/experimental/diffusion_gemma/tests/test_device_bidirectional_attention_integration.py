# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device integration test for the real Gemma4 masked non-causal prefill path."""

import os

import pytest

if os.environ.get("DG_RUN_DEVICE") != "1":
    pytest.skip("set DG_RUN_DEVICE=1 to run QB2 bidirectional attention integration tests", allow_module_level=True)

import torch

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tests.test_factory import find_layer_idx, num_layers_for_full_attention_group
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4.tests.unit.test_model import (
    _create_hf_model,
    _create_hf_text_config,
    _hf_model_state_to_tt_state,
)
from models.demos.gemma4.tt.attention.kv_phase import KVCachePhase
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.model import Gemma4Model
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
from models.experimental.diffusion_gemma.reference.attention_mask import build_canvas_denoise_mask
from models.experimental.diffusion_gemma.tt.denoise_forward import denoise_attention_forward
from tests.ttnn.utils_for_testing import assert_with_pcc
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding, apply_rotary_pos_emb


pytestmark = pytest.mark.use_module_device
NEG = -1.0e9


def _build_tt_model(mesh_device, hf_model, hf_text_config, *, num_layers, max_seq_len):
    model_args = Gemma4ModelArgs.from_hf_config(hf_text_config)
    model_args._hf_text_config = hf_text_config
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    return Gemma4Model(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=_hf_model_state_to_tt_state(hf_model),
        ccl_manager=CCLManager(mesh_device, num_links=1) if tp > 1 else None,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=max_seq_len,
        max_local_batch_size=1,
        num_layers=num_layers,
        create_kv_cache=True,
    )


def _mesh_mapper(mesh_device):
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    return ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None


def _to_torch(tt_tensor, mesh_device):
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    return ttnn.to_torch(ttnn.get_device_tensors(tt_tensor)[0]) if is_mesh else ttnn.to_torch(tt_tensor)


def _to_device(mesh_device, value):
    return ttnn.from_torch(
        value,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=_mesh_mapper(mesh_device),
    )


def _torch_attention_reference(hf_model, hf_text_config, layer_idx, canvas_hidden, kv_hidden, mask):
    layer_type = hf_text_config.layer_types[layer_idx]
    attn = hf_model.layers[layer_idx].self_attn
    head_dim = attn.head_dim
    q_shape = (*canvas_hidden.shape[:-1], -1, head_dim)
    kv_shape = (*kv_hidden.shape[:-1], -1, head_dim)

    rope = Gemma4TextRotaryEmbedding(hf_text_config)
    pos_ids = torch.arange(kv_hidden.shape[1]).unsqueeze(0)
    cos, sin = rope(kv_hidden, pos_ids, layer_type=layer_type)
    q_cos = cos[:, -canvas_hidden.shape[1] :, :]
    q_sin = sin[:, -canvas_hidden.shape[1] :, :]

    query = attn.q_norm(attn.q_proj(canvas_hidden).view(q_shape))
    query = apply_rotary_pos_emb(query, q_cos, q_sin, unsqueeze_dim=2).transpose(1, 2)

    key_linear = attn.k_proj(kv_hidden).view(kv_shape)
    value_linear = attn.v_proj(kv_hidden).view(kv_shape) if attn.v_proj is not None else key_linear
    key = attn.k_norm(key_linear)
    key = apply_rotary_pos_emb(key, cos, sin, unsqueeze_dim=2).transpose(1, 2)
    value = attn.v_norm(value_linear).transpose(1, 2)

    if attn.num_key_value_groups != 1:
        key = key.repeat_interleave(attn.num_key_value_groups, dim=1)
        value = value.repeat_interleave(attn.num_key_value_groups, dim=1)
    out = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, is_causal=False, scale=1.0
    )
    out = out.transpose(1, 2).reshape(canvas_hidden.shape[0], canvas_hidden.shape[1], -1)
    return attn.o_proj(out)


@parametrize_mesh_with_fabric([(1, 4)])
def test_real_attention_prefill_accepts_all_attend_noncausal_mask(mesh_device, reset_seeds):
    torch.manual_seed(4)
    seq_len = 32
    vocab_size = 256

    hf_text_config = _create_hf_text_config(vocab_size=vocab_size, num_layers=1)
    if getattr(hf_text_config, "enable_moe_block", False):
        hf_text_config.num_experts = 4
        hf_text_config.top_k_experts = 2
    hf_model = _create_hf_model(hf_text_config)
    model_args = Gemma4ModelArgs.from_hf_config(hf_text_config)
    model_args._hf_text_config = hf_text_config

    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    tt_model = Gemma4Model(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=_hf_model_state_to_tt_state(hf_model),
        ccl_manager=CCLManager(mesh_device, num_links=1) if tp > 1 else None,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=seq_len,
        max_local_batch_size=1,
        num_layers=1,
        create_kv_cache=True,
    )

    tokens = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
    attn_mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.float32)
    with torch.no_grad():
        hf_logits = hf_model(tokens, attention_mask=attn_mask)

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    tt_tokens = ttnn.from_torch(
        tokens.to(torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=mesh_mapper,
    )
    tt_embeds = tt_model.embed_tokens(tt_tokens)
    tt_embeds = ttnn.reshape(tt_embeds, (1, 1, seq_len, model_args.hidden_size))
    tt_embeds = ttnn.to_layout(tt_embeds, ttnn.TILE_LAYOUT)
    tt_mask = ttnn.from_torch(
        attn_mask,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=mesh_mapper,
    )
    tt_logits = tt_model(
        tt_embeds,
        is_decode=False,
        input_ids_torch=tokens,
        kv_phase=KVCachePhase.DENOISE_READONLY,
        attn_mask=tt_mask,
    )
    tt_logits_torch = (
        ttnn.to_torch(ttnn.get_device_tensors(tt_logits)[0]) if is_mesh else ttnn.to_torch(tt_logits)
    ).squeeze(0)

    passing, message = assert_with_pcc(hf_logits.float(), tt_logits_torch.float(), 0.99)
    assert passing, message


@parametrize_mesh_with_fabric([(1, 4)])
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"])
def test_real_attention_denoise_mask_covers_prompt_prefix_for_layer_type(mesh_device, layer_type, reset_seeds):
    torch.manual_seed(5)
    prompt_len = 64
    canvas_len = 256
    total_len = prompt_len + canvas_len

    base_config = _create_hf_text_config(vocab_size=256, num_layers=1)
    num_layers = 1 if layer_type == "sliding_attention" else num_layers_for_full_attention_group(base_config)
    hf_text_config = _create_hf_text_config(vocab_size=256, num_layers=num_layers)
    if getattr(hf_text_config, "enable_moe_block", False):
        hf_text_config.num_experts = 4
        hf_text_config.top_k_experts = 2
    hf_model = _create_hf_model(hf_text_config)
    tt_model = _build_tt_model(mesh_device, hf_model, hf_text_config, num_layers=num_layers, max_seq_len=total_len)
    layer_idx = find_layer_idx(hf_text_config, layer_type)

    prompt_hidden = torch.randn(1, prompt_len, hf_text_config.hidden_size)
    canvas_hidden = torch.randn(1, canvas_len, hf_text_config.hidden_size)
    kv_hidden = torch.cat([prompt_hidden, canvas_hidden], dim=1)
    mask = build_canvas_denoise_mask(
        prompt_len,
        canvas_len,
        local_window=False,
        neg_inf=NEG,
        dtype=torch.float32,
    ).view(1, 1, canvas_len, total_len)
    with torch.no_grad():
        golden = _torch_attention_reference(hf_model, hf_text_config, layer_idx, canvas_hidden, kv_hidden, mask)

    tt_canvas_hidden = _to_device(mesh_device, canvas_hidden.unsqueeze(0))
    tt_prompt_hidden = _to_device(mesh_device, prompt_hidden.unsqueeze(0))
    tt_out = denoise_attention_forward(
        tt_model,
        layer_idx=layer_idx,
        prompt_hidden=tt_prompt_hidden,
        canvas_hidden=tt_canvas_hidden,
    )
    out = _to_torch(tt_out, mesh_device).squeeze(0)

    passing, message = assert_with_pcc(golden.float(), out.float(), 0.99)
    assert passing, message
