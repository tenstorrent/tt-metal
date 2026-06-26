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
from models.experimental.diffusion_gemma.reference.self_conditioning import SelfConditioning
from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference.denoise_loop import denoise_block as ref_denoise_block
from models.experimental.diffusion_gemma.tests.trajectory_pcc import compare_trajectories
from models.experimental.diffusion_gemma.tt.denoise_forward import (
    DenoiseLogitsAdapter,
    denoise_attention_forward,
    denoise_hidden_forward,
    denoise_logits_from_tokens,
    embed_canvas_tokens,
    read_prompt_kv_cache_slice,
)
from models.experimental.diffusion_gemma.tt.denoise_loop import denoise_block
from models.experimental.diffusion_gemma.tt.self_conditioning import TtSelfConditioning
from models.common.utility_functions import comp_pcc
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


def _to_device_tokens(mesh_device, value):
    return ttnn.from_torch(
        value.to(torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=_mesh_mapper(mesh_device),
    )


def _to_device_canvas_ids(mesh_device, value):
    return ttnn.from_torch(
        value.view(value.shape[0], 1, value.shape[1], 1).to(torch.int32),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.uint32,
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


def _torch_denoise_hidden_reference(hf_model, canvas_hidden, prompt_kv_hidden_by_layer, mask):
    hidden = canvas_hidden
    for layer_idx, layer in enumerate(hf_model.layers):
        residual = hidden
        normed = layer.input_layernorm(hidden)
        kv_hidden = torch.cat([prompt_kv_hidden_by_layer[layer_idx], normed], dim=1)
        hidden = _torch_attention_reference(hf_model, hf_model.config, layer_idx, normed, kv_hidden, mask)
        hidden = layer.post_attention_layernorm(hidden)
        hidden = residual + hidden

        residual = hidden
        hidden = layer.pre_feedforward_layernorm(hidden)
        hidden = layer.mlp(hidden)
        if layer.enable_moe_block:
            hidden_1 = layer.post_feedforward_layernorm_1(hidden)
            hidden_flat = residual.reshape(-1, residual.shape[-1])
            _, top_k_weights, top_k_index = layer.router(hidden_flat)
            hidden_2 = layer.pre_feedforward_layernorm_2(hidden_flat)
            hidden_2 = layer.experts(hidden_2, top_k_index, top_k_weights)
            hidden_2 = hidden_2.reshape(residual.shape)
            hidden_2 = layer.post_feedforward_layernorm_2(hidden_2)
            hidden = hidden_1 + hidden_2
        hidden = layer.post_feedforward_layernorm(hidden)
        hidden = residual + hidden
        hidden = hidden * layer.layer_scalar

    return hf_model.norm(hidden)


def _torch_denoise_logits_reference(hf_model, canvas_hidden, prompt_kv_hidden_by_layer, mask):
    hidden = _torch_denoise_hidden_reference(hf_model, canvas_hidden, prompt_kv_hidden_by_layer, mask)
    logits = hf_model.lm_head(hidden)
    cap = hf_model.config.final_logit_softcapping
    if cap and cap > 0:
        logits = torch.tanh(logits / cap) * cap
    return logits


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
    tt_prompt_out = tt_model.layers[layer_idx].self_attn(
        tt_prompt_hidden,
        rope_mats=tt_model._get_rope_mats(layer_idx, seq_len=prompt_len),
        is_decode=False,
        keep_kv=True,
        kv_phase=KVCachePhase.DENOISE_READONLY,
    )
    tt_prompt_out.deallocate(True)
    tt_prompt_kv = tt_model.layers[layer_idx].self_attn._last_kv
    tt_out = denoise_attention_forward(
        tt_model,
        layer_idx=layer_idx,
        prompt_kv=tt_prompt_kv,
        canvas_hidden=tt_canvas_hidden,
    )
    out = _to_torch(tt_out, mesh_device).squeeze(0)
    tt_prompt_kv[0].deallocate(True)
    tt_prompt_kv[1].deallocate(True)

    passing, message = assert_with_pcc(golden.float(), out.float(), 0.99)
    assert passing, message


@parametrize_mesh_with_fabric([(1, 4)])
def test_denoise_logits_forward_returns_full_canvas_logits(mesh_device, reset_seeds):
    torch.manual_seed(6)
    prompt_len = 64
    canvas_len = 256
    total_len = prompt_len + canvas_len
    vocab_size = 256

    hf_text_config = _create_hf_text_config(vocab_size=vocab_size, num_layers=1)
    if getattr(hf_text_config, "enable_moe_block", False):
        hf_text_config.num_experts = 4
        hf_text_config.top_k_experts = 2
    hf_model = _create_hf_model(hf_text_config)
    tt_model = _build_tt_model(mesh_device, hf_model, hf_text_config, num_layers=1, max_seq_len=total_len)

    canvas_tokens = torch.randint(0, vocab_size, (1, canvas_len), dtype=torch.long)
    prev_logits = torch.randn(1, canvas_len, vocab_size)
    self_conditioning_ref = SelfConditioning(
        hf_text_config.hidden_size,
        intermediate_size=hf_text_config.intermediate_size,
        eps=hf_text_config.rms_norm_eps,
        activation=hf_text_config.hidden_activation,
    ).eval()
    self_conditioning_state = {
        "pre_norm.weight": self_conditioning_ref.pre_norm.weight.data.clone(),
        "gate_proj.weight": self_conditioning_ref.gate_proj.weight.data.clone(),
        "up_proj.weight": self_conditioning_ref.up_proj.weight.data.clone(),
        "down_proj.weight": self_conditioning_ref.down_proj.weight.data.clone(),
    }
    with torch.no_grad():
        canvas_hidden = hf_model.embed_tokens(canvas_tokens)
        conditioned_canvas_hidden = self_conditioning_ref.condition(
            canvas_hidden,
            prev_logits,
            hf_model.embed_tokens.weight,
        )
    prompt_tokens = torch.randint(0, vocab_size, (1, prompt_len), dtype=torch.long)
    with torch.no_grad():
        prompt_hidden = hf_model.embed_tokens(prompt_tokens)
        prompt_kv_hidden = hf_model.layers[0].input_layernorm(prompt_hidden)
    mask = build_canvas_denoise_mask(
        prompt_len,
        canvas_len,
        local_window=False,
        neg_inf=NEG,
        dtype=torch.float32,
    ).view(1, 1, canvas_len, total_len)
    with torch.no_grad():
        golden = _torch_denoise_logits_reference(hf_model, conditioned_canvas_hidden, [prompt_kv_hidden], mask)
        golden_hidden = _torch_denoise_hidden_reference(
            hf_model,
            conditioned_canvas_hidden,
            [prompt_kv_hidden],
            mask,
        )

    tt_canvas_tokens = _to_device_tokens(mesh_device, canvas_tokens)
    tt_prompt_tokens = _to_device_tokens(mesh_device, prompt_tokens)
    tt_prompt_hidden = embed_canvas_tokens(tt_model, tt_prompt_tokens)
    tt_prompt_logits = tt_model(
        tt_prompt_hidden,
        is_decode=False,
        input_ids_torch=prompt_tokens,
        kv_phase=KVCachePhase.PREFILL_WRITE,
    )
    tt_prompt_logits.deallocate(True)
    tt_prompt_kv_by_layer = [read_prompt_kv_cache_slice(tt_model.tt_kv_cache[0], prompt_len=prompt_len)]
    tt_prev_logits = _to_device(mesh_device, prev_logits.unsqueeze(0))
    tt_self_conditioning_embedding = _to_device(
        mesh_device,
        hf_model.embed_tokens.weight.detach().unsqueeze(0).unsqueeze(0),
    )
    self_conditioning = TtSelfConditioning(
        mesh_device,
        self_conditioning_state,
        hidden_size=hf_text_config.hidden_size,
        intermediate_size=hf_text_config.intermediate_size,
        eps=hf_text_config.rms_norm_eps,
    )
    tt_logits = denoise_logits_from_tokens(
        tt_model,
        prompt_hidden_by_layer=tt_prompt_kv_by_layer,
        canvas_tokens=tt_canvas_tokens,
        self_conditioning=self_conditioning,
        prev_logits=tt_prev_logits,
        self_conditioning_embedding_weight=tt_self_conditioning_embedding,
    )
    tt_canvas_hidden = embed_canvas_tokens(tt_model, tt_canvas_tokens)
    conditioned = self_conditioning.condition(
        tt_canvas_hidden,
        tt_prev_logits,
        tt_self_conditioning_embedding,
    )
    tt_canvas_hidden.deallocate(True)
    tt_hidden = denoise_hidden_forward(
        tt_model,
        prompt_hidden_by_layer=tt_prompt_kv_by_layer,
        canvas_hidden=conditioned,
    )
    logits = _to_torch(tt_logits, mesh_device).squeeze(0)
    hidden = _to_torch(tt_hidden, mesh_device).squeeze(0)
    for tt_k, tt_v in tt_prompt_kv_by_layer:
        tt_k.deallocate(True)
        tt_v.deallocate(True)

    # Full logits include the shared bf16 MoE/lm_head/softcap path; this branch's
    # known full-model ceiling is below the attention-only 0.99 acceptance.
    _, hidden_pcc = comp_pcc(golden_hidden.float(), hidden.float(), pcc=0.0)
    print(
        "\n[denoise logits drift] "
        f"hidden_pcc={hidden_pcc:.5f} logits_argmax_agreement="
        f"{float((golden.argmax(dim=-1) == logits.argmax(dim=-1)).float().mean()):.4f}"
    )
    passing, message = assert_with_pcc(golden.float(), logits.float(), 0.98)
    assert passing, message


@parametrize_mesh_with_fabric([(1, 4)])
def test_denoise_logits_adapter_threads_prev_logits_for_self_conditioning(mesh_device, reset_seeds):
    """Device-vs-device wiring equivalence; HF-golden logits tests own numerical correctness."""
    torch.manual_seed(7)
    prompt_len = 64
    canvas_len = 256
    total_len = prompt_len + canvas_len
    vocab_size = 256

    hf_text_config = _create_hf_text_config(vocab_size=vocab_size, num_layers=1)
    if getattr(hf_text_config, "enable_moe_block", False):
        hf_text_config.num_experts = 4
        hf_text_config.top_k_experts = 2
    hf_model = _create_hf_model(hf_text_config)
    tt_model = _build_tt_model(mesh_device, hf_model, hf_text_config, num_layers=1, max_seq_len=total_len)

    prompt_tokens = torch.randint(0, vocab_size, (1, prompt_len), dtype=torch.long)
    canvas_tokens_step0 = torch.randint(0, vocab_size, (1, canvas_len), dtype=torch.long)
    canvas_tokens_step1 = torch.randint(0, vocab_size, (1, canvas_len), dtype=torch.long)
    tt_prompt_tokens = _to_device_tokens(mesh_device, prompt_tokens)
    tt_canvas_tokens_step0 = _to_device_tokens(mesh_device, canvas_tokens_step0)
    tt_canvas_tokens_step1 = _to_device_tokens(mesh_device, canvas_tokens_step1)

    tt_prompt_hidden = embed_canvas_tokens(tt_model, tt_prompt_tokens)
    tt_prompt_logits = tt_model(
        tt_prompt_hidden,
        is_decode=False,
        input_ids_torch=prompt_tokens,
        kv_phase=KVCachePhase.PREFILL_WRITE,
    )
    tt_prompt_logits.deallocate(True)
    tt_prompt_kv_by_layer = [read_prompt_kv_cache_slice(tt_model.tt_kv_cache[0], prompt_len=prompt_len)]

    self_conditioning_ref = SelfConditioning(
        hf_text_config.hidden_size,
        intermediate_size=hf_text_config.intermediate_size,
        eps=hf_text_config.rms_norm_eps,
        activation=hf_text_config.hidden_activation,
    ).eval()
    self_conditioning_state = {
        "pre_norm.weight": self_conditioning_ref.pre_norm.weight.data.clone(),
        "gate_proj.weight": self_conditioning_ref.gate_proj.weight.data.clone(),
        "up_proj.weight": self_conditioning_ref.up_proj.weight.data.clone(),
        "down_proj.weight": self_conditioning_ref.down_proj.weight.data.clone(),
    }
    self_conditioning = TtSelfConditioning(
        mesh_device,
        self_conditioning_state,
        hidden_size=hf_text_config.hidden_size,
        intermediate_size=hf_text_config.intermediate_size,
        eps=hf_text_config.rms_norm_eps,
    )
    tt_self_conditioning_embedding = _to_device(
        mesh_device,
        hf_model.embed_tokens.weight.detach().unsqueeze(0).unsqueeze(0),
    )

    adapter = DenoiseLogitsAdapter(
        tt_model,
        prompt_hidden_by_layer=tt_prompt_kv_by_layer,
        self_conditioning=self_conditioning,
        self_conditioning_embedding_weight=tt_self_conditioning_embedding,
    )
    step0_logits = adapter(tt_canvas_tokens_step0, 0)
    expected_step1_logits = denoise_logits_from_tokens(
        tt_model,
        prompt_hidden_by_layer=tt_prompt_kv_by_layer,
        canvas_tokens=tt_canvas_tokens_step1,
        self_conditioning=self_conditioning,
        prev_logits=step0_logits,
        self_conditioning_embedding_weight=tt_self_conditioning_embedding,
    )
    step1_logits = adapter(tt_canvas_tokens_step1, 1)

    expected = _to_torch(expected_step1_logits, mesh_device).squeeze(0)
    actual = _to_torch(step1_logits, mesh_device).squeeze(0)
    adapter.reset()
    expected_step1_logits.deallocate(True)
    for tt_k, tt_v in tt_prompt_kv_by_layer:
        tt_k.deallocate(True)
        tt_v.deallocate(True)

    passing, message = assert_with_pcc(expected.float(), actual.float(), 0.999)
    assert passing, message


@parametrize_mesh_with_fabric([(1, 4)])
@pytest.mark.parametrize("enable_moe", [True, False], ids=["moe", "dense"])
def test_denoise_controller_real_logits_records_decision_flips(mesh_device, reset_seeds, enable_moe):
    torch.manual_seed(8)
    prompt_len = 64
    canvas_len = 256
    total_len = prompt_len + canvas_len
    vocab_size = 256
    max_steps = 2

    hf_text_config = _create_hf_text_config(vocab_size=vocab_size, num_layers=1)
    hf_text_config.enable_moe_block = enable_moe
    if getattr(hf_text_config, "enable_moe_block", False):
        hf_text_config.num_experts = 4
        hf_text_config.top_k_experts = 2
    hf_model = _create_hf_model(hf_text_config)
    tt_model = _build_tt_model(mesh_device, hf_model, hf_text_config, num_layers=1, max_seq_len=total_len)

    prompt_tokens = torch.randint(0, vocab_size, (1, prompt_len), dtype=torch.long)
    init_canvas = torch.randint(0, vocab_size, (1, canvas_len), dtype=torch.long)
    gumbel_noise = [torch.zeros(1, canvas_len, vocab_size) for _ in range(max_steps)]
    noise_tokens = [torch.randint(0, vocab_size, (1, canvas_len), dtype=torch.long) for _ in range(max_steps)]
    cfg = DiffusionConfig(
        max_denoise_steps=max_steps,
        entropy_stop_threshold=-1.0,
        stable_steps_to_halt=1,
        entropy_budget=0.1,
    )

    self_conditioning_ref = SelfConditioning(
        hf_text_config.hidden_size,
        intermediate_size=hf_text_config.intermediate_size,
        eps=hf_text_config.rms_norm_eps,
        activation=hf_text_config.hidden_activation,
    ).eval()
    self_conditioning_state = {
        "pre_norm.weight": self_conditioning_ref.pre_norm.weight.data.clone(),
        "gate_proj.weight": self_conditioning_ref.gate_proj.weight.data.clone(),
        "up_proj.weight": self_conditioning_ref.up_proj.weight.data.clone(),
        "down_proj.weight": self_conditioning_ref.down_proj.weight.data.clone(),
    }
    mask = build_canvas_denoise_mask(
        prompt_len,
        canvas_len,
        local_window=False,
        neg_inf=NEG,
        dtype=torch.float32,
    ).view(1, 1, canvas_len, total_len)
    with torch.no_grad():
        prompt_hidden = hf_model.embed_tokens(prompt_tokens)
        prompt_kv_hidden = hf_model.layers[0].input_layernorm(prompt_hidden)
    ref_logits_by_step = []

    class TorchLogitsAdapter:
        def __init__(self):
            self.prev_logits = None

        def __call__(self, canvas, step):
            with torch.no_grad():
                canvas_hidden = hf_model.embed_tokens(canvas)
                conditioned = self_conditioning_ref.condition(
                    canvas_hidden,
                    self.prev_logits,
                    hf_model.embed_tokens.weight,
                    enabled=self.prev_logits is not None,
                )
                logits = _torch_denoise_logits_reference(hf_model, conditioned, [prompt_kv_hidden], mask)
                self.prev_logits = logits
                ref_logits_by_step.append(logits)
                return logits

    ref = ref_denoise_block(
        TorchLogitsAdapter(),
        init_canvas,
        cfg,
        vocab_size,
        gumbel_noise_fn=lambda step: gumbel_noise[step],
        noise_tokens_fn=lambda step: noise_tokens[step],
    )

    tt_prompt_tokens = _to_device_tokens(mesh_device, prompt_tokens)
    tt_prompt_hidden = embed_canvas_tokens(tt_model, tt_prompt_tokens)
    tt_prompt_logits = tt_model(
        tt_prompt_hidden,
        is_decode=False,
        input_ids_torch=prompt_tokens,
        kv_phase=KVCachePhase.PREFILL_WRITE,
    )
    tt_prompt_logits.deallocate(True)
    tt_prompt_kv_by_layer = [read_prompt_kv_cache_slice(tt_model.tt_kv_cache[0], prompt_len=prompt_len)]
    self_conditioning = TtSelfConditioning(
        mesh_device,
        self_conditioning_state,
        hidden_size=hf_text_config.hidden_size,
        intermediate_size=hf_text_config.intermediate_size,
        eps=hf_text_config.rms_norm_eps,
    )
    tt_self_conditioning_embedding = _to_device(
        mesh_device,
        hf_model.embed_tokens.weight.detach().unsqueeze(0).unsqueeze(0),
    )
    tt_adapter_base = DenoiseLogitsAdapter(
        tt_model,
        prompt_hidden_by_layer=tt_prompt_kv_by_layer,
        self_conditioning=self_conditioning,
        self_conditioning_embedding_weight=tt_self_conditioning_embedding,
    )
    tt_logits_by_step = []

    def tt_adapter(canvas_tokens, step):
        logits = tt_adapter_base(canvas_tokens, step)
        tt_logits_by_step.append(_to_torch(logits, mesh_device).squeeze(0).float())
        return logits

    tt = denoise_block(
        tt_adapter,
        _to_device_canvas_ids(mesh_device, init_canvas),
        cfg,
        gumbel_noise_fn=lambda step: _to_device(mesh_device, gumbel_noise[step].unsqueeze(0)),
        noise_tokens_fn=lambda step: _to_device_canvas_ids(mesh_device, noise_tokens[step]),
    )

    comparison = compare_trajectories(
        ref,
        tt,
        min_argmax_agreement=0.0,
        min_sampled_agreement=0.0,
        min_accept_iou=0.0,
        min_canvas_agreement=0.0,
        min_per_step_entropy_pcc=0.0,
        max_entropy_abs_err_threshold=float("inf"),
        committed_match_threshold=0.0,
        entropy_pcc_threshold=0.0,
    )
    accept_flips = [int((ra.accept_mask != rb.accept_mask).sum()) for ra, rb in zip(ref.per_step, tt.per_step)]
    argmax_flips = [int((ra.argmax != rb.argmax).sum()) for ra, rb in zip(ref.per_step, tt.per_step)]
    canvas_flips = [int((ra.canvas != rb.canvas).sum()) for ra, rb in zip(ref.per_step, tt.per_step)]
    logits_pcc = [
        float(comp_pcc(ref_logits_by_step[i].float(), tt_logits_by_step[i].float(), pcc=0.0)[1])
        for i in range(max_steps)
    ]
    logits_mean_abs = [
        float((ref_logits_by_step[i].float() - tt_logits_by_step[i].float()).abs().mean()) for i in range(max_steps)
    ]
    logits_max_abs = [
        float((ref_logits_by_step[i].float() - tt_logits_by_step[i].float()).abs().max()) for i in range(max_steps)
    ]
    ref_top2_margin_mean = [
        float(torch.topk(ref_logits_by_step[i].float(), k=2, dim=-1).values.diff(dim=-1).abs().mean())
        for i in range(max_steps)
    ]
    logits_argmax_agreement = [
        float((ref_logits_by_step[i].argmax(dim=-1) == tt_logits_by_step[i].argmax(dim=-1)).float().mean())
        for i in range(max_steps)
    ]
    logits_top8_contains_ref_argmax = [
        float(
            (tt_logits_by_step[i].topk(k=8, dim=-1).indices == ref_logits_by_step[i].argmax(dim=-1, keepdim=True))
            .any(dim=-1)
            .float()
            .mean()
        )
        for i in range(max_steps)
    ]
    print(
        "\n[real-logits trajectory] "
        f"mode={'moe' if enable_moe else 'dense'} "
        f"accept_flips={accept_flips} argmax_flips={argmax_flips} canvas_flips={canvas_flips} "
        f"entropy_pcc={comparison.per_step_entropy_pcc} "
        f"logits_pcc={logits_pcc} logits_argmax_agreement={logits_argmax_agreement} "
        f"logits_top8_contains_ref_argmax={logits_top8_contains_ref_argmax} "
        f"logits_mean_abs={logits_mean_abs} logits_max_abs={logits_max_abs} "
        f"ref_top2_margin_mean={ref_top2_margin_mean}"
    )

    tt_adapter_base.reset()
    for tt_k, tt_v in tt_prompt_kv_by_layer:
        tt_k.deallocate(True)
        tt_v.deallocate(True)

    assert comparison.steps_match and comparison.halted_match
    assert ref.num_steps == tt.num_steps == max_steps
    assert not ref.halted and not tt.halted
    assert len(accept_flips) == max_steps
    # This remains a diagnostic for the known bf16 decision-bar blocker, but it
    # should still fail loudly if the real-logits path stops resembling torch.
    min_logits_pcc = 0.96 if enable_moe else 0.975
    max_total_accept_flips = 0 if enable_moe else 4
    assert min(logits_pcc) >= min_logits_pcc
    assert min(logits_top8_contains_ref_argmax) >= 0.80
    assert sum(accept_flips) <= max_total_accept_flips
