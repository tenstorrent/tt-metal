# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN denoise-forward helpers for DiffusionGemma canvas attention.

This is the short-prompt W2 path: canvas queries attend to the frozen prompt
prefix plus the current canvas through an explicit all-attend mask. Sampling,
self-conditioning, and the multi-layer generation loop live in later W3/W4
helpers; this module owns the real masked attention wiring.
"""

from __future__ import annotations

import torch
import ttnn

from models.demos.gemma4.tt.attention.kv_phase import KVCachePhase
from models.experimental.diffusion_gemma.reference.attention_mask import build_canvas_denoise_mask

NEG = -1.0e9


def _replicate_mapper(mesh_device):
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    return ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None


def build_device_canvas_denoise_mask(
    mesh_device,
    *,
    prompt_len: int,
    canvas_len: int,
    dtype=ttnn.bfloat16,
):
    """Build the canonical all-attend `[1, 1, C, P+C]` denoise mask on device."""
    mask = build_canvas_denoise_mask(
        prompt_len,
        canvas_len,
        local_window=False,
        neg_inf=NEG,
        dtype=torch.float32,
    ).view(1, 1, canvas_len, prompt_len + canvas_len)
    return ttnn.from_torch(
        mask,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        mesh_mapper=_replicate_mapper(mesh_device),
    )


def denoise_attention_forward(
    tt_model,
    *,
    layer_idx: int,
    prompt_hidden,
    canvas_hidden,
    attn_mask=None,
):
    """Run one DiffusionGemma denoise attention layer on canvas hidden states.

    Args:
        tt_model: `Gemma4Model` carrying the reused DiffusionGemma decoder weights.
        layer_idx: decoder layer to run.
        prompt_hidden: frozen prompt hidden states `[1, 1, P, H]` on device.
        canvas_hidden: current canvas hidden states `[1, 1, C, H]` on device.
        attn_mask: optional prebuilt `[1, 1, C, P+C]` additive mask on device.

    Returns:
        The attention output for the canvas positions `[1, 1, C, H]`.
    """
    prompt_len = prompt_hidden.shape[-2]
    canvas_len = canvas_hidden.shape[-2]
    kv_hidden = ttnn.concat([prompt_hidden, canvas_hidden], dim=2)
    created_mask = attn_mask is None
    if created_mask:
        attn_mask = build_device_canvas_denoise_mask(
            tt_model.mesh_device,
            prompt_len=prompt_len,
            canvas_len=canvas_len,
        )

    out = tt_model.layers[layer_idx].self_attn(
        canvas_hidden,
        rope_mats=tt_model._get_rope_mats(layer_idx, seq_len=prompt_len + canvas_len),
        is_decode=False,
        kv_phase=KVCachePhase.DENOISE_READONLY,
        attn_mask=attn_mask,
        kv_hidden_states=kv_hidden,
        q_rope_offset=prompt_len,
    )
    kv_hidden.deallocate(True)
    if created_mask:
        attn_mask.deallocate(True)
    return out


def _denoise_layer_forward(tt_model, layer_idx, hidden_states, prompt_kv_hidden, attn_mask, prompt_len):
    layer = tt_model.layers[layer_idx]
    residual = hidden_states
    normed = layer.input_layernorm.forward(hidden_states)
    kv_hidden = ttnn.concat([prompt_kv_hidden, normed], dim=2)
    attn_output = layer.self_attn(
        normed,
        rope_mats=tt_model._get_rope_mats(layer_idx, seq_len=prompt_len + hidden_states.shape[-2]),
        is_decode=False,
        kv_phase=KVCachePhase.DENOISE_READONLY,
        attn_mask=attn_mask,
        kv_hidden_states=kv_hidden,
        q_rope_offset=prompt_len,
    )
    normed.deallocate(True)
    kv_hidden.deallocate(True)

    attn_output = layer.post_attention_layernorm.forward(attn_output)
    hidden_states = ttnn.add(residual, attn_output)
    residual.deallocate(True)
    attn_output.deallocate(True)

    residual = hidden_states
    normed = layer.pre_feedforward_layernorm.forward(hidden_states)
    mlp_output = layer.shared_mlp(normed)
    normed.deallocate(True)

    if layer.enable_moe_block:
        mlp_normed = layer.post_feedforward_layernorm_1.forward(mlp_output)
        mlp_output.deallocate(True)
        expert_input = layer.pre_feedforward_layernorm_2.forward(residual)
        expert_output = layer.moe(residual, expert_input)
        expert_input.deallocate(True)
        expert_normed = layer.post_feedforward_layernorm_2.forward(expert_output)
        expert_output.deallocate(True)
        hidden_states = ttnn.add(mlp_normed, expert_normed)
        mlp_normed.deallocate(True)
        expert_normed.deallocate(True)
    else:
        hidden_states = mlp_output

    hidden_states = layer.post_feedforward_layernorm.forward(hidden_states)
    combined = ttnn.add(residual, hidden_states)
    residual.deallocate(True)
    hidden_states.deallocate(True)
    if layer.layer_scalar != 1.0:
        scaled = ttnn.mul(combined, layer.layer_scalar)
        combined.deallocate(True)
        combined = scaled
    return combined


def denoise_logits_forward(
    tt_model,
    *,
    prompt_hidden_by_layer,
    canvas_hidden,
):
    """Run a short-prompt DiffusionGemma denoise logits forward.

    ``prompt_hidden_by_layer`` provides the frozen encoder-side attention inputs
    used as K/V source for each decoder layer. It is a list of `[1, 1, P, H]`
    device tensors with length matching `tt_model.layers`. The returned logits
    cover all canvas positions, which the diffusion sampler consumes each denoise
    step.
    """
    if len(prompt_hidden_by_layer) != len(tt_model.layers):
        raise ValueError(
            f"prompt_hidden_by_layer has {len(prompt_hidden_by_layer)} entries but model has {len(tt_model.layers)} layers"
        )

    hidden_states = canvas_hidden
    prompt_len = prompt_hidden_by_layer[0].shape[-2]
    canvas_len = canvas_hidden.shape[-2]
    attn_mask = build_device_canvas_denoise_mask(
        tt_model.mesh_device,
        prompt_len=prompt_len,
        canvas_len=canvas_len,
    )
    for layer_idx in range(len(tt_model.layers)):
        hidden_states = _denoise_layer_forward(
            tt_model,
            layer_idx,
            hidden_states,
            prompt_hidden_by_layer[layer_idx],
            attn_mask,
            prompt_len,
        )
    attn_mask.deallocate(True)
    hidden_states = tt_model.norm.forward(hidden_states)
    return tt_model._apply_lm_head(hidden_states, is_decode=False)


def embed_canvas_tokens(tt_model, canvas_tokens):
    """Embed device canvas token ids into `[1, 1, C, H]` TILE hidden states."""
    canvas_len = canvas_tokens.shape[-1]
    canvas_hidden = tt_model.embed_tokens(canvas_tokens)
    if len(canvas_hidden.shape) == 3:
        canvas_hidden = ttnn.reshape(canvas_hidden, (1, 1, canvas_len, tt_model.hidden_size))
    elif canvas_hidden.shape[-2] != canvas_len:
        canvas_hidden = ttnn.reshape(canvas_hidden, (1, 1, canvas_len, tt_model.hidden_size))
    return ttnn.to_layout(canvas_hidden, ttnn.TILE_LAYOUT)


def denoise_logits_from_tokens(
    tt_model,
    *,
    prompt_hidden_by_layer,
    canvas_tokens,
    self_conditioning=None,
    prev_logits=None,
    self_conditioning_embedding_weight=None,
    self_conditioning_compute_kernel_config=None,
):
    """Embed canvas token ids, optionally self-condition, then run denoise logits."""
    canvas_hidden = embed_canvas_tokens(tt_model, canvas_tokens)
    if self_conditioning is not None:
        conditioned = self_conditioning.condition(
            canvas_hidden,
            prev_logits,
            self_conditioning_embedding_weight,
            compute_kernel_config=self_conditioning_compute_kernel_config,
        )
        canvas_hidden.deallocate(True)
        canvas_hidden = conditioned
    return denoise_logits_forward(
        tt_model,
        prompt_hidden_by_layer=prompt_hidden_by_layer,
        canvas_hidden=canvas_hidden,
    )
