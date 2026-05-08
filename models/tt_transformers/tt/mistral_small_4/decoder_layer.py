# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
``Mistral4DecoderLayer`` forward helpers:

- **Dense** MLP (``layer_idx < first_k_dense_replace``): norms + hybrid attention + ``dense_mlp_bf16``.
- **MoE** (``layer_idx >= first_k_dense_replace``): same attention stack; router logits on device;
  routing on host (torch); **routed** experts via :func:`~models.tt_transformers.tt.mistral_small_4.moe_naive.mistral4_naive_moe_routed_bf16`
  (device SwiGLU per expert); **shared** experts via ``dense_mlp_bf16``.

Attention can use :func:`~models.tt_transformers.tt.mistral_small_4.attention_device.attention_forward_device_sdpa_bf16`
when ``use_device_sdpa_attention=True`` (optional ``attention_mask`` is forwarded when set).

Optional ``past_key_values`` (HF ``DynamicCache`` / ``Cache``) is forwarded into attention so decode steps can
reuse K/V (see :func:`~models.tt_transformers.tt.mistral_small_4.model_backbone.language_model_backbone_hybrid_forward_incremental_bf16`).
"""

from __future__ import annotations

import torch

from models.tt_transformers.tt.mistral_small_4.attention_device import attention_forward_device_sdpa_bf16
from models.tt_transformers.tt.mistral_small_4.attention_full import attention_forward_hybrid_bf16
from models.tt_transformers.tt.mistral_small_4.dense_mlp import dense_mlp_bf16
from models.tt_transformers.tt.mistral_small_4.moe_naive import mistral4_naive_moe_routed_bf16
from models.tt_transformers.tt.mistral_small_4.rms_norm import rms_norm_bf16
from models.tt_transformers.tt.mistral_small_4.router import router_logits_bf16
from models.tt_transformers.tt.mistral_small_4.routing import route_tokens_to_experts_reference_torch


def decoder_layer_dense_hybrid_forward_bf16(
    mesh_device,
    hidden_states_bsh: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    position_ids: torch.LongTensor,
    layer: torch.nn.Module,
    attention_mask: torch.Tensor | None = None,
    *,
    use_device_sdpa_attention: bool = False,
    past_key_values=None,
    layer_idx: int | None = None,
) -> torch.Tensor:
    """
    One decoder layer (residual + pre-norm attention + residual + pre-norm dense MLP).

    Requires ``layer.mlp`` to be ``Mistral4MLP`` (set ``first_k_dense_replace > layer_idx`` in config).

    ``hidden_states_bsh``: ``[B, S, H]`` bf16 on host.

    ``use_device_sdpa_attention``: if ``True``, use on-device SDPA for attention; ``attention_mask`` is passed through
    (``None`` or a 4D HF-style additive mask).

    ``past_key_values`` / ``layer_idx``: optional KV cache (defaults ``layer_idx`` to ``layer.self_attn.layer_idx``).
    """
    from transformers.models.mistral4.modeling_mistral4 import Mistral4MLP

    if not isinstance(layer.mlp, Mistral4MLP):
        raise TypeError("decoder_layer_dense_hybrid_forward_bf16 requires Mistral4DecoderLayer with Mistral4MLP")

    residual = hidden_states_bsh
    h = rms_norm_bf16(
        mesh_device,
        hidden_states_bsh,
        layer.input_layernorm.weight.data,
        epsilon=float(layer.input_layernorm.variance_epsilon),
    )
    li = int(layer.self_attn.layer_idx) if layer_idx is None else int(layer_idx)
    if use_device_sdpa_attention:
        h = attention_forward_device_sdpa_bf16(
            mesh_device,
            h,
            position_embeddings,
            position_ids,
            layer.self_attn,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            layer_idx=li,
        )
    else:
        h = attention_forward_hybrid_bf16(
            mesh_device,
            h,
            position_embeddings,
            position_ids,
            layer.self_attn,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            layer_idx=li,
        )
    h = residual + h

    residual = h
    h = rms_norm_bf16(
        mesh_device,
        h,
        layer.post_attention_layernorm.weight.data,
        epsilon=float(layer.post_attention_layernorm.variance_epsilon),
    )
    h = dense_mlp_bf16(
        mesh_device,
        h,
        layer.mlp.gate_proj.weight,
        layer.mlp.up_proj.weight,
        layer.mlp.down_proj.weight,
    )
    return residual + h


def decoder_layer_moe_hybrid_forward_bf16(
    mesh_device,
    hidden_states_bsh: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    position_ids: torch.LongTensor,
    layer: torch.nn.Module,
    attention_mask: torch.Tensor | None = None,
    *,
    use_device_sdpa_attention: bool = False,
    past_key_values=None,
    layer_idx: int | None = None,
) -> torch.Tensor:
    """
    Decoder layer when ``layer.mlp`` is ``Mistral4MoE`` (``layer_idx >= first_k_dense_replace``).

    Router logits from ``router_logits_bf16``; ``route_tokens_to_experts`` matches HF via
    :func:`~models.tt_transformers.tt.mistral_small_4.routing.route_tokens_to_experts_reference_torch`.
    Routed experts use :func:`~models.tt_transformers.tt.mistral_small_4.moe_naive.mistral4_naive_moe_routed_bf16``
    (device matmuls; token/expert indexing on CPU like HF). **Shared** experts use ``dense_mlp_bf16`` on device.

    Note: a standalone ``Mistral4DecoderLayer`` does not run ``PreTrainedModel._init_weights``; the MoE
    router and naive expert tensors start as ``torch.empty`` unless you initialize them (parity tests do).

    ``use_device_sdpa_attention``: same as :func:`decoder_layer_dense_hybrid_forward_bf16`.
    """
    from transformers.models.mistral4.modeling_mistral4 import Mistral4MoE

    if not isinstance(layer.mlp, Mistral4MoE):
        raise TypeError("decoder_layer_moe_hybrid_forward_bf16 requires Mistral4DecoderLayer with Mistral4MoE")

    moe = layer.mlp

    residual = hidden_states_bsh
    h = rms_norm_bf16(
        mesh_device,
        hidden_states_bsh,
        layer.input_layernorm.weight.data,
        epsilon=float(layer.input_layernorm.variance_epsilon),
    )
    li = int(layer.self_attn.layer_idx) if layer_idx is None else int(layer_idx)
    if use_device_sdpa_attention:
        h = attention_forward_device_sdpa_bf16(
            mesh_device,
            h,
            position_embeddings,
            position_ids,
            layer.self_attn,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            layer_idx=li,
        )
    else:
        h = attention_forward_hybrid_bf16(
            mesh_device,
            h,
            position_embeddings,
            position_ids,
            layer.self_attn,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            layer_idx=li,
        )
    h = residual + h

    residual = h
    h = rms_norm_bf16(
        mesh_device,
        h,
        layer.post_attention_layernorm.weight.data,
        epsilon=float(layer.post_attention_layernorm.variance_epsilon),
    )

    b, s, hidden = h.shape
    logits = router_logits_bf16(mesh_device, h, moe.gate.weight)
    topk_indices, topk_weights = route_tokens_to_experts_reference_torch(
        logits,
        n_group=int(moe.n_group),
        n_routed_experts=int(moe.n_routed_experts),
        topk_group=int(moe.topk_group),
        top_k=int(moe.top_k),
        norm_topk_prob=bool(moe.norm_topk_prob),
        routed_scaling_factor=float(moe.routed_scaling_factor),
    )

    h_flat = h.reshape(-1, hidden)
    routed = mistral4_naive_moe_routed_bf16(mesh_device, h_flat, topk_indices, topk_weights, moe.experts).view(
        b, s, hidden
    )

    se = moe.shared_experts
    shared = dense_mlp_bf16(
        mesh_device,
        h,
        se.gate_proj.weight,
        se.up_proj.weight,
        se.down_proj.weight,
    )
    mlp_out = routed + shared
    return residual + mlp_out


def decoder_layer_dense_forward_reference_torch(
    hidden_states_bsh: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    position_ids: torch.LongTensor,
    layer: torch.nn.Module,
    attention_mask: torch.Tensor | None = None,
    *,
    past_key_values=None,
    use_cache: bool = False,
) -> torch.Tensor:
    """HF ``Mistral4DecoderLayer.forward`` (CPU reference)."""
    with torch.no_grad():
        return layer(
            hidden_states_bsh,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )


# Same HF ``Mistral4DecoderLayer.forward``; kept as a separate name for MoE-layer tests / docs.
decoder_layer_moe_forward_reference_torch = decoder_layer_dense_forward_reference_torch
