# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Mini **language-model** backbone for Hugging Face ``Mistral4Model`` (text stack only).

Maps to ``Mistral3ForConditionalGeneration.model.language_model`` in the multimodal checkpoint tree:
``embed_tokens`` → ``layers`` → ``norm``. No vision tower, projector, or ``lm_head``.

The hybrid path matches the rest of this package: device embedding, norms / attention / dense or MoE
blocks as implemented in :mod:`decoder_layer`, then final :func:`rms_norm_bf16`.

.. note::

    Without ``attention_mask_4d``, decoder parity matches the legacy convention (``attention_mask=None``).

    With ``attention_mask_4d`` (e.g. from :func:`~models.tt_transformers.tt.mistral_small_4.causal_mask.mistral4_prefill_causal_attention_mask`),
    hybrid attention matches HF ``eager_attention_forward`` masking. Use
    :func:`language_model_backbone_hybrid_with_hf_prefill_causal_mask_bf16` to build the same mask as
    ``Mistral4Model`` prefill (HF ``embed_tokens`` for mask metadata), then compare against
    :func:`language_model_backbone_last_hidden_hf_reference_torch` for an end-to-end HF reference.

    Pass ``use_device_sdpa_attention=True`` on those entry points to run **ttnn** SDPA with the same masks.

    For **KV-cache** prefill / decode steps (HF ``DynamicCache`` + ``create_causal_mask``), use
    :func:`language_model_backbone_hybrid_forward_incremental_bf16`.
"""

from __future__ import annotations

import torch

from models.tt_transformers.tt.mistral_small_4.causal_mask import (
    mistral4_causal_attention_mask,
    mistral4_prefill_causal_attention_mask,
)
from models.tt_transformers.tt.mistral_small_4.decoder_layer import (
    decoder_layer_dense_hybrid_forward_bf16,
    decoder_layer_moe_hybrid_forward_bf16,
)
from models.tt_transformers.tt.mistral_small_4.embedding import embedding_lookup_bf16
from models.tt_transformers.tt.mistral_small_4.rms_norm import rms_norm_bf16


def _default_prefill_position_ids(batch_size: int, seq_len: int, *, device: torch.device) -> torch.LongTensor:
    """``[B, S]`` positions ``0..S-1`` per row (HF default when no left padding)."""
    return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).contiguous()


def language_model_backbone_forward_reference_torch(
    input_ids: torch.LongTensor,
    position_ids: torch.LongTensor | None,
    model: torch.nn.Module,
    attention_mask_4d: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    CPU reference: embed → each ``Mistral4DecoderLayer`` → ``norm``.

    If ``attention_mask_4d`` is ``None``, layers use a fully-visible attention mask (legacy parity).
    Otherwise it is forwarded to every decoder layer (e.g. HF causal mask from ``create_causal_mask``).
    """
    from transformers.models.mistral4.modeling_mistral4 import Mistral4Model

    if not isinstance(model, Mistral4Model):
        raise TypeError("language_model_backbone_forward_reference_torch expects Mistral4Model")

    with torch.no_grad():
        hidden = model.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = _default_prefill_position_ids(
                int(hidden.shape[0]), int(hidden.shape[1]), device=hidden.device
            )
        cos, sin = model.rotary_emb(hidden, position_ids=position_ids)
        position_embeddings = (cos, sin)
        for layer in model.layers:
            hidden = layer(
                hidden,
                attention_mask=attention_mask_4d,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
        return model.norm(hidden)


def language_model_backbone_hybrid_forward_bf16(
    mesh_device,
    input_ids: torch.LongTensor,
    position_ids: torch.LongTensor | None,
    model: torch.nn.Module,
    attention_mask_4d: torch.Tensor | None = None,
    *,
    use_device_sdpa_attention: bool = False,
    prefused_hidden_states_bsh: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    ``Mistral4Model``-shaped prefill without KV cache: embedding on device, then per-layer hybrid
    decoder (dense or MoE), then final RMSNorm on device.

    ``attention_mask_4d``: optional mask passed into hybrid attention (same object every layer), e.g.
    from :func:`mistral4_prefill_causal_attention_mask` using HF ``embed_tokens`` outputs.

    ``use_device_sdpa_attention``: forwarded to decoder layers for on-device SDPA attention.

    ``prefused_hidden_states_bsh``: optional ``[B, S, H]`` bf16 host tensor (e.g. HF ``embed_tokens`` with
    vision features fused). When set, skips on-device token embedding lookup and runs decoder stack from
    this hidden state (same dtype/device contract as embedding lookup output).
    """
    from transformers.models.mistral4.modeling_mistral4 import Mistral4MLP, Mistral4Model, Mistral4MoE

    if not isinstance(model, Mistral4Model):
        raise TypeError("language_model_backbone_hybrid_forward_bf16 expects Mistral4Model")

    if prefused_hidden_states_bsh is not None:
        hidden = prefused_hidden_states_bsh.to(torch.bfloat16).contiguous()
    else:
        hidden = embedding_lookup_bf16(mesh_device, input_ids, model.embed_tokens.weight.data)
    if position_ids is None:
        position_ids = _default_prefill_position_ids(int(hidden.shape[0]), int(hidden.shape[1]), device=hidden.device)
    cos, sin = model.rotary_emb(hidden, position_ids=position_ids)
    position_embeddings = (cos, sin)

    for layer in model.layers:
        if isinstance(layer.mlp, Mistral4MLP):
            hidden = decoder_layer_dense_hybrid_forward_bf16(
                mesh_device,
                hidden,
                position_embeddings,
                position_ids,
                layer,
                attention_mask=attention_mask_4d,
                use_device_sdpa_attention=use_device_sdpa_attention,
            )
        elif isinstance(layer.mlp, Mistral4MoE):
            hidden = decoder_layer_moe_hybrid_forward_bf16(
                mesh_device,
                hidden,
                position_embeddings,
                position_ids,
                layer,
                attention_mask=attention_mask_4d,
                use_device_sdpa_attention=use_device_sdpa_attention,
            )
        else:
            raise TypeError(f"Unsupported MLP type: {type(layer.mlp)}")

    return rms_norm_bf16(
        mesh_device,
        hidden,
        model.norm.weight.data,
        epsilon=float(model.norm.variance_epsilon),
    )


def language_model_backbone_last_hidden_hf_reference_torch(
    input_ids: torch.LongTensor,
    position_ids: torch.LongTensor | None,
    model: torch.nn.Module,
    *,
    attention_mask_2d: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact HF ``Mistral4Model.forward`` last hidden state (causal mask + ``create_causal_mask`` inside)."""
    from transformers.models.mistral4.modeling_mistral4 import Mistral4Model

    if not isinstance(model, Mistral4Model):
        raise TypeError("language_model_backbone_last_hidden_hf_reference_torch expects Mistral4Model")

    with torch.no_grad():
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask_2d,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        ).last_hidden_state


def language_model_backbone_hybrid_with_hf_prefill_causal_mask_bf16(
    mesh_device,
    input_ids: torch.LongTensor,
    position_ids: torch.LongTensor | None,
    model: torch.nn.Module,
    *,
    attention_mask_2d: torch.Tensor | None = None,
    or_mask_function=None,
    and_mask_function=None,
    block_sequence_ids: torch.Tensor | None = None,
    use_device_sdpa_attention: bool = False,
    prefused_hidden_states_bsh: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Hybrid backbone where the attention mask is built like ``Mistral4Model`` prefill: HF
    ``embed_tokens`` → :func:`mistral4_prefill_causal_attention_mask`, then the usual device embedding
    and hybrid layers use that mask.

    Optional ``or_mask_function`` / ``and_mask_function`` / ``block_sequence_ids`` are forwarded to
    ``create_causal_mask`` for multimodal (e.g. vision) extensions.

    ``use_device_sdpa_attention``: if ``True``, each layer uses device SDPA with the built causal ``attn_mask``.
    """
    from transformers.models.mistral4.modeling_mistral4 import Mistral4Model

    if not isinstance(model, Mistral4Model):
        raise TypeError("language_model_backbone_hybrid_with_hf_prefill_causal_mask_bf16 expects Mistral4Model")

    inputs_embeds = (
        prefused_hidden_states_bsh.to(torch.bfloat16).contiguous()
        if prefused_hidden_states_bsh is not None
        else model.embed_tokens(input_ids)
    )
    causal = mistral4_prefill_causal_attention_mask(
        model.config,
        inputs_embeds,
        position_ids,
        attention_mask_2d=attention_mask_2d,
        or_mask_function=or_mask_function,
        and_mask_function=and_mask_function,
        block_sequence_ids=block_sequence_ids,
    )
    return language_model_backbone_hybrid_forward_bf16(
        mesh_device,
        input_ids,
        position_ids,
        model,
        attention_mask_4d=causal,
        use_device_sdpa_attention=use_device_sdpa_attention,
        prefused_hidden_states_bsh=prefused_hidden_states_bsh,
    )


def language_model_backbone_hybrid_forward_incremental_bf16(
    mesh_device,
    input_ids: torch.LongTensor,
    position_ids: torch.LongTensor | None,
    model: torch.nn.Module,
    *,
    past_key_values=None,
    attention_mask_2d: torch.Tensor | None = None,
    use_cache: bool = True,
    use_device_sdpa_attention: bool = False,
    prefused_hidden_states_bsh: torch.Tensor | None = None,
) -> tuple[torch.Tensor, object | None]:
    """
    ``Mistral4Model`` forward split across calls: device embedding, per-layer hybrid decoder with optional
    **HF KV cache** (``transformers.cache_utils.DynamicCache``), final RMSNorm on device.

    When ``use_cache`` is ``False``, behaves like :func:`language_model_backbone_hybrid_forward_bf16` with
    no cache and returns ``(hidden, None)``.

    When ``use_cache`` is ``True``, creates a ``DynamicCache`` on first call if ``past_key_values`` is
    ``None``, mutates it in-place like ``Mistral4Model.forward``, and returns ``(hidden, past_key_values)``.
    Build the additive mask with ``transformers.masking_utils.create_causal_mask`` via
    :func:`~models.tt_transformers.tt.mistral_small_4.causal_mask.mistral4_causal_attention_mask`
    (same inputs as HF: ``embed_tokens`` outputs for mask metadata, ``past_key_values``, ``position_ids``).

    ``position_ids``: if ``None``, uses ``torch.arange(S) + past_seen_tokens`` per HF when a cache is present.
    """
    from transformers.cache_utils import DynamicCache
    from transformers.models.mistral4.modeling_mistral4 import Mistral4MLP, Mistral4Model, Mistral4MoE

    if not isinstance(model, Mistral4Model):
        raise TypeError("language_model_backbone_hybrid_forward_incremental_bf16 expects Mistral4Model")

    if not use_cache:
        out = language_model_backbone_hybrid_forward_bf16(
            mesh_device,
            input_ids,
            position_ids,
            model,
            attention_mask_4d=None,
            use_device_sdpa_attention=use_device_sdpa_attention,
            prefused_hidden_states_bsh=prefused_hidden_states_bsh,
        )
        return out, None

    if past_key_values is None:
        past_key_values = DynamicCache(config=model.config)

    cache_len_before = past_key_values.get_seq_length()
    inputs_embeds_for_mask = (
        prefused_hidden_states_bsh.to(torch.bfloat16).contiguous()
        if prefused_hidden_states_bsh is not None and cache_len_before == 0
        else model.embed_tokens(input_ids)
    )
    if position_ids is None:
        past_seen = past_key_values.get_seq_length()
        s = int(inputs_embeds_for_mask.shape[1])
        position_ids = (torch.arange(s, device=inputs_embeds_for_mask.device, dtype=torch.long) + past_seen).unsqueeze(
            0
        )

    causal = mistral4_causal_attention_mask(
        model.config,
        inputs_embeds_for_mask,
        position_ids,
        attention_mask_2d=attention_mask_2d,
        past_key_values=past_key_values,
    )

    if prefused_hidden_states_bsh is not None and cache_len_before == 0:
        hidden = prefused_hidden_states_bsh.to(torch.bfloat16).contiguous()
    else:
        hidden = embedding_lookup_bf16(mesh_device, input_ids, model.embed_tokens.weight.data)
    cos, sin = model.rotary_emb(hidden, position_ids=position_ids)
    position_embeddings = (cos, sin)

    for layer in model.layers:
        if isinstance(layer.mlp, Mistral4MLP):
            hidden = decoder_layer_dense_hybrid_forward_bf16(
                mesh_device,
                hidden,
                position_embeddings,
                position_ids,
                layer,
                attention_mask=causal,
                use_device_sdpa_attention=use_device_sdpa_attention,
                past_key_values=past_key_values,
            )
        elif isinstance(layer.mlp, Mistral4MoE):
            hidden = decoder_layer_moe_hybrid_forward_bf16(
                mesh_device,
                hidden,
                position_embeddings,
                position_ids,
                layer,
                attention_mask=causal,
                use_device_sdpa_attention=use_device_sdpa_attention,
                past_key_values=past_key_values,
            )
        else:
            raise TypeError(f"Unsupported MLP type: {type(layer.mlp)}")

    hidden = rms_norm_bf16(
        mesh_device,
        hidden,
        model.norm.weight.data,
        epsilon=float(model.norm.variance_epsilon),
    )
    return hidden, past_key_values
