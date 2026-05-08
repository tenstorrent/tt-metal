# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
``Mistral4ForCausalLM`` **prefill logits** (no loss, no KV cache): hybrid backbone + device LM head.

The overload without a causal mask matches the legacy **stacked** reference (decoder layers with
``attention_mask=None``; see :mod:`model_backbone`).

When you pass ``attention_mask_4d`` (or use :func:`causal_lm_prefill_logits_hybrid_with_hf_prefill_causal_mask_bf16`),
hybrid attention matches HF ``eager_attention_forward`` masking, typically built from
:func:`mistral4_prefill_causal_attention_mask`.

Pass ``use_device_sdpa_attention=True`` to use on-device SDPA in the backbone (see :mod:`model_backbone`).
"""

from __future__ import annotations

import torch

from models.tt_transformers.tt.mistral_small_4.lm_head import lm_head_logits_bf16, lm_head_logits_reference_torch
from models.tt_transformers.tt.mistral_small_4.model_backbone import (
    language_model_backbone_forward_reference_torch,
    language_model_backbone_hybrid_forward_bf16,
    language_model_backbone_hybrid_with_hf_prefill_causal_mask_bf16,
)


def causal_lm_prefill_logits_reference_torch(
    input_ids: torch.LongTensor,
    position_ids: torch.LongTensor | None,
    model: torch.nn.Module,
    attention_mask_4d: torch.Tensor | None = None,
) -> torch.Tensor:
    """CPU: stacked backbone reference → ``lm_head`` logits ``[B, S, V]``."""
    from transformers.models.mistral4.modeling_mistral4 import Mistral4ForCausalLM

    if not isinstance(model, Mistral4ForCausalLM):
        raise TypeError("causal_lm_prefill_logits_reference_torch expects Mistral4ForCausalLM")

    with torch.no_grad():
        hidden = language_model_backbone_forward_reference_torch(
            input_ids, position_ids, model.model, attention_mask_4d=attention_mask_4d
        )
        return lm_head_logits_reference_torch(hidden, model.lm_head.weight.data)


def causal_lm_prefill_logits_hybrid_bf16(
    mesh_device,
    input_ids: torch.LongTensor,
    position_ids: torch.LongTensor | None,
    model: torch.nn.Module,
    attention_mask_4d: torch.Tensor | None = None,
    *,
    use_device_sdpa_attention: bool = False,
) -> torch.Tensor:
    """Device hybrid backbone + device LM head; returns host bf16 logits ``[B, S, vocab_size]``."""
    from transformers.models.mistral4.modeling_mistral4 import Mistral4ForCausalLM

    if not isinstance(model, Mistral4ForCausalLM):
        raise TypeError("causal_lm_prefill_logits_hybrid_bf16 expects Mistral4ForCausalLM")

    hidden = language_model_backbone_hybrid_forward_bf16(
        mesh_device,
        input_ids,
        position_ids,
        model.model,
        attention_mask_4d=attention_mask_4d,
        use_device_sdpa_attention=use_device_sdpa_attention,
    )
    return lm_head_logits_bf16(mesh_device, hidden, model.lm_head.weight.data)


def causal_lm_prefill_logits_hf_reference_torch(
    input_ids: torch.LongTensor,
    position_ids: torch.LongTensor | None,
    model: torch.nn.Module,
    *,
    attention_mask_2d: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact HF ``Mistral4ForCausalLM.forward`` logits (``logits_to_keep`` default: full sequence)."""
    from transformers.models.mistral4.modeling_mistral4 import Mistral4ForCausalLM

    if not isinstance(model, Mistral4ForCausalLM):
        raise TypeError("causal_lm_prefill_logits_hf_reference_torch expects Mistral4ForCausalLM")

    with torch.no_grad():
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask_2d,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        ).logits


def causal_lm_prefill_logits_hybrid_with_hf_prefill_causal_mask_bf16(
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
    """Hybrid logits using HF-style prefill causal mask (see :func:`language_model_backbone_hybrid_with_hf_prefill_causal_mask_bf16`)."""
    from transformers.models.mistral4.modeling_mistral4 import Mistral4ForCausalLM

    if not isinstance(model, Mistral4ForCausalLM):
        raise TypeError("causal_lm_prefill_logits_hybrid_with_hf_prefill_causal_mask_bf16 expects Mistral4ForCausalLM")

    hidden = language_model_backbone_hybrid_with_hf_prefill_causal_mask_bf16(
        mesh_device,
        input_ids,
        position_ids,
        model.model,
        attention_mask_2d=attention_mask_2d,
        or_mask_function=or_mask_function,
        and_mask_function=and_mask_function,
        block_sequence_ids=block_sequence_ids,
        use_device_sdpa_attention=use_device_sdpa_attention,
        prefused_hidden_states_bsh=prefused_hidden_states_bsh,
    )
    return lm_head_logits_bf16(mesh_device, hidden, model.lm_head.weight.data)
