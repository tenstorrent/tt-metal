# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
``Mistral4ForCausalLM`` **incremental** forward (KV cache): hybrid backbone steps + LM head.

Use :func:`causal_lm_incremental_step_logits_hybrid_bf16` for each prefill or decode chunk, then
:func:`causal_lm_greedy_generate_hybrid_bf16` for a minimal greedy sampling loop (parity / bring-up).

Constraints match :func:`~models.tt_transformers.tt.mistral_small_4.model_backbone.language_model_backbone_hybrid_forward_incremental_bf16`
(``batch_size == 1`` for device SDPA paths). Optional ``attention_mask_2d`` is forwarded to HF ``create_causal_mask``.
"""

from __future__ import annotations

import torch

from models.tt_transformers.tt.mistral_small_4.lm_head import lm_head_logits_bf16
from models.tt_transformers.tt.mistral_small_4.model_backbone import (
    language_model_backbone_hybrid_forward_incremental_bf16,
)


def causal_lm_incremental_step_logits_hybrid_bf16(
    mesh_device,
    input_ids: torch.LongTensor,
    model: torch.nn.Module,
    *,
    past_key_values=None,
    position_ids: torch.LongTensor | None = None,
    attention_mask_2d: torch.Tensor | None = None,
    use_device_sdpa_attention: bool = False,
    prefused_hidden_states_bsh: torch.Tensor | None = None,
) -> tuple[torch.Tensor, object]:
    """
    One ``Mistral4ForCausalLM`` forward slice: backbone (with KV cache) + ``lm_head``.

    Args:
        input_ids: ``[B, S]`` — full prompt on first call, or ``[B, 1]`` decode token(s).
        model: HF ``Mistral4ForCausalLM`` (uses ``model.model`` for the backbone).
        past_key_values: HF ``Cache``; ``None`` starts a new ``DynamicCache`` inside the backbone when
            ``use_cache=True`` implicit via backbone (always caching here).

    Returns:
        ``(logits, past_key_values)`` with logits host bf16 ``[B, S, vocab_size]``.
    """
    from transformers.models.mistral4.modeling_mistral4 import Mistral4ForCausalLM

    if not isinstance(model, Mistral4ForCausalLM):
        raise TypeError("causal_lm_incremental_step_logits_hybrid_bf16 expects Mistral4ForCausalLM")

    hidden, past_key_values = language_model_backbone_hybrid_forward_incremental_bf16(
        mesh_device,
        input_ids,
        position_ids,
        model.model,
        past_key_values=past_key_values,
        attention_mask_2d=attention_mask_2d,
        use_cache=True,
        use_device_sdpa_attention=use_device_sdpa_attention,
        prefused_hidden_states_bsh=prefused_hidden_states_bsh,
    )
    logits = lm_head_logits_bf16(mesh_device, hidden, model.lm_head.weight.data)
    return logits, past_key_values


def causal_lm_greedy_generate_hybrid_bf16(
    mesh_device,
    prompt_ids: torch.LongTensor,
    model: torch.nn.Module,
    max_new_tokens: int,
    *,
    attention_mask_2d: torch.Tensor | None = None,
    use_device_sdpa_attention: bool = False,
) -> torch.LongTensor:
    """
    Greedy decoding: prefill ``prompt_ids``, then ``max_new_tokens`` argmax steps.

    Uses :func:`causal_lm_incremental_step_logits_hybrid_bf16` throughout. ``prompt_ids`` is ``[B, P]``;
    returns ``[B, P + max_new_tokens]``. For ``max_new_tokens == 0``, returns ``prompt_ids.clone()``.

    Intended for bring-up parity vs HF (see tests): ``batch_size`` should be ``1`` when device SDPA is on.
    """
    from transformers.models.mistral4.modeling_mistral4 import Mistral4ForCausalLM

    if not isinstance(model, Mistral4ForCausalLM):
        raise TypeError("causal_lm_greedy_generate_hybrid_bf16 expects Mistral4ForCausalLM")

    if max_new_tokens <= 0:
        return prompt_ids.clone()

    past = None
    logits, past = causal_lm_incremental_step_logits_hybrid_bf16(
        mesh_device,
        prompt_ids,
        model,
        past_key_values=past,
        attention_mask_2d=attention_mask_2d,
        use_device_sdpa_attention=use_device_sdpa_attention,
    )
    next_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = torch.cat([prompt_ids, next_ids], dim=-1)

    for _ in range(max_new_tokens - 1):
        logits, past = causal_lm_incremental_step_logits_hybrid_bf16(
            mesh_device,
            next_ids,
            model,
            past_key_values=past,
            attention_mask_2d=attention_mask_2d,
            use_device_sdpa_attention=use_device_sdpa_attention,
        )
        next_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_ids], dim=-1)

    return generated


def causal_lm_greedy_generate_hf_reference_torch(
    prompt_ids: torch.LongTensor,
    model: torch.nn.Module,
    max_new_tokens: int,
    *,
    attention_mask_2d: torch.Tensor | None = None,
) -> torch.LongTensor:
    """
    HF greedy loop using ``Mistral4ForCausalLM`` with ``DynamicCache``, mirroring :func:`causal_lm_greedy_generate_hybrid_bf16`.
    """
    from transformers.cache_utils import DynamicCache
    from transformers.models.mistral4.modeling_mistral4 import Mistral4ForCausalLM

    if not isinstance(model, Mistral4ForCausalLM):
        raise TypeError("causal_lm_greedy_generate_hf_reference_torch expects Mistral4ForCausalLM")

    if max_new_tokens <= 0:
        return prompt_ids.clone()

    past = DynamicCache(config=model.config)
    with torch.no_grad():
        kwargs = dict(input_ids=prompt_ids, past_key_values=past, use_cache=True)
        if attention_mask_2d is not None:
            kwargs["attention_mask"] = attention_mask_2d
        out = model(**kwargs)
        next_ids = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([prompt_ids, next_ids], dim=-1)

        for _ in range(max_new_tokens - 1):
            kwargs = dict(input_ids=next_ids, past_key_values=past, use_cache=True)
            if attention_mask_2d is not None:
                kwargs["attention_mask"] = attention_mask_2d
            out = model(**kwargs)
            next_ids = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_ids], dim=-1)

    return generated
