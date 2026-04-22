# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Remote Hub ``DotsOCRForCausalLM.prepare_inputs_for_generation`` indexes
``cache_position[0]`` unconditionally. Recent ``transformers`` versions can pass
``cache_position=None`` on the prefill path, which raises ``TypeError`` and breaks
``model.generate()`` for multimodal inputs.

We replace the method on the loaded module instance with logic equivalent to the Hub
implementation, plus fixes for Transformers 5.x ``generate()``:

* ``cache_position`` may be ``None`` on the prefill path (Hub code did ``cache_position[0]``).
* ``model_kwargs`` retain ``pixel_values`` / ``image_grid_thw`` on decode steps; Dots must not
  forward them once ``input_ids`` are decode-only (no ``image_token_id`` rows).
"""

from __future__ import annotations

import types
from typing import Any

import torch.nn as nn


def _is_prefill_from_cache(past_key_values: Any) -> bool:
    """First generation step has no KV cache; decode steps always have a populated cache."""
    if past_key_values is None:
        return True
    try:
        if hasattr(past_key_values, "get_seq_length"):
            return int(past_key_values.get_seq_length()) == 0
    except Exception:
        pass
    return False


def patch_dots_ocr_prepare_inputs_for_generation(model: nn.Module) -> None:
    if model.__class__.__name__ != "DotsOCRForCausalLM":
        return

    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    def patched(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        model_inputs = Qwen2ForCausalLM.prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )
        # ``generate`` keeps the original ``pixel_values`` / ``image_grid_thw`` in ``model_kwargs`` for every
        # step, but ``GenerationMixin`` forwards them on decode steps too. Dots' ``prepare_inputs_embeds``
        # then sees ``pixel_values`` with ``input_ids`` that no longer contain ``image_token_id`` rows.
        # Use KV cache presence — ``cache_position`` is often ``None`` on every step in Transformers 5.x.
        prefill = _is_prefill_from_cache(past_key_values)
        if prefill and pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        if not prefill:
            model_inputs.pop("pixel_values", None)
            model_inputs.pop("image_grid_thw", None)
        return model_inputs

    model.prepare_inputs_for_generation = types.MethodType(patched, model)
