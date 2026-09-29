# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Helpers the vendored glm5_next modeling code imports that transformers 5.12.1 lacks.

The kernel-hub and accelerate decorators are no-ops (the eager torch paths run). The recurrent mask keeps the 2D
padding mask for the current tokens only (the indexer reads a local [B, S] mask). The vision helpers are never
called: the vision tower is out of scope.
"""

from __future__ import annotations


def use_kernel_func_from_hub_with_fallback(*_args, **_kwargs):
    return lambda fn: fn


def force_accelerate_hooks(*_args, **_kwargs):
    return lambda fn: fn


def create_recurrent_attention_mask(config=None, inputs_embeds=None, attention_mask=None, past_key_values=None, **_):
    if attention_mask is None:
        return None
    assert attention_mask.dim() == 2, "only 2D padding masks"
    return attention_mask[:, -inputs_embeds.shape[1] :]


def _vision_only(*_args, **_kwargs):
    raise NotImplementedError("the GLM-5.3 vision tower is out of scope for this bring-up")


get_max_seqlen = get_vision_attention_seqlens = get_vision_position_ids = _vision_only
