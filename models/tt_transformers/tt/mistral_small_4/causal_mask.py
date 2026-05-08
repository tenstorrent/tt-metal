# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Prefill causal masks aligned with Hugging Face ``transformers.masking_utils.create_causal_mask``."""

from __future__ import annotations

from collections.abc import Callable

import torch


def mistral4_prefill_causal_attention_mask(
    config,
    inputs_embeds_bsh: torch.Tensor,
    position_ids: torch.LongTensor | None,
    *,
    attention_mask_2d: torch.Tensor | None = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
    block_sequence_ids: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """
    Build the same causal attention mask ``Mistral4Model`` uses for prefill (``past_key_values=None``).

    Args:
        config: ``Mistral4Config`` (or any config understood by ``create_causal_mask``).
        inputs_embeds_bsh: ``[B, S, H]`` — only batch/seq/dtype/device are used for mask construction.
        position_ids: ``[B, S]`` long, or ``None`` (HF-style; passed through to ``create_causal_mask``).
        attention_mask_2d: optional HF padding mask ``[B, S]`` (or ``None``).
        or_mask_function, and_mask_function, block_sequence_ids:
            Optional HF ``create_causal_mask`` extras (e.g. multimodal image-token overlays).

    Returns:
        A ``torch.Tensor`` mask to pass into ``Mistral4DecoderLayer`` / hybrid attention, or ``None``
        if the helper short-circuits. Raises if ``create_causal_mask`` returns a non-tensor type
        (e.g. some flex-attn ``BlockMask`` paths), which the current bf16 hybrid stack does not support.
    """
    from transformers.masking_utils import create_causal_mask

    out = create_causal_mask(
        config,
        inputs_embeds_bsh,
        attention_mask_2d,
        past_key_values=None,
        position_ids=position_ids,
        or_mask_function=or_mask_function,
        and_mask_function=and_mask_function,
        block_sequence_ids=block_sequence_ids,
    )
    if out is None:
        return None
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "create_causal_mask returned a non-tensor mask; mistral_small_4 hybrid attention only supports torch.Tensor masks."
        )
    return out


def mistral4_causal_attention_mask(
    config,
    inputs_embeds_bsh: torch.Tensor,
    position_ids: torch.LongTensor | None,
    *,
    attention_mask_2d: torch.Tensor | None = None,
    past_key_values=None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
    block_sequence_ids: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """
    Same as :func:`mistral4_prefill_causal_attention_mask` but forwards ``past_key_values`` to
    ``create_causal_mask`` (prefill + decode / KV-cache steps).
    """
    from transformers.masking_utils import create_causal_mask

    out = create_causal_mask(
        config,
        inputs_embeds_bsh,
        attention_mask_2d,
        past_key_values=past_key_values,
        position_ids=position_ids,
        or_mask_function=or_mask_function,
        and_mask_function=and_mask_function,
        block_sequence_ids=block_sequence_ids,
    )
    if out is None:
        return None
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "create_causal_mask returned a non-tensor mask; mistral_small_4 hybrid attention only supports torch.Tensor masks."
        )
    return out
