# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-side glue between PaddleOCR-VL's vision tower and its text decoder.

Splices projected image embeddings into the text embedding stream at the
``<|IMAGE_PLACEHOLDER|>`` positions, and builds M-RoPE cos/sin tables from
HuggingFace's ``get_rope_index`` plus the text rotary module, converted to
tt-metal's interleaved layout. Same approach as
``models/demos/qwen3_vl/tt/common.py``.
"""

from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import torch

from models.tt_transformers.tt.load_checkpoints import convert_rope_style_hf_to_meta


def splice_image_embeddings(
    input_ids: torch.Tensor,
    text_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Drop ``image_embeds`` into ``text_embeds`` at the placeholder positions.

    ``input_ids`` is ``[S]`` or ``[1, S]``, ``text_embeds`` ``[S, dim]``, and
    ``image_embeds`` ``[n_image_tokens, dim]``. The counts must match exactly:
    a mismatch means the vision tower produced a different number of merged
    tokens than the processor reserved slots for, which is a bug worth failing
    on rather than truncating past.
    """
    ids = input_ids.reshape(-1)
    positions = torch.nonzero(ids == image_token_id, as_tuple=True)[0]

    if positions.numel() == 0:
        return text_embeds.clone()

    assert positions.numel() == image_embeds.shape[0], (
        f"{positions.numel()} image placeholder tokens but {image_embeds.shape[0]} vision embeddings; "
        "the processor's grid and the tower's merge disagree"
    )

    out = text_embeds.clone()
    out[positions] = image_embeds.to(out.dtype)
    return out


def multimodal_rope_from_hf(
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    reference_model,
    model_args,
    pad_token_id: int,
    min_positions: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build M-RoPE cos/sin for one sequence, covering the generated tail too.

    The tables are computed once over a padded length so decode can keep
    indexing into them without a rebuild per step. Returns
    ``(cos, sin, rope_deltas)`` with cos/sin shaped ``[1, 1, padded_len, head_dim]``
    in tt-metal's interleaved layout.

    ``min_positions`` forces the tables to span at least that many positions.
    Callers need it because prefill pads the sequence to its own granularity
    (``get_padded_prefill_len``, a multiple of 128 and at least 1024), and decode
    then indexes past the prompt; tables sized only to the prompt would be short
    on both counts.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    needed = max(input_ids.shape[-1], min_positions)
    padded_len = min(model_args.max_seq_len, max(2 ** math.ceil(math.log(needed, 2)), 128))
    padded = torch.nn.functional.pad(input_ids, (0, padded_len - input_ids.shape[-1]), value=pad_token_id)
    attention_mask = torch.ones_like(padded, dtype=torch.int64)

    kwargs = dict(image_grid_thw=image_grid_thw, video_grid_thw=None, attention_mask=attention_mask)

    # transformers >=5 requires mm_token_type_ids (0=text, 1=image, 2=video). The
    # processor emits it; reconstruct it from the placeholder ids so this helper
    # works from ids alone.
    if "mm_token_type_ids" in inspect.signature(reference_model.model.get_rope_index).parameters:
        config = reference_model.config
        mm_token_type_ids = torch.zeros_like(padded, dtype=torch.int32)
        if getattr(config, "image_token_id", None) is not None:
            mm_token_type_ids[padded == config.image_token_id] = 1
        if getattr(config, "video_token_id", None) is not None:
            mm_token_type_ids[padded == config.video_token_id] = 2
        kwargs["mm_token_type_ids"] = mm_token_type_ids

    position_ids, rope_deltas = reference_model.model.get_rope_index(padded, **kwargs)

    # The HF rotary module only reads .device and .dtype off its first argument,
    # but it does call .to(device) on inv_freq, so the device must be real.
    x = SimpleNamespace(device=torch.device("cpu"), dtype=torch.bfloat16)
    cos, sin = reference_model.model.language_model.rotary_emb(x, position_ids)

    # cos/sin arrive as [3, batch, seq, head_dim]: one table per M-RoPE axis.
    # Collapse them to a single table the way apply_multimodal_rotary_pos_emb
    # does, by walking the head dim in mrope_section-sized chunks and taking the
    # temporal, height and width table in turn. Doubling the section list covers
    # both halves of the rotate-half layout.
    mrope_section = reference_model.config.text_config.rope_parameters["mrope_section"]
    sections = list(mrope_section) * 2
    assert sum(sections) == cos.shape[-1], f"mrope_section {sections} does not sum to head_dim {cos.shape[-1]}"
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(sections, dim=-1))], dim=-1)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(sections, dim=-1))], dim=-1)

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (*convert_rope_style_hf_to_meta(cos, sin), rope_deltas)
