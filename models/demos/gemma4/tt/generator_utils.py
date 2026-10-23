# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python helpers of the Gemma 4 vLLM generator that do not need vLLM.

``generator_vllm`` imports vLLM at module scope, so anything unit-tested on a
runner without vLLM (the tt-metal unit-test jobs) lives here instead.
"""

import torch


def filter_warmup_sampling_params(configs, sampler):
    """Drop warmup sampling configs a sampler cannot run.

    ``sampler._allow_topk_sampling`` False: no non-greedy config at all (they all need the
    top-k/top-p program). ``sampler._allow_penalties_sampling`` False: no config with penalties.
    Greedy (temperature 0) and ``None`` always stay.
    """
    allow_penalties = bool(getattr(sampler, "_allow_penalties_sampling", True))
    allow_topk = bool(getattr(sampler, "_allow_topk_sampling", True))
    kept = []
    for cfg in configs:
        if cfg is None:
            kept.append(cfg)
            continue
        temperature = cfg.temperature
        first = temperature[0] if isinstance(temperature, (list, tuple)) else temperature
        if first == 0.0:
            kept.append(cfg)
            continue
        if not allow_topk:
            continue
        penalties = cfg.presence_penalty
        has_penalties = penalties is not None and (
            any(p not in (None, 0.0) for p in penalties) if isinstance(penalties, (list, tuple)) else penalties != 0.0
        )
        if not allow_penalties and has_penalties:
            continue
        kept.append(cfg)
    return kept


def merge_soft_tokens(input_ids: torch.Tensor, soft_tokens, pad_token_id: int, *, num_cached: int = 0):
    """Line up every modality's soft tokens with its placeholder slots in one prompt.

    ``soft_tokens`` is a list of ``(placeholder_token_id, embeds)`` with ``embeds`` a
    host ``[num_valid, text_hidden]`` tensor in prompt order for that modality.
    Returns ``(llm_ids, positions, merged_embeds)``: the prompt ids with every
    placeholder replaced by ``pad_token_id``, the placeholder positions sorted
    ascending across modalities (``int32 [num_valid]``) and the soft tokens in
    that same order (``bf16 [num_valid, text_hidden]``), ready for one scatter.

    ``num_cached`` (vLLM prefix caching): an item whose placeholders all sit inside
    the cached prefix arrives without its payload, so a modality may carry fewer
    soft tokens than placeholders. They then fill that modality's LAST placeholders
    (items are forwarded in prompt order, cached ones first) and every placeholder
    past the cached prefix must receive one. Any other count mismatch raises.
    """
    input_ids = input_ids.reshape(-1).to(torch.long)
    placeholder_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    positions_per_modality = []
    embeds_per_modality = []
    for token_id, embeds in soft_tokens:
        if token_id is None:
            raise ValueError("soft tokens for a modality this checkpoint has no placeholder id for")
        slot_mask = input_ids == token_id
        slots = torch.where(slot_mask)[0]
        n_slots, n_soft = int(slots.numel()), int(embeds.shape[0])
        if n_soft != n_slots:
            uncovered = slots[: max(n_slots - n_soft, 0)]
            if n_soft > n_slots or num_cached <= 0 or int(uncovered.max()) >= num_cached:
                raise ValueError(
                    f"Placeholder count for token id {token_id} ({n_slots}) != soft tokens ({n_soft}). "
                    "The processor's placeholder run lengths and the embedder's valid-token counts must agree "
                    "(a shortfall is only valid for items fully inside the cached prefix)."
                )
            slots = slots[n_slots - n_soft :]
        positions_per_modality.append(slots)
        embeds_per_modality.append(embeds.to(torch.bfloat16))
        placeholder_mask |= slot_mask
    if not positions_per_modality:
        raise ValueError("no soft tokens to merge")
    positions = torch.cat(positions_per_modality)
    order = torch.argsort(positions)
    positions = positions[order].to(torch.int32)
    merged_embeds = torch.cat(embeds_per_modality, dim=0)[order].contiguous()
    llm_ids = input_ids.clone()
    llm_ids[placeholder_mask] = pad_token_id
    return llm_ids, positions, merged_embeds


def items_within_prefill_window(offsets, lengths, window_end: int):
    """Indices of a request's multimodal items whose placeholders end inside ``window_end``.

    The plugin forwards every item of a scheduled request on every prefill step, while
    a chunked prefill only covers the prompt up to the chunk's end. An item whose
    placeholders start past that end has no slots yet and belongs to a later chunk;
    merging it now would fail the placeholder count. vLLM never ends a chunk inside an
    item (``disable_chunked_mm_input``), so an item is either fully inside the window or
    fully past it. ``offsets`` / ``lengths`` are the per-item ``mm_position`` lists the
    plugin forwards (``mm_placeholder_offsets`` / ``mm_placeholder_lengths``); an item
    without a range (older plugins) is kept.
    """
    n = max(len(offsets or ()), len(lengths or ()))
    kept = []
    for i in range(n):
        offset = offsets[i] if offsets is not None and i < len(offsets) else None
        length = lengths[i] if lengths is not None and i < len(lengths) else None
        if offset is not None and length is not None and int(offset) + int(length) > int(window_end):
            continue
        kept.append(i)
    return kept
