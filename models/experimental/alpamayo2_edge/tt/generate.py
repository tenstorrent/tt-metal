# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Chain-of-Causation generation: TT prefill and per-token TT decode, with sampling and stop logic on the host.

The loop follows the reference `ReferenceModel.decode`: rows keep decoding past their own anchor
(`<|traj_future_start|>`) until every row has one, and one more step caches the last anchor, so the cache ends
at the longest prefix for the action expert.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.experimental.alpamayo2_edge.reference.config import TokenIds
from models.experimental.alpamayo2_edge.reference.inputs import ModelInputs
from models.experimental.alpamayo2_edge.reference.model import (
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    find_anchor_offsets,
    process_logits,
)
from models.experimental.alpamayo2_edge.tt.text import AlpamayoCache, TtLanguageModel
from models.tt_dit.utils import tensor


@dataclass
class CotResult:
    tokens: torch.Tensor  # [B, T] generated ids; the last step is the latest row's anchor
    offsets: torch.Tensor  # [B] prompt length + index of each row's first anchor + 1
    prefix_length: int  # rows of `cache` that hold the prompt and the CoT: the largest offset
    logits: torch.Tensor | None  # [B, T, V] fp32 raw logits that tokens[:, t] was drawn from
    # The B-sample cache, positioned at `prefix_length`; owned by the caller, who deallocates it. The expert writes
    # its action rows past the prefix, so the LM cannot decode from it after the flow. None once released.
    cache: AlpamayoCache | None


def sample(
    logits: torch.Tensor,
    *,
    tokens: TokenIds,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    generator: torch.Generator,
) -> torch.Tensor:
    """One id per row of the [B, V] raw logits, drawn after the upstream processors exactly as the reference draws."""
    probs = torch.softmax(process_logits(logits, tokens, temperature, top_p), dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(1)


def _supported_batch(batch: int) -> bool:
    # tt_dit's decode attention lays the batch out on a core range of fewer than 8 or a multiple of 8 cores.
    return batch > 0 and (batch < 8 or batch % 8 == 0)


def check_generation_args(
    inp: ModelInputs, *, num_samples: int, max_new_tokens: int, forced: torch.Tensor | None
) -> None:
    """Raises `ValueError` on arguments `generate_cot` cannot run, before any device work."""
    if inp.input_ids.shape[0] != 1 or not bool(inp.attention_mask.all()):
        msg = "generation takes a single unpadded prompt"
        raise ValueError(msg)
    if not _supported_batch(num_samples):
        msg = f"num_samples must be in [1, 8) or a multiple of 8, got {num_samples}"
        raise ValueError(msg)
    if max_new_tokens < 1:
        msg = f"max_new_tokens must be positive, got {max_new_tokens}"
        raise ValueError(msg)
    if forced is not None and forced.shape[0] != num_samples:
        msg = f"forced CoT has {forced.shape[0]} rows for {num_samples} samples"
        raise ValueError(msg)


def generate_cot(
    lm: TtLanguageModel,
    inp: ModelInputs,
    vision_embeds: ttnn.Tensor,
    *,
    tokens: TokenIds,
    cache: AlpamayoCache,
    num_samples: int = 1,
    forced: torch.Tensor | None = None,
    seed: int = 0,
    max_new_tokens: int = MAX_NEW_TOKENS,
    return_logits: bool = False,
) -> CotResult:
    """Prefills the prompt into `cache`, then samples (or teacher-forces) `num_samples` CoTs to their anchors.

    Args:
        lm: The language model, with its weights loaded.
        inp: A single unpadded prompt.
        vision_embeds: `[V, hidden]` embeddings of the prompt's image tokens, in order.
        tokens: The checkpoint's special token ids.
        cache: An empty single-sequence cache. With `num_samples` > 1 the samples continue in a repeated copy and
            `cache` is deallocated after the prefill; `CotResult.cache` is the one to use and free.
        forced: `[num_samples, T]` ids taken in place of sampling.
        seed: Seeds the sampling generator.
        max_new_tokens: Most tokens generated per row; an anchor drawn on the last one is still cached.
        return_logits: Keeps the raw logits of every step.

    Raises:
        ValueError: The prompt's image-token count differs from the rows of `vision_embeds`, or a row has no anchor
            after `max_new_tokens` tokens or at the end of `forced`.
    """
    check_generation_args(inp, num_samples=num_samples, max_new_tokens=max_new_tokens, forced=forced)
    prompt_len = inp.input_ids.shape[1]
    # The device embedding scatters the vision rows into the image tokens without a bounds check.
    image_tokens = int((inp.input_ids == tokens.image).sum())
    if image_tokens != vision_embeds.shape[0]:
        msg = f"{image_tokens} image tokens for {vision_embeds.shape[0]} vision embeddings"
        raise ValueError(msg)
    rope_deltas = inp.rope_deltas[:, 0].repeat_interleave(num_samples)
    # The last decoded row and its rope index, position + delta, must both fit the cache and its rope table.
    needed = prompt_len + max_new_tokens + max(0, int(rope_deltas.max()))
    if needed > cache.size:
        msg = f"a {cache.size}-row cache cannot hold {prompt_len} prompt and {max_new_tokens} new tokens"
        raise ValueError(msg)

    device = lm.device
    anchor = tokens.future_start
    logits = lm.prefill(
        tensor.from_torch(inp.input_ids, device=device, dtype=ttnn.uint32),
        positions=tensor.from_torch(inp.position_ids.float(), device=device, dtype=ttnn.float32),
        vision_embeds=vision_embeds,
        vision_mask=tensor.from_torch(inp.input_ids == tokens.image, device=device),
        cache=cache,
    )
    step_logits = tensor.to_torch(logits).float().expand(num_samples, -1)
    if num_samples > 1:
        prefill_cache, cache = cache, cache.repeat_to(num_samples)
        prefill_cache.deallocate()

    tt_rope_offset = tensor.from_torch(
        rope_deltas.to(torch.int32), device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    def decode(ids: torch.Tensor) -> torch.Tensor:
        out = lm.decode_step(
            tensor.from_torch(ids, device=device, dtype=ttnn.uint32), cache=cache, rope_offset=tt_rope_offset
        )
        return tensor.to_torch(out).float()[:num_samples]

    generator = torch.Generator().manual_seed(seed)
    generated, kept_logits = [], []
    found = torch.zeros(num_samples, dtype=torch.bool)
    # Rows of the cache written so far; a drawn token is cached by the decode step that consumes it.
    rows_cached = prompt_len
    while True:
        if forced is None:
            next_ids = sample(step_logits, tokens=tokens, generator=generator)
        elif len(generated) < forced.shape[1]:
            next_ids = forced[:, len(generated)].long()
        else:
            rows = (~found).nonzero(as_tuple=True)[0].tolist()
            msg = (
                f"No <traj_future_start> token found in generated rows {rows}: "
                f"the forced CoT ran out after {len(generated)} tokens."
            )
            raise ValueError(msg)
        generated.append(next_ids)
        if return_logits:
            kept_logits.append(step_logits)
        found |= next_ids == anchor
        if len(generated) >= max_new_tokens:
            # Out of budget: the token just drawn is not cached yet.
            break
        step_logits = decode(next_ids)
        rows_cached += 1
        if found.all():
            # The latest anchor is cached; the logits of this step are not used.
            break

    generated = torch.stack(generated, dim=1)
    offsets = find_anchor_offsets(generated, anchor) + prompt_len
    prefix_length = int(offsets.max())
    # Stopping at `max_new_tokens` leaves the last drawn token, the latest anchor, out of the cache.
    for column in range(rows_cached - prompt_len, prefix_length - prompt_len):
        decode(generated[:, column])
        rows_cached += 1

    return CotResult(
        tokens=generated,
        offsets=offsets,
        prefix_length=prefix_length,
        logits=torch.stack(kept_logits, dim=1) if return_logits else None,
        cache=cache,
    )
