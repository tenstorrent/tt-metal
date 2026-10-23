# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# PyTorch reference for the HunyuanImage-3.0 autoregressive text-generation path
# (recaption / think / img_ratio sub-stages of `generate_image`). Golden for the host
# logic in `tt/generate.py` (which adds only the device backbone adapter on top).
#
# Extracted / adapted from:
#   HunyuanImage-3.0/hunyuan_image_3/modeling_hunyuan_image_3.py
#     HunyuanImage3ForCausalMM._StageTransitionLogitsProcessor   lines 3004-3040 (verbatim)
#     HunyuanImage3ForCausalMM.generate (mode="gen_text")        line 3088
#       (delegates sampling to HF GenerationMixin: temperature / top-k / top-p / repetition
#        penalty — mirrored here against transformers.generation.logits_process)
#
# The sampling math is provider-standard (HF) and inherently host-side, so this golden
# IS the implementation; `tt/generate.py` re-exports it unchanged and contributes only
# the device wiring. The bit-exact tests pin this against upstream + HF.

from dataclasses import dataclass

import torch


@dataclass
class SamplingConfig:
    """HF-style sampling knobs (subset mirrored from the reference GenerationConfig)."""

    do_sample: bool = True
    temperature: float = 1.0
    top_k: int = 0  # 0 disables
    top_p: float = 1.0  # 1.0 disables
    repetition_penalty: float = 1.0  # 1.0 disables
    max_new_tokens: int = 256


class StageTransitionLogitsProcessor:
    """Verbatim port of `generate._StageTransitionLogitsProcessor` (modeling:3004-3040).

    `stage_transitions` is a list of `(stop_id, [append_ids])`: when `stop_id` is the
    last generated token (once), the `append_ids` are force-emitted in order before
    free generation resumes. Operates per batch row, mutating `scores` in place
    (masked to the dtype min except the forced token).
    """

    def __init__(self, stage_transitions, batch_size: int):
        self.transition_map = {stop_id: list(append_ids) for stop_id, append_ids in stage_transitions}
        self.pending_tokens = [[] for _ in range(batch_size)]
        self.completed = [set() for _ in range(batch_size)]

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        last_tokens = input_ids[:, -1]
        min_score = torch.finfo(scores.dtype).min
        for i in range(batch_size):
            last_token = last_tokens[i].item()
            if self.pending_tokens[i] and last_token == self.pending_tokens[i][0]:
                self.pending_tokens[i].pop(0)
            if self.pending_tokens[i]:
                scores[i].fill_(min_score)
                scores[i, self.pending_tokens[i][0]] = 0
                continue
            if last_token in self.transition_map and last_token not in self.completed[i]:
                self.completed[i].add(last_token)
                next_tokens = self.transition_map[last_token]
                if next_tokens:
                    self.pending_tokens[i] = list(next_tokens)
                    scores[i].fill_(min_score)
                    scores[i, self.pending_tokens[i][0]] = 0
        return scores


class ConditionalSliceVocabLogitsProcessor:
    """Port of upstream ``_ConditionalSliceVocabLogitsProcessor`` (modeling:3042-3072).

    When the last token is in ``trigger_token_ids``, restrict sampling to ratio vocab
    slices (used after ``<img_size_N>`` when ``image_size='auto'``).
    """

    def __init__(
        self,
        trigger_token_ids: list[int],
        vocab_start: int,
        vocab_end: int,
        other_slices: list | None = None,
        *,
        force_greedy: bool = False,
    ):
        self.trigger_token_ids = set(trigger_token_ids)
        self.vocab_start = vocab_start
        self.vocab_end = vocab_end
        self.other_slices = other_slices or []
        self.force_greedy = force_greedy

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        last_tokens = input_ids[:, -1]
        min_score = torch.finfo(scores.dtype).min
        for i in range(scores.shape[0]):
            if int(last_tokens[i].item()) not in self.trigger_token_ids:
                continue
            original = scores[i].clone()
            scores[i].fill_(min_score)
            scores[i, self.vocab_start : self.vocab_end] = original[self.vocab_start : self.vocab_end]
            for entry in self.other_slices:
                if isinstance(entry, (tuple, list)) and len(entry) == 2:
                    start, end = entry
                    scores[i, start:end] = original[start:end]
                else:
                    tid = int(entry)
                    scores[i, tid] = original[tid]
            if self.force_greedy:
                max_token_id = int(scores[i].argmax().item())
                scores[i].fill_(min_score)
                scores[i, max_token_id] = 0
        return scores


def apply_repetition_penalty(scores: torch.Tensor, input_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    """HF CTRL-style repetition penalty (matches transformers RepetitionPenaltyLogitsProcessor)."""
    if penalty == 1.0:
        return scores
    for i in range(scores.shape[0]):
        seen = torch.unique(input_ids[i])
        s = scores[i, seen]
        scores[i, seen] = torch.where(s > 0, s / penalty, s * penalty)
    return scores


def top_k_top_p_filter(scores: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Mask logits outside top-k / nucleus(top-p) to the dtype min (matches HF warpers)."""
    min_score = torch.finfo(scores.dtype).min
    if top_k and top_k > 0:
        k = min(top_k, scores.shape[-1])
        kth = torch.topk(scores, k, dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < kth, min_score)
    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scores, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()  # keep at least the top token
        remove[..., 0] = False
        remove_scattered = remove.scatter(-1, sorted_idx, remove)
        scores = scores.masked_fill(remove_scattered, min_score)
    return scores


def sample_next_token(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    config: SamplingConfig,
    *,
    processor: StageTransitionLogitsProcessor = None,
    logits_processors: list | None = None,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """One sampling step: logits [B, V] + history -> next token ids [B].

    Order matches HF / the reference: logits processors (stage transitions) first,
    then warpers (temperature, top-k, top-p), then greedy/multinomial sample.
    """
    scores = logits.float().clone()
    scores = apply_repetition_penalty(scores, input_ids, config.repetition_penalty)
    processors = []
    if processor is not None:
        processors.append(processor)
    if logits_processors:
        processors.extend(logits_processors)
    for lp in processors:
        scores = lp(input_ids, scores)
    if not config.do_sample:
        return scores.argmax(dim=-1)
    if config.temperature != 1.0:
        scores = scores / config.temperature
    scores = top_k_top_p_filter(scores, config.top_k, config.top_p)
    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def generate_text(
    forward_logits_fn,
    input_ids,
    *,
    config: SamplingConfig = None,
    stage_transitions=None,
    final_stop_tokens=None,
    logits_processors: list | None = None,
    generator: torch.Generator = None,
):
    """Autoregressive decode loop. Mirrors `generate(mode="gen_text")` (modeling:3088).

    Args:
        forward_logits_fn: callable `torch.LongTensor [B, S] -> torch.FloatTensor [B, V]`
                           returning next-token logits for the current sequence.
        input_ids:         prompt token ids, torch.LongTensor [B, S] (or [S]).
        config:            SamplingConfig (defaults: sampling on, temperature 1.0).
        stage_transitions: list of (stop_id, [append_ids]) for multi-stage forcing.
                           Requires `final_stop_tokens` (the reference enforces this).
        final_stop_tokens: token ids that end generation (EOS set).

    Returns:
        dict {"sequences": [B, S+n], "new_tokens": list[list[int]] per row}.
    """
    config = config or SamplingConfig()
    if stage_transitions and final_stop_tokens is None:
        raise ValueError("final_stop_tokens must be provided when stage_transitions is set")
    ids = input_ids if isinstance(input_ids, torch.Tensor) else torch.tensor(input_ids)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    ids = ids.long()
    B = ids.shape[0]

    processor = StageTransitionLogitsProcessor(stage_transitions, B) if stage_transitions else None
    stop_set = set(final_stop_tokens or [])
    new_tokens = [[] for _ in range(B)]
    finished = [False] * B

    for _ in range(config.max_new_tokens):
        logits = forward_logits_fn(ids)  # [B, V]
        next_ids = sample_next_token(
            logits,
            ids,
            config,
            processor=processor,
            logits_processors=logits_processors,
            generator=generator,
        )
        for i in range(B):
            if not finished[i]:
                new_tokens[i].append(int(next_ids[i].item()))
        ids = torch.cat([ids, next_ids.unsqueeze(1)], dim=1)
        for i in range(B):
            if not finished[i] and int(next_ids[i].item()) in stop_set:
                finished[i] = True
        if all(finished) and stop_set:
            break

    return {"sequences": ids, "new_tokens": new_tokens}
