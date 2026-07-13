# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side sampling helpers for Gemma4 demos and the FastAPI server."""

from __future__ import annotations

import torch


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """HF-style repetition penalty over tokens present in *input_ids*.

    Positive logits are divided by *penalty*; negative logits are multiplied.
    No-op when ``penalty <= 1.0``.
    """
    if penalty <= 1.0:
        return logits

    scores = logits.float().clone()
    squeezed = False
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
        squeezed = True

    if input_ids.dim() == 1:
        batch_ids = [input_ids]
    else:
        batch_ids = [input_ids[i] for i in range(scores.shape[0])]

    vocab = scores.shape[-1]
    for batch_idx, ids in enumerate(batch_ids):
        if batch_idx >= scores.shape[0]:
            break
        for token_id in set(int(t) for t in ids.tolist()):
            if token_id < 0 or token_id >= vocab:
                continue
            score = scores[batch_idx, token_id]
            scores[batch_idx, token_id] = torch.where(
                score < 0,
                score * penalty,
                score / penalty,
            )

    if squeezed:
        scores = scores.squeeze(0)
    return scores.to(dtype=logits.dtype)


def host_sample(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    *,
    input_ids: torch.Tensor | None = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """Sample next tokens on host. Greedy argmax for temperature==0, else top-p.

    *logits*: ``[B, vocab]`` or ``[B, 1, vocab]``.
    Returns: ``torch.LongTensor [B, 1]``.
    """
    if logits.dim() == 3:
        logits = logits[:, -1, :]

    if input_ids is not None and repetition_penalty > 1.0:
        logits = apply_repetition_penalty(logits, input_ids, repetition_penalty)

    if not temperature or temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    probs = torch.softmax(logits.float() / temperature, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    choice = torch.multinomial(sorted_probs, num_samples=1)
    return torch.gather(sorted_idx, -1, choice)


class RepetitionStreakGuard:
    """Stop generation when the same token repeats *max_streak* times in a row."""

    def __init__(self, max_streak: int) -> None:
        self.max_streak = max_streak
        self._last_tok: int | None = None
        self._streak = 0

    def reset(self) -> None:
        self._last_tok = None
        self._streak = 0

    def observe(self, token_id: int) -> bool:
        """Record *token_id*. Return True if the streak limit is reached."""
        if self.max_streak <= 0:
            return False
        if self._last_tok == token_id:
            self._streak += 1
        else:
            self._last_tok = token_id
            self._streak = 1
        return self._streak >= self.max_streak
