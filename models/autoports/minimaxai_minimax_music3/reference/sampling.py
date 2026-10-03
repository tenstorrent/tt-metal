"""Sampling contract (encoders.py): top-k multinomial with a torch.Generator (CPU), the c0 classifier-free guidance
with the conditional row's top-k restriction, and the depth-code guidance. All in float32."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from models.autoports.minimaxai_minimax_music3.config import (
    AR_CFG_SCALE,
    AR_CFG_TOP_K,
    AR_SAMPLING_TOP_K,
    AUDIO_CODE_OFFSET,
    AUDIO_END_TOKEN_ID,
    SEMANTIC_VOCAB_SIZE,
)


def sample_top_k(
    logits: torch.Tensor, generator: Optional[torch.Generator], top_k: int = AR_SAMPLING_TOP_K
) -> torch.Tensor:
    """logits [..., V] -> sampled index [...] (exact port of `_sample_top_k`)."""
    values = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    top_k = min(top_k, values.shape[-1])
    threshold = torch.topk(values, top_k, dim=-1).values[..., -1, None]
    values = values.masked_fill(values < threshold, -float("inf"))
    probs = torch.nan_to_num(F.softmax(values, dim=-1), nan=0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sample_device = generator.device if generator is not None else probs.device
    return torch.multinomial(probs.to(sample_device), 1, generator=generator).squeeze(-1).to(probs.device)


def full_vocab_mask(vocab_size: int, device=None) -> torch.Tensor:
    """True = masked. Only the semantic range and the end-of-audio token are allowed for c0."""
    m = torch.ones(vocab_size, dtype=torch.bool, device=device)
    m[AUDIO_CODE_OFFSET : AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE] = False
    m[AUDIO_END_TOKEN_ID] = False
    return m


def guided_c0_logits_full(logits: torch.Tensor, vocab_mask: torch.Tensor) -> torch.Tensor:
    """Full-vocab version (reference): logits [2, V] float (row 0 cond, row 1 uncond) -> guided [1, V]."""
    logits = logits.float().masked_fill(vocab_mask, -float("inf"))
    conditional, unconditional = logits[0:1], logits[1:2]
    guided = unconditional + (conditional - unconditional) * AR_CFG_SCALE
    threshold = torch.topk(conditional, AR_CFG_TOP_K, dim=-1).values[..., -1, None]
    guided = guided.masked_fill(conditional < threshold, -float("inf"))
    return guided.masked_fill(vocab_mask.unsqueeze(0), -float("inf"))


def slice_logits(logits_full: torch.Tensor) -> torch.Tensor:
    """[.., V] -> [.., 16385]: rows 0..16383 = semantic codes, row 16384 = end-of-audio."""
    return torch.cat(
        (
            logits_full[..., AUDIO_CODE_OFFSET : AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE],
            logits_full[..., AUDIO_END_TOKEN_ID : AUDIO_END_TOKEN_ID + 1],
        ),
        dim=-1,
    )


def guided_c0_logits_sliced(logits: torch.Tensor) -> torch.Tensor:
    """Sliced version (what the TT head produces): logits [2, 16385] float -> guided [1, 16385]. Identical to the
    full version restricted to the un-masked ids (a -inf entry never enters a top-k threshold in the full version)."""
    logits = logits.float()
    conditional, unconditional = logits[0:1], logits[1:2]
    guided = unconditional + (conditional - unconditional) * AR_CFG_SCALE
    threshold = torch.topk(conditional, AR_CFG_TOP_K, dim=-1).values[..., -1, None]
    return guided.masked_fill(conditional < threshold, -float("inf"))


def sliced_to_token(index: int) -> int:
    return AUDIO_END_TOKEN_ID if index == SEMANTIC_VOCAB_SIZE else index + AUDIO_CODE_OFFSET


def guided_depth_logits(logits: torch.Tensor) -> torch.Tensor:
    """logits [2, 1024] (cond, uncond) -> guided [1, 1024] (CFG only; top-k happens in sample_top_k)."""
    conditional, unconditional = logits[:1].float(), logits[1:2].float()
    return unconditional + (conditional - unconditional) * AR_CFG_SCALE
