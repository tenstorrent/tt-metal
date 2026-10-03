"""Host-side sampler: an exact port of fish-speech inference.py sample()/logits_to_probs() + RAS.

Used in Phase A (host sampling on the readback logits). Phase D moves this on device.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import torch

from models.autoports.fishaudio_s2_pro.config import DEFAULT_TOP_K, RAS_HIGH_TEMP, RAS_HIGH_TOP_P, RAS_WIN_SIZE


def logits_to_probs(logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
    """fish-speech inference.py:54-77 (float32)."""
    logits = logits.float()
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    top_k_mask = torch.arange(logits.shape[-1]) >= top_k
    remove = (cum_probs > top_p) | top_k_mask
    remove[0] = False
    indices_to_remove = remove.scatter(-1, sorted_indices, remove)
    logits = torch.where(indices_to_remove, torch.tensor(float("-inf")), logits)
    logits = logits / max(temperature, 1e-5)
    return torch.nn.functional.softmax(logits, dim=-1)


def gumbel_race(probs: torch.Tensor, generator: Optional[torch.Generator] = None) -> int:
    """multinomial_sample_one_no_sync: argmax(probs / Exp(1))."""
    q = torch.rand(probs.shape, generator=generator, dtype=probs.dtype)
    q = -torch.log(q.clamp_min(1e-20))
    return int(torch.argmax(probs / q))


def sample(logits: torch.Tensor, temperature: float, top_p: float, top_k: int = DEFAULT_TOP_K, generator=None) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits))
    return gumbel_race(logits_to_probs(logits, temperature, top_p, top_k), generator)


class S2Sampler:
    """Constrained slow-token sampling (semantic range + <|im_end|>) with RAS, plus fast-codebook sampling."""

    def __init__(
        self,
        vocab_size: int,
        semantic_begin: int,
        semantic_end: int,
        im_end: int,
        top_k: int = DEFAULT_TOP_K,
        seed: Optional[int] = None,
    ):
        self.bias = torch.full((vocab_size,), float("-inf"), dtype=torch.float32)
        self.bias[semantic_begin : semantic_end + 1] = 0.0
        self.bias[im_end] = 0.0
        self.semantic_begin, self.semantic_end, self.im_end = semantic_begin, semantic_end, im_end
        self.top_k = top_k
        self.gen = torch.Generator()
        if seed is not None:
            self.gen.manual_seed(seed)
        self.previous = deque(maxlen=RAS_WIN_SIZE)

    def reset(self):
        self.previous.clear()

    def sample_slow(self, logits: torch.Tensor, temperature: float, top_p: float, greedy: bool = False) -> int:
        biased = logits.float()[: self.bias.shape[0]] + self.bias
        if greedy or temperature <= 0:
            tok = int(torch.argmax(biased))
        else:
            tok = sample(biased, temperature, top_p, self.top_k, self.gen)
            high = sample(biased, RAS_HIGH_TEMP, RAS_HIGH_TOP_P, self.top_k, self.gen)
            is_semantic = self.semantic_begin <= tok <= self.semantic_end
            if is_semantic and tok in self.previous:
                tok = high
        self.previous.append(tok)
        return tok

    def sample_fast(self, logits: torch.Tensor, temperature: float, top_p: float, greedy: bool = False) -> int:
        if greedy or temperature <= 0:
            return int(torch.argmax(logits))
        return sample(logits.float(), temperature, top_p, self.top_k, self.gen)
