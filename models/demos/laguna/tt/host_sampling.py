# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host sampling for requests the traced device sampler cannot serve (repetition/presence/frequency
penalties).

The TT vLLM plugin keeps penalized requests on the device-sampling path and passes the token history
(``prompt_tokens`` / ``output_tokens``) to ``decode_forward``. Laguna's traced sampler (Sampling1D)
has no penalty support, so the adapter takes logits from an eager decode and samples here instead.
The math follows vLLM's torch reference (``vllm.model_executor.layers.utils.apply_penalties`` and
``vllm._custom_ops.apply_repetition_penalties_torch``); this module only adds the plugin's ``-1``
history padding and per-row seeded sampling. Pure torch, no device.
"""
from __future__ import annotations

import secrets

import torch


def penalties_active(sp) -> bool:
    """True when any row asks for a repetition, presence or frequency penalty."""
    if sp is None:
        return False
    rep = getattr(sp, "repetition_penalty", None)
    pres = getattr(sp, "presence_penalty", None)
    freq = getattr(sp, "frequency_penalty", None)
    return bool(
        (rep is not None and (torch.as_tensor(rep, dtype=torch.float32) != 1.0).any())
        or (pres is not None and (torch.as_tensor(pres, dtype=torch.float32) != 0.0).any())
        or (freq is not None and (torch.as_tensor(freq, dtype=torch.float32) != 0.0).any())
    )


def _bin_counts(tokens: torch.Tensor, vocab: int, rows: int) -> torch.Tensor:
    """[rows, vocab] occurrence counts; ids outside [0, vocab) (plugin pads with -1) are ignored."""
    counts = torch.zeros((rows, vocab + 1), dtype=torch.int64)
    if tokens is None or tokens.numel() == 0:
        return counts[:, :vocab]
    ids = torch.as_tensor(tokens, dtype=torch.int64)[:rows]
    ids = torch.where((ids >= 0) & (ids < vocab), ids, torch.full_like(ids, vocab))
    counts[: ids.shape[0]].scatter_add_(1, ids, torch.ones_like(ids))
    return counts[:, :vocab]


def apply_penalties(logits, prompt_tokens, output_tokens, presence, frequency, repetition):
    """Penalize ``logits`` [rows, vocab] (float32, returned new) exactly as vLLM does."""
    logits = logits.to(torch.float32).clone()
    rows, vocab = logits.shape
    prompt_mask = _bin_counts(prompt_tokens, vocab, rows) > 0
    output_counts = _bin_counts(output_tokens, vocab, rows)
    output_mask = output_counts > 0

    def per_row(values, neutral):
        if values is None:
            return torch.full((rows,), neutral, dtype=torch.float32)
        v = torch.as_tensor(values, dtype=torch.float32).reshape(-1)[:rows]
        if v.numel() < rows:
            v = torch.cat([v, torch.full((rows - v.numel(),), neutral, dtype=torch.float32)])
        return v

    rep = per_row(repetition, 1.0).unsqueeze(1).expand(rows, vocab)
    penalties = torch.where(prompt_mask | output_mask, rep, torch.ones_like(rep))
    logits *= torch.where(logits > 0, 1.0 / penalties, penalties)
    logits -= per_row(frequency, 0.0).unsqueeze(1) * output_counts.to(torch.float32)
    logits -= per_row(presence, 0.0).unsqueeze(1) * output_mask.to(torch.float32)
    return logits


def sample_row(logits_row: torch.Tensor, *, temperature: float, top_k: int, top_p: float, seed, step: int) -> int:
    """Greedy (temperature <= 0) or temperature/top-k/top-p sampling of one [vocab] row.

    A request seed makes the draw reproducible: the generator is seeded from (seed, step), so identical
    seeded requests produce identical tokens regardless of what else ran before. No seed draws fresh."""
    if temperature <= 0.0:
        return int(torch.argmax(logits_row))
    scaled = logits_row.to(torch.float32) / float(temperature)
    if 0 < top_k < scaled.numel():
        kth = torch.topk(scaled, int(top_k)).values[-1]
        scaled = torch.where(scaled < kth, torch.full_like(scaled, float("-inf")), scaled)
    probs = torch.softmax(scaled, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, order = torch.sort(probs, descending=True)
        keep = torch.cumsum(sorted_probs, dim=-1) - sorted_probs < top_p  # always keeps the top token
        sorted_probs = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        probs = torch.zeros_like(probs).scatter_(0, order, sorted_probs)
        probs /= probs.sum()
    generator = torch.Generator()
    base = int(seed) if seed is not None else secrets.randbelow(2**62)
    generator.manual_seed((base * 1_000_003 + int(step)) % (2**63 - 1))
    return int(torch.multinomial(probs, 1, generator=generator))


def sample_penalized(logits, sp, prompt_tokens, output_tokens, positions) -> torch.Tensor:
    """Apply penalties then sample every row; returns int32 token ids [rows]."""
    logits = apply_penalties(
        logits,
        prompt_tokens,
        output_tokens,
        getattr(sp, "presence_penalty", None),
        getattr(sp, "frequency_penalty", None),
        getattr(sp, "repetition_penalty", None),
    )
    rows = logits.shape[0]
    n = 0 if getattr(sp, "temperature", None) is None else len(sp.temperature)
    out = torch.zeros(rows, dtype=torch.int32)
    for row in range(rows):
        if row >= n:
            out[row] = int(torch.argmax(logits[row]))
            continue
        temperature = float(sp.temperature[row]) if sp.temperature is not None else 1.0
        top_k = int(sp.top_k[row]) if getattr(sp, "top_k", None) is not None else 0
        top_p = float(sp.top_p[row]) if getattr(sp, "top_p", None) is not None else 1.0
        seed = sp.seed[row] if getattr(sp, "seed", None) is not None else None
        seed = None if seed is None or int(seed) < 0 else int(seed)  # -1 is the plugin's "no seed"
        out[row] = sample_row(
            logits[row], temperature=temperature, top_k=top_k, top_p=top_p, seed=seed, step=int(positions[row])
        )
    return out
