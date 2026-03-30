# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared draft branching helpers for NextN MTP adapters (bundle-local minimal copy)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

from specfr.config import EagleConfig


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(dtype)


def _clone_hf_kv(past_key_values: tuple) -> tuple:
    return tuple(
        tuple(t.clone() for t in layer_kv)
        for layer_kv in past_key_values
    )


def _draft_path_rank_key(appended_probs: Sequence[float]) -> tuple[float, float, float]:
    if not appended_probs:
        return (float("-inf"), float("-inf"), float("-inf"))
    p = list(appended_probs)
    return (max(p), p[-1], sum(p))


def draft_requires_positive_top_k(cfg: EagleConfig) -> bool:
    if getattr(cfg, "draft_mtp_greedy", False):
        return False
    if getattr(cfg, "draft_branching", "top_k") == "temperature_top_p":
        return False
    return cfg.top_k <= 0


def truncate_beams_by_draft_confidence(
    beams: list,
    max_paths: int,
    get_appended_probs: Callable[[object], Sequence[float]],
) -> list:
    if max_paths <= 0 or len(beams) <= max_paths:
        return beams
    return sorted(
        beams,
        key=lambda b: _draft_path_rank_key(get_appended_probs(b)),
        reverse=True,
    )[:max_paths]


def draft_branch_token_ids_from_logits(
    logits_1d: torch.Tensor,
    cfg: EagleConfig,
    k: int,
    generator: torch.Generator | None,
) -> list[int]:
    branching = getattr(cfg, "draft_branching", "top_k")
    if branching not in ("top_k", "temperature_top_p"):
        raise ValueError(f"Unknown draft_branching={branching!r}; expected 'top_k' or 'temperature_top_p'.")
    logits_flat = logits_1d.float().reshape(-1)
    vocab = int(logits_flat.numel())
    k_raw = int(k)

    if branching != "temperature_top_p":
        if k_raw <= 0:
            return []
        ki = min(k_raw, vocab)
        _, idx = torch.topk(logits_flat, k=ki, dim=-1)
        return [int(x) for x in idx.tolist()]

    temp = float(getattr(cfg, "draft_temperature", 0.6))
    top_p = float(getattr(cfg, "draft_top_p", 0.95))
    top_p = min(max(top_p, 1e-6), 1.0)

    if temp <= 0:
        return [int(torch.argmax(logits_flat, dim=-1).item())]

    probs = F.softmax(logits_flat / temp, dim=-1).cpu()
    sorted_p, sorted_i = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_p, dim=-1)
    ge = cumsum >= top_p
    if bool(ge.any().item()):
        n_keep = int(ge.nonzero(as_tuple=False)[0].item()) + 1
    else:
        n_keep = int(sorted_p.numel())
    n_keep = max(1, min(n_keep, int(sorted_p.numel())))
    slice_p = sorted_p[:n_keep].clone()
    slice_p = slice_p / (slice_p.sum() + 1e-20)

    if k_raw <= 0:
        return [int(sorted_i[j].item()) for j in range(n_keep)]

    n_draw = min(min(k_raw, vocab), n_keep)
    draws = torch.multinomial(slice_p, num_samples=n_draw, replacement=False, generator=generator)
    return [int(sorted_i[int(j)].item()) for j in draws.tolist()]
