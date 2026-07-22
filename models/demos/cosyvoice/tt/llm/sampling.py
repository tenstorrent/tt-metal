"""RAS (Repetition-Aware Sampling) for CosyVoice2 LLM decode.

Ported from `cosyvoice/utils/common.py::ras_sampling` (reference repo @ 074ca6d).
Runs entirely on host (CPU) — logits are read back from device each step.
"""

from __future__ import annotations

from typing import List

import torch


def nucleus_sampling(weighted_scores: torch.Tensor, top_p: float = 0.8, top_k: int = 25) -> int:
    prob: List[float] = []
    indices: List[int] = []
    cum_prob = 0.0
    sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
    for i in range(len(sorted_idx)):
        if cum_prob < top_p and len(prob) < top_k:
            cum_prob += sorted_value[i].item()
            prob.append(sorted_value[i].item())
            indices.append(sorted_idx[i].item())
        else:
            break
    prob_t = torch.tensor(prob, device=weighted_scores.device)
    indices_t = torch.tensor(indices, dtype=torch.long, device=weighted_scores.device)
    top_ids = indices_t[prob_t.multinomial(1, replacement=True)].item()
    return top_ids


def random_sampling(weighted_scores: torch.Tensor) -> int:
    top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True).item()
    return top_ids


def ras_sampling(
    weighted_scores: torch.Tensor,
    decoded_tokens: List[int],
    top_p: float = 0.8,
    top_k: int = 25,
    win_size: int = 10,
    tau_r: float = 0.1,
) -> int:
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
    rep_num = (torch.tensor(decoded_tokens[-win_size:]).to(weighted_scores.device) == top_ids).sum().item()
    if rep_num >= win_size * tau_r:
        weighted_scores[top_ids] = -float("inf")
        top_ids = random_sampling(weighted_scores)
    return top_ids


def sampling_ids(
    weighted_scores: torch.Tensor,
    decoded_tokens: List[int],
    speech_token_size: int = 6561,
    ignore_eos: bool = True,
    top_p: float = 0.8,
    top_k: int = 25,
    win_size: int = 10,
    tau_r: float = 0.1,
) -> int:
    if ignore_eos:
        weighted_scores[speech_token_size] = -float("inf")
    return ras_sampling(weighted_scores, decoded_tokens, top_p, top_k, win_size, tau_r)
