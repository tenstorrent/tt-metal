# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host mirror of the on-device ttnn.sampling rule (top-k, temperature, top-p) used by DFlash2 verification."""

import torch
from loguru import logger


def normalize_sampling(sampling, max_top_k):
    """Return None for greedy, else (temperature, top_p, k) with k = max_top_k if top_k < 1 else min(top_k, max_top_k).

    top_p must be in (0, 1], as in HF and vLLM; top_p=0 is rejected on purpose (ValueError).
    """
    if sampling is None:
        return None
    temperature = float(sampling.get("temperature", 1.0))
    top_p = float(sampling.get("top_p", 1.0))
    top_k = int(sampling.get("top_k", 0))
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if temperature == 0:
        return None
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if top_k > max_top_k:
        logger.warning(f"top_k={top_k} exceeds device max_top_k={max_top_k}; clamping")
    k = max_top_k if top_k < 1 else min(top_k, max_top_k)
    return temperature, top_p, k


def lane_seeds(seed, n=32):
    """n distinct, well-mixed seeds in [1, 2**31 - 1) derived from `seed`.
    Seeds `seed + i` correlate lanes across steps through the seed manager's (seed XOR counter) hash,
    so lanes need unrelated seeds."""
    g = torch.Generator().manual_seed(int(seed))
    out = []
    while len(out) < n:
        v = int(torch.randint(1, 2**31 - 1, (1,), generator=g))
        if v not in out:
            out.append(v)
    return out


def _bf16(x):
    return float(torch.tensor(float(x), dtype=torch.float32).to(torch.bfloat16).float())


def filtered_probs(values, ids, temperature, top_p, top_k):
    """values [R, C] float logits, ids [R, C] int64 (C >= top_k, any column order), top_k already normalised.
    Returns (probs [R, top_k] float32, ids [R, top_k] int64), rows sorted by logit descending."""
    values = torch.as_tensor(values).float().to(torch.bfloat16).float()
    ids = torch.as_tensor(ids).long()
    top_values, index = torch.sort(values, dim=-1, descending=True, stable=True)
    top_values = top_values[:, :top_k]
    top_ids = ids.gather(-1, index[:, :top_k])
    if top_k == 1:
        return torch.ones_like(top_values), top_ids
    inv_t = _bf16(1.0 / temperature)
    probs = torch.softmax(top_values * inv_t, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    over = cum > _bf16(top_p)
    first = torch.where(over.any(dim=-1), over.float().argmax(dim=-1), torch.full((probs.shape[0],), top_k - 1))
    keep = torch.arange(top_k).unsqueeze(0) <= first.unsqueeze(-1)
    probs = probs * keep
    return probs / probs.sum(dim=-1, keepdim=True), top_ids


def draw(probs_row, ids_row, generator):
    """Draw one token id from a probability row."""
    return int(ids_row[int(torch.multinomial(probs_row, 1, generator=generator))])
