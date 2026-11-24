# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host references for device-sampled verify tests: vLLM's sampling distribution and a chi-square p-value.

``target_probs`` is the distribution vLLM's v1 sampler draws from (vllm/v1/sample/ops/topk_topp_sampler.py
``apply_top_k_top_p``): logits divided by the temperature, everything below the k-th largest value removed (ties at
the k-th value kept), then, over the softmax of what is left in ascending order, every token whose cumulative mass is
at most 1 - top_p removed, the largest always kept. ``top_p_rule="at_most"`` is the other reading of a top-p cut: keep
the largest tokens while their cumulative mass stays at most top_p (at least one).
"""

import torch


def target_probs(logits, temperature, top_k, top_p, top_p_rule="vllm"):
    """[V] float64 probabilities of one row's draw."""
    x = torch.as_tensor(logits, dtype=torch.float64) / float(temperature)
    if 0 < int(top_k) < x.numel():
        kth = torch.topk(x, int(top_k)).values[-1]
        x = torch.where(x < kth, torch.full_like(x, float("-inf")), x)
    if float(top_p) < 1.0:
        order = torch.argsort(x, descending=True)
        probs = torch.softmax(x[order], dim=0)
        cum = torch.cumsum(probs, dim=0)
        if top_p_rule == "vllm":
            keep = (cum - probs) < float(top_p)  # the ascending tail with mass <= 1 - p goes
        elif top_p_rule == "at_most":
            keep = cum <= float(top_p)
        else:
            raise ValueError(f"unknown top_p_rule {top_p_rule!r}")
        keep[0] = True
        drop = order[~keep]
        x[drop] = float("-inf")
    return torch.softmax(x, dim=0)


def chi_square_p(counts, probs, min_expected=5.0):
    """p-value of observed ``counts`` [V] against ``probs`` [V]: Pearson's statistic over the cells whose expected
    count is at least ``min_expected``, the rest pooled into one cell. A count in a cell of probability 0 gives 0."""
    counts = torch.as_tensor(counts, dtype=torch.float64)
    probs = torch.as_tensor(probs, dtype=torch.float64)
    n = float(counts.sum())
    if bool(((probs == 0) & (counts > 0)).any()):
        return 0.0
    expected = probs * n
    big = expected >= min_expected
    obs = torch.cat([counts[big], counts[~big].sum().reshape(1)])
    exp = torch.cat([expected[big], expected[~big].sum().reshape(1)])
    if float(exp[-1]) < min_expected:
        obs, exp = obs[:-1], exp[:-1]
        obs[-1] += counts[~big].sum()
        exp[-1] += expected[~big].sum()
    df = obs.numel() - 1
    if df < 1:
        return 1.0
    stat = float(((obs - exp) ** 2 / exp).sum())
    return float(torch.special.gammaincc(torch.tensor(df / 2.0), torch.tensor(stat / 2.0)))
