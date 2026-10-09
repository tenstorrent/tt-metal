# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU model of the in-trace sampler (tt/device_sampler.py) against the exact full-vocabulary sampler.

Why these tests exist next to the 'sample_exact' ones of test_vllm_interface.py: the in-trace draw (a) takes the inverse CDF in VOCABULARY order, not in sorted-candidate order, so for the same uniform it returns
another token of the same distribution (token identity with the sorted draw is not expected; the distribution is checked), and (b) finds the top-k / top-p thresholds by a k-ary search with a finite resolution
(range / J**levels in scaled-logit units), so tokens within that distance of the exact boundary may be kept or dropped (the boundary band is checked explicitly).
"""

import pytest
import torch

from models.demos.blackhole.deepseek_v41_flash.tt.device_sampler import (
    emulate_draw,
    emulate_keep_weights,
    emulate_topk_weights,
)

V = 1600
J, LEVELS, LEVELS_K = 16, 3, 4  # the production sampler: tt/dsv41_model.Model._make_sampler


def exact_probs(full, T, tk, tp):
    srt, si = torch.sort(full.double(), descending=True)
    pr = torch.softmax(srt / T, 0)
    if tk > 0:
        pr, si = pr[:tk] / pr[:tk].sum(), si[:tk]
    if tp < 1.0:
        pr = pr * ((pr.cumsum(0) - pr) < tp)
        pr = pr / pr.sum()
    out = torch.zeros(full.numel(), dtype=torch.double)
    out[si] = pr
    return out


CFGS = [
    (1.0, 0, 0.95),
    (1.0, 0, 1.0),
    (0.6, 0, 0.9),
    (1.0, 7, 0.95),
    (1.0, 5, 1.0),
    (1.3, 40, 0.8),
    (1.0, 129280, 0.95),
]


@pytest.mark.parametrize("scale", [0.3, 3.0, 12.0])  # flat -> peaked rows
@pytest.mark.parametrize("cfg", CFGS)
def test_kept_set_is_the_exact_support_up_to_the_boundary_band(scale, cfg):
    torch.manual_seed(3)
    full = torch.randn(V) * scale
    T, tk, tp = cfg
    ex = exact_probs(full, T, tk, tp) > 0
    got = emulate_keep_weights(full, T, tk, tp, J, LEVELS, LEVELS_K) > 0
    diff = ex ^ got
    if diff.any():  # every wrongly kept / dropped token lies within the search resolution of the boundary
        s = (full - full.max()) / T
        kept_min = s[ex].min()
        res = (s.max() - s.min()) / J**LEVELS * 1.5 + 1e-3
        assert bool((s[diff] - kept_min).abs().max() <= res), (cfg, scale, int(diff.sum()))


@pytest.mark.parametrize("scale", [0.3, 3.0])
@pytest.mark.parametrize("cfg", CFGS)
def test_distribution_of_the_vocabulary_order_draw_matches_the_exact_sampler(scale, cfg):
    """Total variation distance of the empirical law of many draws against the exact nucleus distribution (the noise floor of N draws over C support tokens is about sqrt(C / (2 pi N)))."""
    torch.manual_seed(5)
    full = torch.randn(V) * scale
    T, tk, tp = cfg
    ref = exact_probs(full, T, tk, tp)
    w = emulate_keep_weights(full, T, tk, tp, J, LEVELS, LEVELS_K)
    N = 200000
    u = torch.rand(N, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    emp = torch.bincount(emulate_draw(w, u), minlength=V).double() / N
    tv = 0.5 * float((emp - ref).abs().sum())
    C = int((ref > 0).sum())
    assert tv < 3.0 * (C / (2 * 3.14159 * N)) ** 0.5 + 5e-3, (tv, C)


def test_greedy_rows_are_the_exact_argmax_and_top_k_one_is_a_point_mass():
    torch.manual_seed(2)
    full = torch.randn(V) * 3
    w = emulate_keep_weights(full, 1.0, 1, 1.0, J, LEVELS, LEVELS_K)
    assert int((w > 0).sum()) >= 1 and int(emulate_draw(w, 0.37)) == int(full.argmax())


def test_a_flat_vocabulary_row_keeps_exactly_top_k_tokens_with_the_extra_count_level():
    """Flat rows (a real vocabulary, logits within a small band): the k-th and (k+1)-th largest scaled logits are closer than the resolution of a 3-level count search (range / 16**3), so the
    kept set had k + 1 tokens in ~20% of the (row, k) cases, i.e. one token of mass 1 / k too much (device test: Kolmogorov deviation 0.044 at k = 20). The top-k search resolves one more level
    (``levels_k`` 4): exactly k tokens in every case here."""
    V_ = 129280
    wrong3 = wrong4 = 0
    for seed in range(6):
        row = torch.randn(V_, generator=torch.Generator().manual_seed(seed)) * 0.05
        for k in (5, 20, 50):
            wrong3 += int((emulate_keep_weights(row, 0.6, k, 1.0, J, LEVELS, 3) > 0).sum()) != k
            wrong4 += int((emulate_keep_weights(row, 0.6, k, 1.0, J, LEVELS, LEVELS_K) > 0).sum()) != k
    assert wrong4 == 0 and wrong3 >= 1, (wrong3, wrong4)


@pytest.mark.parametrize("scale", [0.3, 3.0, 12.0])
@pytest.mark.parametrize("cfg", [(1.0, 20, 1.0), (0.6, 20, 0.95), (1.0, 32, 0.9), (1.3, 5, 0.95), (1.0, 2, 1.0)])
def test_topk_path_keeps_exactly_the_exact_support_and_law(scale, cfg):
    """The candidate path (``forward_topk``) has no resolution band: kept set and law are the exact ones (random fp32 logits: no ties)."""
    torch.manual_seed(5)
    full = torch.randn(8 * 400) * scale
    T, tk, tp = cfg
    w = emulate_topk_weights(full, T, tk, tp, kcand=32, cols=8)
    ex = exact_probs(full, T, tk, tp)
    assert torch.equal(w > 0, ex > 0) or float(((w > 0) ^ (ex > 0)).sum()) == 0, (cfg, scale)
    assert float((w.double() / w.double().sum() - ex).abs().max()) < 1e-6
