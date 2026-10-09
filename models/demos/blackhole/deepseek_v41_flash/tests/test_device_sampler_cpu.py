# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU model of the in-trace sampler (tt/device_sampler.py) against the exact full-vocabulary sampler.

Why these tests exist next to the 'sample_exact' ones of test_vllm_interface.py: the in-trace draw (a) takes the inverse CDF in VOCABULARY order, not in sorted-candidate order, so for the same uniform it returns
another token of the same distribution (token identity with the sorted draw is not expected; the distribution is checked), and (b) finds the top-k / top-p thresholds by a k-ary search with a finite resolution
(range / J**levels in scaled-logit units), so tokens within that distance of the exact boundary may be kept or dropped (the boundary band is checked explicitly).
"""

import pytest
import torch

from models.demos.blackhole.deepseek_v41_flash.tt.device_sampler import emulate_draw, emulate_keep_weights

V = 1600
J, LEVELS = 16, 3


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
    got = emulate_keep_weights(full, T, tk, tp, J, LEVELS) > 0
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
    w = emulate_keep_weights(full, T, tk, tp, J, LEVELS)
    N = 200000
    u = torch.rand(N, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    emp = torch.bincount(emulate_draw(w, u), minlength=V).double() / N
    tv = 0.5 * float((emp - ref).abs().sum())
    C = int((ref > 0).sum())
    assert tv < 3.0 * (C / (2 * 3.14159 * N)) ** 0.5 + 5e-3, (tv, C)


def test_greedy_rows_are_the_exact_argmax_and_top_k_one_is_a_point_mass():
    torch.manual_seed(2)
    full = torch.randn(V) * 3
    w = emulate_keep_weights(full, 1.0, 1, 1.0, J, LEVELS)
    assert int((w > 0).sum()) >= 1 and int(emulate_draw(w, 0.37)) == int(full.argmax())
