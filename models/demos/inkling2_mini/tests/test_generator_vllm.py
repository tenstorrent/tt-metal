# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""The vLLM adapter's own sampler (tt/generator_vllm.py) keeps the tokens vLLM's sampler keeps. Host only."""

import pytest
import torch

from models.demos.inkling2_mini.tt.generator_vllm import Inkling2ForCausalLM

V = 200058


def reference(lg, t, top_k, top_p):
    """vLLM's rule (v1/sample/ops/topk_topp_sampler.py) with a full sort: {token id: probability}."""
    x = lg.double() / t
    if 0 < top_k < x.numel():
        x = x.masked_fill(x < torch.topk(x, top_k).values[-1], float("-inf"))
    if top_p < 1.0:
        srt, order = x.sort(descending=False)
        mask = srt.softmax(-1).cumsum(-1) <= 1 - top_p
        mask[-1] = False
        x = x.scatter(0, order, srt.masked_fill(mask, float("-inf")))
    p = x.softmax(-1)
    return {int(i): float(p[i]) for i in p.nonzero().reshape(-1)}


@pytest.mark.parametrize("scale", [0.02, 1.0, 6.0])  # flat (the nucleus is wider than NUCLEUS_K), usual, peaked
@pytest.mark.parametrize("top_k, top_p", [(0, 1.0), (40, 1.0), (0, 0.95), (40, 0.95), (1, 1.0), (V, 0.5)])
def test_candidates_match_full_sort(scale, top_k, top_p):
    lg = torch.randn(V, generator=torch.Generator().manual_seed(7)) * scale
    for t in (0.7, 1.3):
        p, idx = Inkling2ForCausalLM._candidates(lg, t, top_k, top_p)
        ids = range(V) if idx is None else idx.tolist()
        got = dict(zip(ids, (p.double() / p.double().sum()).tolist()))
        ref = reference(lg, t, top_k, top_p)
        # a token whose cumulative mass sits within fp32 rounding of the top_p cut may fall on either side
        edge = set(got) ^ set(ref)
        assert len(edge) <= 1 and all(max(got.get(i, 0), ref.get(i, 0)) < 1e-3 for i in edge), (len(got), len(ref))
        assert all(abs(got[i] - ref[i]) < 1e-4 for i in set(got) & set(ref))


def test_seeded_draws_repeat():
    m = object.__new__(Inkling2ForCausalLM)
    m._rng = torch.Generator()
    lg = torch.randn(V, generator=torch.Generator().manual_seed(3)) * 3
    runs = []
    for _ in range(2):
        m._rng.manual_seed(11)
        runs.append([m._sample(lg, 0.8, 40, 0.95) for _ in range(50)])
    assert runs[0] == runs[1] and len(set(runs[0])) > 1
