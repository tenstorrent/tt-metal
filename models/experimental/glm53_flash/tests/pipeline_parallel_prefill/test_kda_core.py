# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The composite chunked KDA delta rule (``kda_core``) against HF ``chunk_kimi_delta_attention``."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.experimental.glm53_flash.tt.pipeline_parallel_prefill.common import TPContext
from models.experimental.glm53_flash.tt.pipeline_parallel_prefill.kda import CHUNK, _Consts, kda_core

hf = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")

H, D = 16, 128


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


def _l2(x):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


@pytest.mark.parametrize("seq", [32, 96, 256, 2048])
@pytest.mark.parametrize("regime", ["typical", "lower_bound", "correlated"])
def test_kda_core_matches_hf(stage, seq, regime):
    """typical: random keys, g ~ -5 sigmoid(N(-4, 1)); lower_bound: every channel at the -5 bound; correlated: keys
    sharing one direction with beta ~ 1 and weak decay, so |A| ~ 1 (real KDA heads do this; it is where a Neumann-series
    inverse of I + A cancels catastrophically)."""
    torch.manual_seed(seq)
    ctx = TPContext(stage)
    q, k, v = (torch.randn(1, seq, H, D) for _ in range(3))
    g = -5 * torch.sigmoid(torch.randn(1, seq, H, D) - 4)
    beta = torch.sigmoid(torch.randn(1, seq, H))
    if regime == "lower_bound":
        g = torch.full((1, seq, H, D), -5.0)
    elif regime == "correlated":
        k = torch.randn(1, 1, H, D) + 0.3 * torch.randn(1, seq, H, D)
        g = -5 * torch.sigmoid(torch.randn(1, seq, H, D) - 8)
        beta = torch.sigmoid(torch.randn(1, seq, H) + 3)
    s0 = torch.randn(1, H, D, D) * 0.1
    ref, ref_s = hf.chunk_kimi_delta_attention(
        q, k, v, g=g, beta=beta, initial_state=s0, output_final_state=True, use_qk_l2norm_in_kernel=True
    )

    n = seq // CHUNK
    heads = lambda x: x[0].transpose(0, 1).reshape(H, n, CHUNK, -1)
    up = lambda x: ctx.upload(x, ttnn.float32)
    o, s = kda_core(
        ctx,
        _Consts(ctx, H),
        up(heads(_l2(q) * D**-0.5)),
        up(heads(_l2(k))),
        up(heads(v)),
        up(heads(g)),
        up(heads(beta[..., None].expand(-1, -1, -1, D))),
        up(s0[0]),
    )
    got = ctx.download(o).float().reshape(H, seq, D).transpose(0, 1)[None]
    got_s = ctx.download(s).float()[None]
    p_o, p_s = pcc(got, ref), pcc(got_s, ref_s)
    err = (got - ref).abs().max().item()
    print(f"\nseq {seq} {regime}: out PCC {p_o:.6f} max|err| {err:.3e}; state PCC {p_s:.6f}")
    assert torch.isfinite(got).all() and torch.isfinite(got_s).all()
    assert p_o > 0.9999 and p_s > 0.9999
