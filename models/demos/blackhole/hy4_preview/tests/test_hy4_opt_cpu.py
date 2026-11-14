# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the layer-other fusions: rope column permutation (half-split -> adjacent pair), fused-projection weight splits, Opt flag parsing."""
import torch

from models.demos.blackhole.hy4_preview.tt.hy4_attn_opt import Opt, half_to_interleave_perm
from models.demos.blackhole.hy4_preview.tt.hy4_rope import apply_rope_host, rope_tables


def _rope_span(x, pos, first, interleave):
    """rope on a 64-wide span of x [n,128]; fp32."""
    sl = slice(0, 64) if first else slice(64, 128)
    c, s = rope_tables(pos, 64, 1e7, interleave)
    y = x.clone()
    y[:, sl] = apply_rope_host(x[:, sl], c, s, interleave)
    return y


def test_permutation_reproduces_half_split_rope_exactly():
    torch.manual_seed(0)
    n = 5
    pos = torch.arange(1000, 1000 + n)
    for first in (True, False):
        p = half_to_interleave_perm(128, 64, first)
        assert sorted(p.tolist()) == list(range(128))
        x = torch.randn(n, 128, dtype=torch.float64).float()
        ref = _rope_span(x, pos, first, interleave=False)  # original half-split rope on the original basis
        got_perm = _rope_span(
            x[:, p], pos, first, interleave=True
        )  # permuted columns + adjacent-pair rope (what rotary_embedding_llama computes)
        assert torch.equal(got_perm, ref[:, p]) or torch.allclose(got_perm, ref[:, p], atol=1e-6)


def test_permuted_indexer_weights_give_identical_scores():
    """k = LN(h @ Wk) * w + b then rope; q = qa @ Wq (32 heads x 128) then rope; score = sum_h relu(q.k) * wp. Permuting Wk / LN params / Wq columns per head and switching to the
    adjacent-pair rope leaves every score unchanged (dot over 128 is permutation invariant; LN statistics too)."""
    torch.manual_seed(1)
    D, R, nh, dh = 256, 128, 4, 128
    h, qa = torch.randn(3, D), torch.randn(3, R)
    wk, wq = torch.randn(D, dh) * 0.05, torch.randn(R, nh * dh) * 0.05
    lw, lb, wp = torch.randn(dh), torch.randn(dh), torch.rand(3, nh)
    pos = torch.arange(500, 503)
    keys = torch.randn(7, D)  # earlier keys come from other positions: same pipeline

    def run(wk, lw, lb, wq, interleave, first):
        def kf(x, pp):
            k = torch.nn.functional.layer_norm(x @ wk, (dh,), lw, lb, 1e-6)
            return _rope_span(k, pp, first, interleave)

        k_all = torch.cat([kf(keys, torch.arange(7)), kf(h, pos)])
        q = _rope_span((qa @ wq).reshape(3 * nh, dh), pos.repeat_interleave(nh), first, interleave).reshape(3, nh, dh)
        sc = torch.relu(torch.einsum("qhd,td->qht", q, k_all)) * wp[..., None]
        return sc.sum(1)

    for first in (True, False):
        p = half_to_interleave_perm(dh, 64, first)
        wq_p = wq.reshape(R, nh, dh)[:, :, p].reshape(R, nh * dh)
        a = run(wk, lw, lb, wq, False, first)
        b = run(wk[:, p], lw[p], lb[p], wq_p, True, first)
        assert torch.allclose(a, b, atol=2e-4, rtol=1e-4), (a - b).abs().max()


def test_fused_projection_column_splits():
    torch.manual_seed(2)
    D, cols = 128, 8
    qa_w, kv_w, g_w = torch.randn(D, 256 * cols), torch.randn(D, 576), torch.randn(D, 2048 * cols)
    wk, wp = torch.randn(D, 128), torch.randn(D, 32)
    h = torch.randn(2, D)
    parts = [
        torch.cat([qa_w[:, c * 256 : (c + 1) * 256], kv_w, g_w[:, c * 2048 : (c + 1) * 2048], wk, wp], -1)
        for c in range(cols)
    ]
    wf = torch.cat(parts, -1)
    nloc = wf.shape[-1] // cols
    assert nloc == 3040
    offs = [0, 256, 832, 2880, 3008, 3040]
    for c in range(cols):
        loc = h @ wf[:, c * nloc : (c + 1) * nloc]
        ref = [h @ qa_w[:, c * 256 : (c + 1) * 256], h @ kv_w, h @ g_w[:, c * 2048 : (c + 1) * 2048], h @ wk, h @ wp]
        for i, r in enumerate(ref):
            assert torch.allclose(loc[:, offs[i] : offs[i + 1]], r, atol=1e-4)


def test_opt_flags():
    assert Opt.parse("").tag() == "base"
    assert Opt.parse("all") == Opt.all_on()
    assert Opt.parse("l1,rope1").tag() == "l1,rope1"
