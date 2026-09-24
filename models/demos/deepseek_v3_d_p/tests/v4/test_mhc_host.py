# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Algebra lock for the mHC decomposition (tt/v4/mhc_math.py) against the reference DeepseekV4HyperConnection /
HyperHead / decoder-layer mix, device-free, fp32. If this passes, the ttnn module only has to reproduce these ops."""

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import deepseek_v4_flash_hf_config
from models.demos.deepseek_v3_d_p.tt.v4 import mhc_math as M


def _cfg(hidden=512):
    cfg = deepseek_v4_flash_hf_config(num_hidden_layers=4)
    cfg.hidden_size = hidden
    return cfg


def _ref_hc(cfg, seed):
    torch.manual_seed(seed)
    ref = DeepseekV4HyperConnection(cfg).eval()
    with torch.no_grad():
        ref.fn.normal_(0.0, 0.02)
        ref.base.normal_(0.0, 0.3)
        ref.scale.copy_(torch.tensor([1.1, 0.9, 1.3]))
    return ref


@pytest.mark.parametrize("tp", [1, 4])
@pytest.mark.parametrize("hidden,seq", [(512, 64), (4096, 40)])
def test_pre_site_matches_reference(tp, hidden, seq):
    cfg = _cfg(hidden)
    ref = _ref_hc(cfg, seed=7)
    torch.manual_seed(11)
    h = torch.randn(1, seq, M.HC, hidden) * 2.0  # [B, S, H, D] as the reference consumes it
    post_ref, comb_ref, collapsed_ref = ref(h)  # post/comb fp32, collapsed in h's dtype (fp32 here)

    W = M.prep_hyper_connection(ref.fn.detach(), ref.base.detach(), ref.scale.detach(), hidden)
    W["rms_eps"] = cfg.rms_norm_eps
    streams = [h[0, :, i, :] for i in range(M.HC)]
    pre, post, comb, collapsed = M.pre_site(streams, W, eps=cfg.hc_eps, sinkhorn_iters=cfg.hc_sinkhorn_iters, tp=tp)

    torch.testing.assert_close(post[:, M.POST0 : M.POST0 + M.HC], post_ref[0], rtol=1e-5, atol=1e-5)
    comb_4x4 = comb[:, M.COMB0 : M.COMB0 + M.HC * M.HC].reshape(seq, M.HC, M.HC)
    torch.testing.assert_close(comb_4x4, comb_ref[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(collapsed, collapsed_ref[0], rtol=1e-5, atol=1e-4)
    # padded columns of comb stay exactly zero. Sinkhorn ends on a COLUMN normalisation, so column sums are
    # 1 - eps/sum (exact to ~1e-5); row sums are only approximately 1 after 20 iterations (the reference's own
    # residual is a few percent with these logits) -- equality with the reference above is the real check.
    assert torch.all(comb[:, : M.COMB0] == 0) and torch.all(comb[:, M.COMB0 + 16 : M.ONE_COL] == 0)
    assert torch.all(comb[:, M.ONE_COL] == 1) and torch.all(comb[:, M.ONE_COL + 1 :] == 0)
    assert (comb_4x4.sum(-2) - 1).abs().max() < 1e-4
    assert (comb_4x4.sum(-1) - 1).abs().max() < 0.1


def test_mix_site_matches_decoder_layer_formula():
    cfg = _cfg(512)
    ref = _ref_hc(cfg, seed=3)
    torch.manual_seed(5)
    seq = 48
    h = torch.randn(1, seq, M.HC, 512)
    y = torch.randn(1, seq, 512)
    post_ref, comb_ref, _ = ref(h)
    # the decoder layer (REF :1143-1149): post.unsqueeze(-1) * y.unsqueeze(-2) + comb^T @ h
    out_ref = post_ref.unsqueeze(-1) * y.unsqueeze(-2) + torch.matmul(comb_ref.transpose(-1, -2), h)

    W = M.prep_hyper_connection(ref.fn.detach(), ref.base.detach(), ref.scale.detach(), 512)
    streams = [h[0, :, i, :] for i in range(M.HC)]
    pre, post, comb, _ = M.pre_site(streams, W, eps=cfg.hc_eps, sinkhorn_iters=cfg.hc_sinkhorn_iters)
    out = M.mix_site(streams, y[0], post, comb)
    for k in range(M.HC):
        torch.testing.assert_close(out[k], out_ref[0, :, k, :], rtol=1e-5, atol=1e-4)


def test_hyper_head_matches_reference():
    cfg = _cfg(512)
    torch.manual_seed(9)
    head = DeepseekV4HyperHead(cfg).eval()
    with torch.no_grad():
        head.hc_fn.normal_(0.0, 0.02)
        head.hc_base.normal_(0.0, 0.3)
        head.hc_scale.fill_(0.8)
    h = torch.randn(1, 32, M.HC, 512)
    out_ref = head(h)
    W = M.prep_hyper_head(head.hc_fn.detach(), head.hc_base.detach(), head.hc_scale.detach(), 512)
    W["rms_eps"] = cfg.rms_norm_eps
    out = M.hyper_head([h[0, :, i, :] for i in range(M.HC)], W, eps=cfg.hc_eps, tp=2)
    torch.testing.assert_close(out, out_ref[0], rtol=1e-5, atol=1e-4)


def test_constant_matrices_sum_rows_and_columns():
    X = torch.zeros(3, M.ROW)
    X[:, M.COMB0 : M.COMB0 + 16] = torch.arange(1.0, 17.0).repeat(3, 1)
    rows = (X @ M.row_sum_matrix())[0, M.COMB0 : M.COMB0 + 16].reshape(4, 4)
    cols = (X @ M.col_sum_matrix())[0, M.COMB0 : M.COMB0 + 16].reshape(4, 4)
    m = torch.arange(1.0, 17.0).reshape(4, 4)
    torch.testing.assert_close(rows, m.sum(-1, keepdim=True).expand(4, 4))
    torch.testing.assert_close(cols, m.sum(-2, keepdim=True).expand(4, 4))
    assert (X @ M.row_sum_matrix())[:, : M.COMB0].abs().sum() == 0
    # the augmented matrices: eps on the comb columns via the ones column, exactly 1 elsewhere
    Y = torch.zeros(1, M.ROW)
    Y[0, M.ONE_COL] = 1.0
    ra = Y @ M.row_sum_matrix_aug(1e-6)
    assert (
        torch.all(ra[0, M.COMB0 : M.COMB0 + 16] == 1e-6) and torch.all(ra[0, : M.COMB0] == 1) and ra[0, M.ONE_COL] == 1
    )
