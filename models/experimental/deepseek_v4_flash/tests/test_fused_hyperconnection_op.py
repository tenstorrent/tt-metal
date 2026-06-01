# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit test for the fused ``ttnn.experimental.deepseek.fused_hyperconnection`` op.

The op implements only the ``pre`` / ``post`` / ``comb`` / ``collapsed`` portion of
``DeepSeekV4HyperConnection.forward`` (hyperconnection.py lines 87-108) given the three
already-computed linear projections ``pre_w`` / ``post_w`` / ``comb_w``. This test is fully
self-contained: it builds random projections + biases, runs the op on device, and compares
against a torch reference of exactly that math (no HuggingFace / RMSNorm / matmul involved).
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc


PCC_THRESHOLD = 0.98


def _torch_reference(
    hidden_streams: torch.Tensor,
    pre_w: torch.Tensor,
    post_w: torch.Tensor,
    comb_w: torch.Tensor,
    pre_b: torch.Tensor,
    post_b: torch.Tensor,
    comb_b: torch.Tensor,
    hc: int,
    iters: int,
    pre_scale: float,
    post_scale: float,
    comb_scale: float,
    eps: float,
):
    """Mirror of hyperconnection.py lines 87-108, in torch float32."""
    b, s, _, d = hidden_streams.shape
    t = b * s

    pre = torch.sigmoid(pre_w * pre_scale + pre_b) + eps  # [1,1,T,H]
    post = 2.0 * torch.sigmoid(post_w * post_scale + post_b)  # [1,1,T,H]

    comb_logits = (comb_w * comb_scale + comb_b).reshape(1, t, hc, hc)
    comb = torch.softmax(comb_logits, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)  # column
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)  # row
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)  # column

    hs = hidden_streams.reshape(1, t, hc, d)
    pre_col = pre.reshape(1, t, hc, 1)
    collapsed = (hs * pre_col).sum(dim=-2, keepdim=True)  # [1,T,1,D]

    post = post.reshape(b, s, hc, 1)
    comb = comb.reshape(b, s, hc, hc)
    collapsed = collapsed.reshape(b, s, 1, d)
    return post, comb, collapsed


# Decode-only fused op: a single token (T == B * S == 1).
@pytest.mark.parametrize("seq_len", (1,))
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize("sinkhorn_iters", (1, 20))
def test_fused_hyperconnection_op(device, reset_seeds, batch_size, seq_len, sinkhorn_iters):
    hc = 4  # number of streams (hc_mult)
    d = 512  # hidden_size
    t = batch_size * seq_len
    eps = 1.0e-6
    pre_scale, post_scale, comb_scale = 1.0, 1.0, 1.0

    hidden_streams = torch.randn(batch_size, seq_len, hc, d, dtype=torch.float32)
    pre_w = torch.randn(1, 1, t, hc, dtype=torch.float32) * 0.5
    post_w = torch.randn(1, 1, t, hc, dtype=torch.float32) * 0.5
    comb_w = torch.randn(1, 1, t, hc * hc, dtype=torch.float32) * 0.5
    pre_b = torch.randn(1, 1, 1, hc, dtype=torch.float32) * 0.1
    post_b = torch.randn(1, 1, 1, hc, dtype=torch.float32) * 0.1
    comb_b = torch.randn(1, 1, 1, hc * hc, dtype=torch.float32) * 0.1

    ref_post, ref_comb, ref_collapsed = _torch_reference(
        hidden_streams,
        pre_w,
        post_w,
        comb_w,
        pre_b,
        post_b,
        comb_b,
        hc,
        sinkhorn_iters,
        pre_scale,
        post_scale,
        comb_scale,
        eps,
    )

    def to_tt(x):
        return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    post_tt, comb_tt, collapsed_tt = ttnn.experimental.deepseek.fused_hyperconnection(
        to_tt(hidden_streams),
        pre_w=to_tt(pre_w),
        post_w=to_tt(post_w),
        comb_w=to_tt(comb_w),
        pre_bias=to_tt(pre_b),
        post_bias=to_tt(post_b),
        comb_bias=to_tt(comb_b),
        num_streams=hc,
        sinkhorn_iters=sinkhorn_iters,
        pre_scale=pre_scale,
        post_scale=post_scale,
        comb_scale=comb_scale,
        eps=eps,
    )

    outputs = {
        "post": (ttnn.to_torch(post_tt).reshape(ref_post.shape).float(), ref_post),
        "comb": (ttnn.to_torch(comb_tt).reshape(ref_comb.shape).float(), ref_comb),
        "collapsed": (ttnn.to_torch(collapsed_tt).reshape(ref_collapsed.shape).float(), ref_collapsed),
    }

    all_pass = True
    msgs = []
    for name, (got, ref) in outputs.items():
        passing, pcc_message = comp_pcc(ref, got, pcc=PCC_THRESHOLD)
        logger.info(f"[fused_hyperconnection:{name}] {comp_allclose(ref, got)}")
        logger.info(f"[fused_hyperconnection:{name}] PCC: {pcc_message}")
        all_pass = all_pass and passing
        if not passing:
            msgs.append(f"{name}: {pcc_message}")

    assert all_pass, (
        f"fused_hyperconnection PCC < {PCC_THRESHOLD} "
        f"(batch={batch_size}, seq={seq_len}, iters={sinkhorn_iters}): {'; '.join(msgs)}"
    )
