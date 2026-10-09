# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit test for the fused ``ttnn.experimental.deepseek.fused_hyperconnection`` op.

The op implements the ``pre`` / ``post`` / ``comb`` / ``collapsed`` portion of
``DeepSeekV4HyperConnection.forward`` (hyperconnection.py) given the packed linear
projection ``fused_w`` (shape ``[1, 1, T, (2+H)*H]``). It splits ``fused_w`` into its
``pre_w`` / ``post_w`` / ``comb_w`` slices inside the ``fused_hyperconnection_pre_post``
device kernel; ``pre_w`` / ``post_w`` are consumed in-place and ``comb_w`` is returned
already laid out as the ``[1, T, H, H]`` comb matrix. This test is fully self-contained:
it builds a random ``fused_w`` + biases, runs the op on device, and compares against a
torch reference of exactly that math (no HuggingFace / RMSNorm / matmul involved).
``test_fused_hyperconnection_op_pre_mix`` also covers the optional ``pre_mix`` collapse weights
(DeepSeek V4.1) on both the single-user and the multi-token device programs.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.deepseek_v4_flash.tt.common import width_sharded_l1_config


PCC_THRESHOLD = 0.98


def _torch_reference(
    hidden_streams: torch.Tensor,
    fused_w: torch.Tensor,
    pre_b: torch.Tensor,
    post_b: torch.Tensor,
    comb_b: torch.Tensor,
    hc: int,
    iters: int,
    pre_scale: float,
    post_scale: float,
    comb_scale: float,
    eps: float,
    pre_mix: torch.Tensor | None = None,
):
    """Mirror of hyperconnection.py, in torch float32. ``pre_mix`` ``[B,S,1,H]``, when given,
    replaces ``pre`` in the collapse. Returns ``(post, comb, collapsed, pre)``."""
    b, s, _, d = hidden_streams.shape
    t = b * s

    pre_w, post_w, comb_w = torch.split(fused_w, [hc, hc, hc * hc], dim=-1)
    pre = torch.sigmoid(pre_w * pre_scale + pre_b) + eps  # [1,1,T,H]
    post = 2.0 * torch.sigmoid(post_w * post_scale + post_b)  # [1,1,T,H]

    comb_logits = (comb_w * comb_scale + comb_b).reshape(1, t, hc, hc)
    comb = torch.softmax(comb_logits, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)  # column
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)  # row
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)  # column

    hs = hidden_streams.reshape(1, t, hc, d)
    pre_col = (pre if pre_mix is None else pre_mix).reshape(1, t, hc, 1)
    collapsed = (hs * pre_col).sum(dim=-2, keepdim=True)  # [1,T,1,D]

    post = post.reshape(b, s, hc, 1)
    comb = comb.reshape(b, s, hc, hc)
    collapsed = collapsed.reshape(b, s, 1, d)
    pre = pre.reshape(b, s, 1, hc)
    return post, comb, collapsed, pre


# The T = B * S tokens are independent and are spread across the core grid, so batched decode
# (B > 1) and multi-token prefill (S > 1) run through the same two device ops. T = 40 crosses a
# tile row of fused_w (token t lives in row t % 32 of tile row t / 32), and T = 200 exceeds the
# core grid so each core loops over several tokens and restages fused_w as it crosses tile rows.
# T = 1 takes the single-user program, which needs sharded inputs; test_fused_hyperconnection_op_pre_mix
# covers it.
@pytest.mark.parametrize("batch_size, seq_len", ((8, 1), (1, 4), (5, 8), (25, 8)))
@pytest.mark.parametrize("sinkhorn_iters", (1, 20))
def test_fused_hyperconnection_op(device, reset_seeds, batch_size, seq_len, sinkhorn_iters):
    hc = 4  # number of streams (hc_mult)
    d = 512  # hidden_size
    t = batch_size * seq_len
    eps = 1.0e-6
    pre_scale, post_scale, comb_scale = 1.0, 1.0, 1.0

    hidden_streams = torch.randn(batch_size, seq_len, hc, d, dtype=torch.float32)
    fused_w = torch.randn(1, 1, t, (2 + hc) * hc, dtype=torch.float32) * 0.5
    pre_b = torch.randn(1, 1, 1, hc, dtype=torch.float32) * 0.1
    post_b = torch.randn(1, 1, 1, hc, dtype=torch.float32) * 0.1
    comb_b = torch.randn(1, 1, 1, hc * hc, dtype=torch.float32) * 0.1

    ref_post, ref_comb, ref_collapsed, _ = _torch_reference(
        hidden_streams,
        fused_w,
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
        fused_w=to_tt(fused_w),
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

    _check_outputs(
        (post_tt, comb_tt, collapsed_tt),
        (ref_post, ref_comb, ref_collapsed),
        f"batch={batch_size}, seq={seq_len}, iters={sinkhorn_iters}",
    )


def _check_outputs(got_tt, refs, context: str) -> None:
    """PCC-compare the op's ``(post, comb, collapsed[, pre])`` against the torch reference."""
    all_pass = True
    msgs = []
    names = ("post", "comb", "collapsed", "pre")[: len(refs)]
    for name, got_t, ref in zip(names, got_tt, refs, strict=True):
        got = ttnn.to_torch(got_t).reshape(ref.shape).float()
        passing, pcc_message = comp_pcc(ref, got, pcc=PCC_THRESHOLD)
        logger.info(f"[fused_hyperconnection:{name}] {comp_allclose(ref, got)}")
        logger.info(f"[fused_hyperconnection:{name}] PCC: {pcc_message}")
        all_pass = all_pass and passing
        if not passing:
            msgs.append(f"{name}: {pcc_message}")

    assert all_pass, f"fused_hyperconnection PCC < {PCC_THRESHOLD} ({context}): {'; '.join(msgs)}"


# T = 1 takes the single-user device program, which wants the layouts DeepSeekV4HyperConnection
# builds: a ROW_MAJOR fused_w row on core (0, 0) and the streams width-sharded over 8 cores. T > 1
# takes the multi-token program on interleaved inputs. Each runs with the op's own pre (V4) and with
# a pre_mix from elsewhere (V4.1); post / comb must not depend on which, and given pre_mix the op also
# returns the pre it computed.
@pytest.mark.parametrize("batch_size, seq_len", ((1, 1), (8, 1), (5, 8)))
@pytest.mark.parametrize("use_pre_mix", (False, True), ids=("own_pre", "pre_mix"))
def test_fused_hyperconnection_op_pre_mix(device, reset_seeds, batch_size, seq_len, use_pre_mix):
    hc = 4
    d = 512
    t = batch_size * seq_len
    sinkhorn_iters = 20
    eps = 1.0e-6
    pre_scale, post_scale, comb_scale = 1.0, 1.0, 1.0

    hidden_streams = torch.randn(batch_size, seq_len, hc, d, dtype=torch.float32)
    # The fn matmul's N is padded to a tile; the op reads only the first (2+H)*H columns.
    fused_w = torch.zeros(1, 1, t, ttnn.TILE_SIZE, dtype=torch.float32)
    fused_w[..., : (2 + hc) * hc] = torch.randn(1, 1, t, (2 + hc) * hc) * 0.5
    pre_b = torch.randn(1, 1, 1, hc, dtype=torch.float32) * 0.1
    post_b = torch.randn(1, 1, 1, hc, dtype=torch.float32) * 0.1
    comb_b = torch.randn(1, 1, 1, hc * hc, dtype=torch.float32) * 0.1
    pre_mix = torch.rand(batch_size, seq_len, 1, hc, dtype=torch.float32) if use_pre_mix else None

    refs = _torch_reference(
        hidden_streams,
        fused_w[..., : (2 + hc) * hc],
        pre_b,
        post_b,
        comb_b,
        hc,
        sinkhorn_iters,
        pre_scale,
        post_scale,
        comb_scale,
        eps,
        pre_mix=pre_mix,
    )

    def to_tt(x, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=memory_config)

    if t == 1:
        fused_w_tt = to_tt(
            fused_w,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
                    [1, ttnn.TILE_SIZE],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        hidden_tt = to_tt(hidden_streams, memory_config=width_sharded_l1_config(hc, d, device, num_cores=8))
    else:
        fused_w_tt = to_tt(fused_w)
        hidden_tt = to_tt(hidden_streams)

    outputs = ttnn.experimental.deepseek.fused_hyperconnection(
        hidden_tt,
        fused_w=fused_w_tt,
        pre_bias=to_tt(pre_b),
        post_bias=to_tt(post_b),
        comb_bias=to_tt(comb_b),
        num_streams=hc,
        sinkhorn_iters=sinkhorn_iters,
        pre_scale=pre_scale,
        post_scale=post_scale,
        comb_scale=comb_scale,
        eps=eps,
        pre_mix=None if pre_mix is None else to_tt(pre_mix),
    )

    num_outputs = 4 if use_pre_mix else 3
    assert len(outputs) == num_outputs, f"expected {num_outputs} outputs, got {len(outputs)}"
    _check_outputs(outputs, refs[:num_outputs], f"batch={batch_size}, seq={seq_len}, pre_mix={use_pre_mix}")
