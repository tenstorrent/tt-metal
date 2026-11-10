# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the stream-major hyper-connection ops of the V4.1 prefill.

* ``ttnn.experimental.deepseek.stream_mix`` over streams ``[1, hc, T, D]``: the collapse with ``pre`` weights
  ``[1, T, 1, hc]`` and the residual mix with ``comb`` ``[1, T, hc, hc]``, ``post`` ``[1, T, hc, 1]`` and the
  sublayer output ``[1, 1, T, D]``.
* ``ttnn.experimental.deepseek.fused_hyperconnection_weights``: ``fused_hyperconnection`` without the collapse,
  returning ``(post, comb, pre)``; chained into ``stream_mix`` it gives the op's ``collapsed``.

Self-contained: random inputs against a torch reference of the same math.

    pytest models/experimental/deepseek_v41_flash/tests/prefill/test_stream_mix_op.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.deepseek_v4_flash.tests.decode.test_fused_hyperconnection_op import _torch_reference

PCC = 0.999
HC = 4
# One partial row block (40), whole row blocks (128), and a prefill chunk (1024).
SEQ_LENS = (40, 128, 1024)


def _to_tt(x: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _check(expected: torch.Tensor, got_tt: ttnn.Tensor, what: str) -> None:
    got = ttnn.to_torch(got_tt).float()
    assert tuple(got.shape) == tuple(expected.shape), f"{what}: shape {tuple(got.shape)} != {tuple(expected.shape)}"
    passing, pcc = comp_pcc(expected, got, PCC)
    logger.info(f"{what}: {comp_allclose(expected, got)}, PCC {pcc}")
    assert passing, f"{what}: PCC {pcc} < {PCC}"


@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("d", (512, 5120))
def test_stream_mix_collapse(device, reset_seeds, seq_len, d):
    streams = torch.randn(1, HC, seq_len, d).bfloat16().float()
    pre = torch.rand(1, seq_len, 1, HC).bfloat16().float()
    # out[t] = sum_j pre[t, j] * streams[j, t]
    expected = torch.einsum("tj,jtd->td", pre[0, :, 0], streams[0]).reshape(1, 1, seq_len, d)

    got = ttnn.experimental.deepseek.stream_mix(_to_tt(streams, device), _to_tt(pre, device))
    _check(expected, got, f"collapse T={seq_len} D={d}")


@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("d", (512, 5120))
def test_stream_mix_residual(device, reset_seeds, seq_len, d):
    streams = torch.randn(1, HC, seq_len, d).bfloat16().float()
    comb = torch.rand(1, seq_len, HC, HC).bfloat16().float()
    post = (2 * torch.rand(1, seq_len, HC, 1)).bfloat16().float()
    sub = torch.randn(1, 1, seq_len, d).bfloat16().float()
    # out[i, t] = sum_j comb[t, j, i] * streams[j, t] + post[t, i] * sub[t]
    mixed = torch.einsum("tji,jtd->itd", comb[0], streams[0])
    placed = post[0, :, :, 0].transpose(0, 1).unsqueeze(-1) * sub[0]  # [hc, T, D]
    expected = (mixed + placed).unsqueeze(0)

    got = ttnn.experimental.deepseek.stream_mix(
        _to_tt(streams, device), _to_tt(comb, device), _to_tt(post, device), _to_tt(sub, device)
    )
    _check(expected, got, f"residual mix T={seq_len} D={d}")


@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("sinkhorn_iters", (1, 20))
def test_fused_hyperconnection_weights(device, reset_seeds, seq_len, sinkhorn_iters):
    d = 512
    eps = 1.0e-6
    pre_scale, post_scale, comb_scale = 1.0, 1.0, 1.0
    streams = torch.randn(1, seq_len, HC, d)  # token-major, as the reference takes it
    # The fn matmul's N is padded to a tile; the op reads only the first (2+H)*H columns.
    fused_w = torch.zeros(1, 1, seq_len, ttnn.TILE_SIZE)
    fused_w[..., : (2 + HC) * HC] = torch.randn(1, 1, seq_len, (2 + HC) * HC) * 0.5
    pre_b = torch.randn(1, 1, 1, HC) * 0.1
    post_b = torch.randn(1, 1, 1, HC) * 0.1
    comb_b = torch.randn(1, 1, 1, HC * HC) * 0.1
    ref_post, ref_comb, ref_collapsed, ref_pre = _torch_reference(
        streams,
        fused_w[..., : (2 + HC) * HC],
        pre_b,
        post_b,
        comb_b,
        HC,
        sinkhorn_iters,
        pre_scale,
        post_scale,
        comb_scale,
        eps,
    )

    post, comb, pre = ttnn.experimental.deepseek.fused_hyperconnection_weights(
        fused_w=_to_tt(fused_w, device),
        pre_bias=_to_tt(pre_b, device),
        post_bias=_to_tt(post_b, device),
        comb_bias=_to_tt(comb_b, device),
        num_streams=HC,
        sinkhorn_iters=sinkhorn_iters,
        pre_scale=pre_scale,
        post_scale=post_scale,
        comb_scale=comb_scale,
        eps=eps,
    )
    context = f"T={seq_len} iters={sinkhorn_iters}"
    _check(ref_post, post, f"post {context}")
    _check(ref_comb, comb, f"comb {context}")
    _check(ref_pre, pre, f"pre {context}")

    # The returned pre is zero-padded, so it collapses the stream-major streams directly.
    stream_major = _to_tt(streams.permute(0, 2, 1, 3).contiguous(), device)
    collapsed = ttnn.experimental.deepseek.stream_mix(stream_major, pre)
    _check(ref_collapsed.reshape(1, 1, seq_len, d), collapsed, f"collapsed {context}")
