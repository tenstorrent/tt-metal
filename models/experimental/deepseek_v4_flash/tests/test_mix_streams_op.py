# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit test for the fused ``ttnn.experimental.deepseek.mix_streams`` op.

The op implements ``DeepSeekV4DecoderLayer._mix`` (decoder_layer.py)::

    new_streams = post[..., None] * sublayer_out[..., None, :] + comb.T @ streams

as a single device kernel. This test is self-contained: it builds random inputs,
runs the op on device and compares against a torch reference of exactly that math.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc

PCC_THRESHOLD = 0.999


def _torch_reference(
    post: torch.Tensor, comb: torch.Tensor, sublayer_out: torch.Tensor, streams: torch.Tensor
) -> torch.Tensor:
    placement = post * sublayer_out  # [B,S,hc,1] * [B,S,1,D] -> [B,S,hc,D]
    mixed = torch.matmul(comb.transpose(-1, -2), streams)
    return placement + mixed


@pytest.mark.parametrize("batch_size, seq_len", ((1, 1), (2, 1), (1, 4)))
@pytest.mark.parametrize("hc", (4, 32))
@pytest.mark.parametrize("d", (512, 4096))
def test_mix_streams_op(device, reset_seeds, batch_size, seq_len, hc, d):
    post = torch.rand(batch_size, seq_len, hc, 1, dtype=torch.float32) * 2.0
    comb = torch.softmax(torch.randn(batch_size, seq_len, hc, hc, dtype=torch.float32), dim=-1)
    sublayer_out = torch.randn(batch_size, seq_len, 1, d, dtype=torch.float32)
    streams = torch.randn(batch_size, seq_len, hc, d, dtype=torch.float32)

    reference = _torch_reference(post, comb, sublayer_out, streams)

    def to_tt(x):
        return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out_tt = ttnn.experimental.deepseek.mix_streams(to_tt(post), to_tt(comb), to_tt(sublayer_out), to_tt(streams))
    got = ttnn.to_torch(out_tt).float()

    assert tuple(out_tt.shape) == tuple(reference.shape)
    passing, pcc_message = comp_pcc(reference, got, pcc=PCC_THRESHOLD)
    logger.info(f"[mix_streams] {comp_allclose(reference, got)}")
    logger.info(f"[mix_streams] PCC: {pcc_message}")
    assert passing, f"mix_streams PCC < {PCC_THRESHOLD} (B={batch_size}, S={seq_len}, hc={hc}, D={d}): {pcc_message}"
