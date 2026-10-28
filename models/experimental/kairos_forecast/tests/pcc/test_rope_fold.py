# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""D3 R1-1: the folded q @ R / k @ R weight columns are, as stored on the device, bitwise the permuted and negated
q / k columns, in bf16 and bfp8 (order-only, so RoPE outputs stay bitwise equal)."""

import pytest
import torch

import ttnn
from models.experimental.kairos_forecast.reference import dataflow as DF
from models.experimental.kairos_forecast.tt.common import upload_linear
from models.experimental.kairos_forecast.tt.model import DK, H, _half_rows, rotated_rows


def _rotate_half_cols(t):
    """Stored (in, out) weight of x -> the stored weight of x @ R: per head, col i = -col i + 32, col i + 32 = col i."""
    h = t.reshape(t.shape[0], H, DK)
    return torch.cat([-h[..., DK // 2 :], h[..., : DK // 2]], -1).reshape(t.shape)


@pytest.mark.parametrize("dtype", ["bfloat16", "bfloat8_b"])
def test_rotated_rows_bitwise(device, dtype):
    W = DF.load_weights()
    for name in ("encoder.block.0.layer.0.SelfAttention.q", "decoder.block.5.layer.1.EncDecAttention.k"):
        w = _half_rows(W[f"{name}.weight"])
        dt = getattr(ttnn, dtype)
        plain = ttnn.to_torch(upload_linear(device, w, dt)).reshape(-1, H * DK)
        folded = ttnn.to_torch(upload_linear(device, rotated_rows(w), dt)).reshape(-1, H * DK)
        expect = _rotate_half_cols(plain)
        bits = lambda t: t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32}[t.dtype])  # noqa: E731
        assert folded.dtype == expect.dtype and torch.equal(bits(folded), bits(expect)), (name, dtype)
