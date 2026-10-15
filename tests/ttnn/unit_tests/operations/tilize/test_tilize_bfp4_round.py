# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 5a: bfloat4_b output is rounded to nearest (reader pre-pass), matching the host
converter (round-to-nearest per 16-element shared-exponent group) instead of packer truncation."""

import pytest
import torch
import ttnn

from ttnn.operations.tilize.tilize import tilize


def _host_bfp4(x_padded):
    """Host round-to-nearest bfloat4_b conversion of an already tile-aligned tensor."""
    return ttnn.to_torch(ttnn.from_torch(x_padded, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()


def _make(dtype, shape):
    x = torch.randn(shape)
    if dtype == ttnn.fp8_e4m3:
        return x.to(torch.float8_e4m3fn).to(torch.float32)
    if dtype == ttnn.bfloat16:
        return x.bfloat16()
    return x


@pytest.mark.parametrize(
    "shape, pad_value",
    [
        ((1, 1, 64, 128), None),  # segment path, aligned
        ((1, 1, 3200, 32), None),  # coalesced bank reads path
        ((1, 1, 50, 50), -7.0),  # padded W/H tails, negative fill
        ((1, 1, 50, 64), 3.0),  # padded H tail
    ],
    ids=str,
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.fp8_e4m3], ids=str)
def test_tilize_bfp4_matches_host_rounding(device, shape, pad_value, dtype):
    torch.manual_seed(0)
    x = _make(dtype, shape)
    t = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    kwargs = {"dtype": ttnn.bfloat4_b, "bfp4_round_to_nearest": True}
    padded = list(shape)
    if pad_value is not None:
        kwargs["pad_value"] = pad_value
        padded[-2] = (shape[-2] + 31) // 32 * 32
        padded[-1] = (shape[-1] + 31) // 32 * 32
    out = tilize(t, **kwargs)
    y = out.cpu().to_torch_with_padded_shape().float() if pad_value is not None else ttnn.to_torch(out).float()

    xp = x.float()
    if pad_value is not None:
        xp = torch.nn.functional.pad(xp, (0, padded[-1] - shape[-1], 0, padded[-2] - shape[-2]), value=pad_value)
    ref = _host_bfp4(xp)
    # Device round-to-nearest-even + packer truncation == host round-to-nearest-even, bit for bit.
    mismatch = (y != ref).float().mean().item()
    assert mismatch == 0.0, f"mismatch fraction vs host RTN: {mismatch}"
    logical = y[..., : shape[-2], : shape[-1]]
    pcc = torch.corrcoef(torch.stack([logical.flatten().double(), x.float().flatten().double()]))[0, 1].item()
    assert pcc > 0.985, pcc
