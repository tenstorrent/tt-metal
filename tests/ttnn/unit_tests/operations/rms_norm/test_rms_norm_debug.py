# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic debug tests for rms_norm — DO NOT DELETE (documents bring-up / debugging).

Hand-calculable inputs:
- all ones: mean(x^2) = 1 -> rsqrt(1 + eps) ~= 1 -> out == x (== gamma when given).
- constant 2: mean(x^2) = 4 -> out = 2 * rsqrt(4) = 1.
- row-indexed constant r+1: out == 1 for every row (exercises per-row independence and the col broadcast).
"""

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm


def _run(device, torch_x, layout, dtype=ttnn.bfloat16, torch_gamma=None, gamma_layout=ttnn.ROW_MAJOR_LAYOUT):
    tt_x = ttnn.from_torch(torch_x, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    kwargs = {}
    if torch_gamma is not None:
        kwargs["gamma"] = ttnn.from_torch(
            torch_gamma, dtype=dtype, layout=gamma_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    out = rms_norm(tt_x, **kwargs)
    return ttnn.to_torch(out).float()


def _expected(x, gamma=None, eps=1e-6):
    xf = x.float()
    out = xf / torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + eps)
    if gamma is not None:
        out = out * gamma.float().reshape(-1)
    return out


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["TILE", "RM"])
def test_all_ones_single_tile(device, layout):
    x = torch.ones(1, 1, 32, 32, dtype=torch.bfloat16)
    out = _run(device, x, layout)
    assert torch.allclose(out, torch.ones_like(out), atol=0.05), f"max diff {(out - 1).abs().max()}"


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["TILE", "RM"])
@pytest.mark.parametrize("W", [64, 50, 512], ids=["W64", "W50", "W512"])
def test_row_constant(device, layout, W):
    # row r holds (r % 7) + 1 -> every output element == 1
    H = 64
    x = (
        (torch.arange(H).remainder(7) + 1)
        .float()
        .reshape(1, 1, H, 1)
        .expand(1, 1, H, W)
        .contiguous()
        .to(torch.bfloat16)
    )
    out = _run(device, x, layout)
    assert torch.allclose(out, torch.ones_like(out), atol=0.05), f"max diff {(out - 1).abs().max()}"


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["TILE", "RM"])
def test_gamma_column_index(device, layout):
    # x = 2 everywhere -> normalized 1 -> out == gamma[w] = w / 10
    W = 96
    x = torch.full((1, 1, 32, W), 2.0, dtype=torch.bfloat16)
    g = (torch.arange(W).float() / 10.0).reshape(1, 1, 1, W).to(torch.bfloat16)
    out = _run(device, x, layout, torch_gamma=g)
    exp = _expected(x, g)
    assert torch.allclose(out, exp, atol=0.1, rtol=0.02), f"max diff {(out - exp).abs().max()}"


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["TILE", "RM"])
def test_w_split_row_constant(device, layout):
    # R2 (W split + root combine): one tile-row, wide W. Row r holds r+1 -> out == 1.
    W = 16384
    H = 32
    x = (torch.arange(H) + 1).float().reshape(1, 1, H, 1).expand(1, 1, H, W).contiguous().to(torch.bfloat16)
    out = _run(device, x, layout)
    assert torch.allclose(out, torch.ones_like(out), atol=0.05), f"max diff {(out - 1).abs().max()}"
