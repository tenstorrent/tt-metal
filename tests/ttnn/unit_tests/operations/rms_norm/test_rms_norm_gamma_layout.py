# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gamma-layout matrix for rms_norm (Refinement 2 — tiled-gamma support).

Covers the new SUPPORTED["gamma_layout"] = TILE value alongside the phase-1 RM
gamma, across:
  * gamma_layout   — ROW_MAJOR (phase-1 contract) and TILE (Refinement 2)
  * input layout   — TILE and ROW_MAJOR (gamma layout is an INDEPENDENT knob;
                     RM input + TILE gamma is a valid TARGET cell)
  * dtype          — bf16 / f32, and the mixed-precision LLM pattern (bf16
                     activations + f32 TILE gamma), plus bf8b gamma (block-float
                     implies TILE gamma; tile-aligned only)
  * alignment      — tile-aligned + W/H/both non-aligned (native masked reduce)

The RM-gamma column is the non-regression guard for the phase-1 path; the TILE
column is the new native tiled reader (no host-side gamma transform). The op is
called directly with gamma already in the requested layout — no ttnn.to_layout /
ttnn.tilize wrapper (native reader-path change).

Device comes from this directory's conftest (module-scoped) — never opened here.
"""

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm


PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    if torch.allclose(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def rms_norm_reference(x, gamma, epsilon):
    x = x.to(torch.float32)
    out = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out


def _run(device, shape, input_layout, dtype, gamma_layout, gamma_dtype, epsilon=1e-6):
    torch.manual_seed(42)
    W = shape[-1]

    torch_input = torch.randn(shape, dtype=torch.float32)
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=input_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    torch_gamma = torch.randn(W, dtype=torch.float32)
    ttnn_gamma = ttnn.from_torch(
        torch_gamma.reshape(1, 1, 1, W),
        dtype=gamma_dtype,
        layout=gamma_layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = rms_norm(ttnn_input, gamma=ttnn_gamma, epsilon=epsilon)

    assert ttnn_output.layout == input_layout, f"output layout {ttnn_output.layout} != input {input_layout}"
    torch_output = ttnn.to_torch(ttnn_output).to(torch.float32)
    assert list(torch_output.shape) == list(shape)

    expected = rms_norm_reference(torch_input, torch_gamma, epsilon)
    gate = min(PCC[dtype], PCC[gamma_dtype])
    pcc = _pcc(torch_output, expected)
    assert pcc >= gate, (
        f"PCC {pcc:.5f} < {gate} (shape={shape}, in_layout={input_layout}, dtype={dtype}, "
        f"gamma_layout={gamma_layout}, gamma_dtype={gamma_dtype})"
    )


# Aligned + W/H/both non-aligned; ranks 2/3/4; wide-hidden.
SHAPES = [
    pytest.param((32, 64), id="32x64_aligned_2d"),
    pytest.param((64, 128), id="64x128_aligned"),
    pytest.param((2, 4, 64, 128), id="multi_batch_4d"),
    pytest.param((2, 64, 128), id="rank3"),
    pytest.param((1, 1, 32, 256), id="wide_hidden"),
    pytest.param((32, 50), id="w_non_aligned"),
    pytest.param((50, 64), id="h_non_aligned"),
    pytest.param((1, 1, 47, 50), id="hw_non_aligned"),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["in_tile", "in_rm"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("gamma_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["g_tile", "g_rm"])
def test_rms_norm_gamma_layout(device, shape, input_layout, dtype, gamma_layout):
    """gamma_layout {TILE, RM} × input_layout {TILE, RM} × dtype {bf16, f32}."""
    _run(device, shape, input_layout, dtype, gamma_layout, gamma_dtype=dtype)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["in_tile", "in_rm"])
def test_rms_norm_tile_gamma_mixed_precision(device, shape, input_layout):
    """Mixed-precision LLM pattern: bf16 activations + f32 TILE gamma."""
    _run(device, shape, input_layout, ttnn.bfloat16, ttnn.TILE_LAYOUT, gamma_dtype=ttnn.float32)


@pytest.mark.parametrize("shape", [(64, 128), (2, 4, 64, 256), (32, 4096)], ids=["2d", "4d", "wide"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "f32"])
def test_rms_norm_bf8b_tile_gamma(device, shape, dtype):
    """bf8b gamma implies TILE gamma (block-float has no RM form); tile-aligned only."""
    _run(device, shape, ttnn.TILE_LAYOUT, dtype, ttnn.TILE_LAYOUT, gamma_dtype=ttnn.bfloat8_b)
