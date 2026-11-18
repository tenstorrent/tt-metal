# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for ttnn.operations.rms_norm (Phase 0 contract).

    RMSNorm(x) = x / sqrt(mean(x^2, dim=-1, keepdim=True) + epsilon) * gamma

Phase 0 corner: dtype in {float32, bfloat16}, fp32_dest_acc_en=True, TILE and ROW_MAJOR,
tile-aligned and non-aligned shapes, rank 2-4, gamma optional (ROW_MAJOR (1,1,1,W); TILE also
accepted), DRAM interleaved. Shapes in `REGIME_SHAPES` pin the op's two built regimes
(LOCAL row-resident and W_SPLIT cross-core combine) on any grid with >= 2 cores - see op_design.md
"Regime-pinned tests".

This file is the spec. Do not modify it to make an implementation pass.
"""

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm

PCC_THRESHOLD = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
}

TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
}


def torch_rms_norm(x, gamma=None, epsilon=1e-6):
    """Reference in float32, returned in the input dtype."""
    xf = x.to(torch.float32)
    y = xf / torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + epsilon)
    if gamma is not None:
        y = y * gamma.to(torch.float32).reshape(-1)
    return y.to(x.dtype)


def compute_pcc(expected, actual):
    a = expected.to(torch.float64).flatten()
    b = actual.to(torch.float64).flatten()
    if torch.equal(a, b):
        return 1.0
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt((a * a).sum()) * torch.sqrt((b * b).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def make_inputs(shape, dtype, layout, device, *, with_gamma, gamma_layout=ttnn.ROW_MAJOR_LAYOUT, gamma_dtype=None):
    torch.manual_seed(42)
    torch_dtype = TORCH_DTYPE[dtype]
    torch_x = torch.randn(shape, dtype=torch.float32).to(torch_dtype)
    tt_x = ttnn.from_torch(
        torch_x, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    torch_gamma, tt_gamma = None, None
    if with_gamma:
        gdtype = gamma_dtype if gamma_dtype is not None else dtype
        W = shape[-1]
        torch_gamma = torch.randn((1, 1, 1, W), dtype=torch.float32).to(TORCH_DTYPE[gdtype])
        tt_gamma = ttnn.from_torch(
            torch_gamma, dtype=gdtype, layout=gamma_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    return torch_x, tt_x, torch_gamma, tt_gamma


def check(tt_out, torch_x, torch_gamma, *, dtype, layout, epsilon=1e-6):
    assert tt_out.layout == layout, f"output layout {tt_out.layout} != input layout {layout}"
    assert tt_out.dtype == dtype, f"output dtype {tt_out.dtype} != input dtype {dtype}"
    assert list(tt_out.shape) == list(torch_x.shape), f"output shape {list(tt_out.shape)} != {list(torch_x.shape)}"
    actual = ttnn.to_torch(tt_out)
    expected = torch_rms_norm(torch_x, torch_gamma, epsilon)
    assert torch.isfinite(actual.to(torch.float32)).all(), "output contains non-finite values"
    pcc = compute_pcc(expected, actual)
    assert pcc >= PCC_THRESHOLD[dtype], f"PCC {pcc:.6f} < {PCC_THRESHOLD[dtype]}"


def maxed_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# Single-tile, multi-tile, non-square, multi-batch, ranks 2/3/4, non-aligned W / H / both.
BASIC_SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 1, 64, 128),  # multi-tile
    (2, 3, 64, 96),  # multi-batch, non-square
    (128, 512),  # rank 2
    (2, 64, 256),  # rank 3
    (1, 1, 32, 50),  # W non-aligned
    (1, 1, 17, 64),  # H non-aligned
    (2, 1, 100, 47),  # both non-aligned, multi-batch
]

# Regime-pinned shapes (op_design.md, Work Distribution).
REGIME_SHAPES = [
    (4, 8, 32, 256),  # LOCAL on any grid
    (1, 1, 32, 4096),  # W_SPLIT by occupancy
    (1, 1, 32, 16384),  # W_SPLIT by residency (a row does not fit one core)
    (1, 1, 32, 4095),  # W_SPLIT + partial last W tile on the last core only
    (1, 1, 32, 4064),  # W_SPLIT, prime tile count (ragged last slice)
    (1, 1, 128, 8192),  # wide, several tile-rows
]

DTYPES = [ttnn.bfloat16, ttnn.float32]
LAYOUTS = [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]


@pytest.mark.parametrize("with_gamma", [False, True], ids=["no_gamma", "gamma"])
@pytest.mark.parametrize("layout", LAYOUTS, ids=["tile", "rm"])
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
@pytest.mark.parametrize("shape", BASIC_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_rms_norm_basic(device, shape, dtype, layout, with_gamma):
    torch_x, tt_x, torch_gamma, tt_gamma = make_inputs(shape, dtype, layout, device, with_gamma=with_gamma)
    tt_out = rms_norm(tt_x, gamma=tt_gamma, compute_kernel_config=maxed_config())
    check(tt_out, torch_x, torch_gamma, dtype=dtype, layout=layout)


@pytest.mark.parametrize("layout", LAYOUTS, ids=["tile", "rm"])
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
@pytest.mark.parametrize("shape", REGIME_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_rms_norm_regimes(device, shape, dtype, layout):
    torch_x, tt_x, torch_gamma, tt_gamma = make_inputs(shape, dtype, layout, device, with_gamma=True)
    tt_out = rms_norm(tt_x, gamma=tt_gamma, compute_kernel_config=maxed_config())
    check(tt_out, torch_x, torch_gamma, dtype=dtype, layout=layout)


@pytest.mark.parametrize("layout", LAYOUTS, ids=["tile", "rm"])
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
def test_rms_norm_default_call(device, dtype, layout):
    """rms_norm(x): no gamma, default epsilon, default compute config (the Phase 0 corner)."""
    shape = (1, 1, 64, 256)
    torch_x, tt_x, _, _ = make_inputs(shape, dtype, layout, device, with_gamma=False)
    tt_out = rms_norm(tt_x)
    check(tt_out, torch_x, None, dtype=dtype, layout=layout)


@pytest.mark.parametrize("epsilon", [1e-5, 1e-2])
@pytest.mark.parametrize("layout", LAYOUTS, ids=["tile", "rm"])
def test_rms_norm_epsilon(device, layout, epsilon):
    shape = (1, 1, 64, 128)
    dtype = ttnn.bfloat16
    torch_x, tt_x, torch_gamma, tt_gamma = make_inputs(shape, dtype, layout, device, with_gamma=True)
    # Shrink the input so epsilon is a visible fraction of mean(x^2).
    small_x = (torch_x.to(torch.float32) * 0.05).to(torch_x.dtype)
    tt_x = ttnn.from_torch(small_x, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_out = rms_norm(tt_x, gamma=tt_gamma, epsilon=epsilon)
    check(tt_out, small_x, torch_gamma, dtype=dtype, layout=layout, epsilon=epsilon)


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
def test_rms_norm_full_invocation(device, dtype):
    """rms_norm(x, gamma=g, epsilon=1e-5, compute_kernel_config=<maxed>)."""
    shape = (2, 1, 64, 320)
    torch_x, tt_x, torch_gamma, tt_gamma = make_inputs(shape, dtype, ttnn.TILE_LAYOUT, device, with_gamma=True)
    tt_out = rms_norm(tt_x, gamma=tt_gamma, epsilon=1e-5, compute_kernel_config=maxed_config())
    check(tt_out, torch_x, torch_gamma, dtype=dtype, layout=ttnn.TILE_LAYOUT, epsilon=1e-5)


@pytest.mark.parametrize("shape", [(1, 1, 32, 72), (1, 1, 64, 4095)], ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("layout", LAYOUTS, ids=["tile", "rm"])
def test_rms_norm_tile_gamma(device, shape, layout):
    """Gamma supplied in TILE layout (row 0 valid) is consumed natively."""
    dtype = ttnn.bfloat16
    torch_x, tt_x, torch_gamma, tt_gamma = make_inputs(
        shape, dtype, layout, device, with_gamma=True, gamma_layout=ttnn.TILE_LAYOUT
    )
    tt_out = rms_norm(tt_x, gamma=tt_gamma)
    check(tt_out, torch_x, torch_gamma, dtype=dtype, layout=layout)


@pytest.mark.parametrize(
    "dtype,gamma_dtype", [(ttnn.bfloat16, ttnn.float32), (ttnn.float32, ttnn.bfloat16)], ids=["bf16_x_fp32_g", "fp32_x_bf16_g"]
)
def test_rms_norm_mixed_gamma_dtype(device, dtype, gamma_dtype):
    shape = (1, 1, 64, 192)
    torch_x, tt_x, torch_gamma, tt_gamma = make_inputs(
        shape, dtype, ttnn.TILE_LAYOUT, device, with_gamma=True, gamma_dtype=gamma_dtype
    )
    tt_out = rms_norm(tt_x, gamma=tt_gamma)
    check(tt_out, torch_x, torch_gamma, dtype=dtype, layout=ttnn.TILE_LAYOUT)


@pytest.mark.parametrize("shape", [(1, 1, 32, 72), (1, 1, 40, 40), (1, 1, 32, 4095)], ids=lambda s: "x".join(map(str, s)))
def test_rms_norm_poisoned_tile_padding(device, shape):
    """TILE padding filled with a loud value must not leak into the RMS denominator."""
    dtype = ttnn.bfloat16
    torch_x, tt_x, torch_gamma, tt_gamma = make_inputs(
        shape, dtype, ttnn.TILE_LAYOUT, device, with_gamma=True, gamma_layout=ttnn.TILE_LAYOUT
    )
    tt_x = ttnn.fill_implicit_tile_padding(tt_x, 1000.0)
    tt_gamma = ttnn.fill_implicit_tile_padding(tt_gamma, 1000.0)
    tt_out = rms_norm(tt_x, gamma=tt_gamma)
    check(tt_out, torch_x, torch_gamma, dtype=dtype, layout=ttnn.TILE_LAYOUT)


def test_rms_norm_rejects_rank_1(device):
    x = ttnn.from_torch(torch.randn(64), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with pytest.raises((ValueError, RuntimeError)):
        rms_norm(x)


@pytest.mark.parametrize("layout", LAYOUTS, ids=["tile", "rm"])
def test_rms_norm_rejects_gamma_width_mismatch(device, layout):
    x = ttnn.from_torch(torch.randn(1, 1, 32, 64), dtype=ttnn.bfloat16, layout=layout, device=device)
    g = ttnn.from_torch(torch.randn(1, 1, 1, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with pytest.raises((ValueError, RuntimeError)):
        rms_norm(x, gamma=g)
