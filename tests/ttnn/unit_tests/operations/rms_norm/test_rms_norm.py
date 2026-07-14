# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for rms_norm — the immutable spec. Do NOT modify.

RMSNorm(x) = x / sqrt(mean(x^2, dim=-1, keepdim=True) + epsilon) * gamma

Covers both layouts (TILE / ROW_MAJOR — the op must handle both natively, no
host-side layout conversion), both phase-0 dtypes (bfloat16 / float32), with and
without gamma, and tile-aligned + non-tile-aligned H/W shapes. All runs use the
phase-0 maxed-out precision corner (fp32_dest_acc_en=True).
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.rms_norm import rms_norm


# --- PCC tolerances (same as the golden suite; not derived from op complexity) ---
PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
}


def pytorch_rms_norm(x, gamma=None, epsilon=1e-6):
    """Reference RMSNorm computed in fp32, returned in the input dtype."""
    original_dtype = x.dtype
    xf = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + epsilon)
    out = xf / rms
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out.to(original_dtype)


def _compute_kernel_config():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True  # phase-0 maxed-out corner
    cfg.math_approx_mode = False
    return cfg


# single-tile, multi-tile, non-square, multi-batch, non-tile-aligned (W and H), wide, 2D/3D/4D
SHAPES = [
    (1, 1, 32, 64),  # small tile-aligned
    (1, 1, 64, 128),  # multi-tile
    (2, 4, 128, 512),  # multi-batch, non-square
    (1, 1, 32, 50),  # W non-aligned
    (1, 1, 17, 64),  # H non-aligned
    (2, 1, 100, 47),  # both non-aligned
    (1, 1, 32, 4096),  # wide hidden
    (4, 128, 512),  # 3D
    (128, 512),  # 2D
    (128, 100),  # 2D, W non-aligned
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("with_gamma", [False, True])
def test_rms_norm(device, shape, dtype, layout, with_gamma):
    torch.manual_seed(42)
    torch_dtype = _TORCH_DTYPE[dtype]
    W = shape[-1]

    torch_input = torch.randn(shape, dtype=torch_dtype)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if with_gamma:
        torch_gamma = torch.randn(W, dtype=torch_dtype)
        ttnn_gamma = ttnn.from_torch(
            torch_gamma.reshape(1, 1, 1, W),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        torch_gamma = None
        ttnn_gamma = None

    expected = pytorch_rms_norm(torch_input, gamma=torch_gamma, epsilon=1e-6)

    ttnn_output = rms_norm(
        ttnn_input,
        gamma=ttnn_gamma,
        epsilon=1e-6,
        compute_kernel_config=_compute_kernel_config(),
    )

    assert ttnn_output.layout == layout, "output layout must match input layout"

    actual = ttnn.to_torch(ttnn_output).reshape(expected.shape)
    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), PCC[dtype])


@pytest.mark.parametrize("epsilon", [1e-6, 1e-5, 1e-2])
def test_rms_norm_epsilon(device, epsilon):
    torch.manual_seed(42)
    shape = (1, 1, 64, 256)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    expected = pytorch_rms_norm(torch_input, gamma=None, epsilon=epsilon)
    ttnn_output = rms_norm(ttnn_input, epsilon=epsilon, compute_kernel_config=_compute_kernel_config())
    actual = ttnn.to_torch(ttnn_output).reshape(expected.shape)
    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), PCC[ttnn.bfloat16])


def test_rms_norm_default_config(device):
    """None compute_kernel_config resolves through default_compute_kernel_config()."""
    torch.manual_seed(42)
    shape = (2, 1, 64, 128)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn(128, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma.reshape(1, 1, 1, 128),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    expected = pytorch_rms_norm(torch_input, gamma=torch_gamma, epsilon=1e-6)
    ttnn_output = rms_norm(ttnn_input, gamma=ttnn_gamma)  # no compute_kernel_config
    actual = ttnn.to_torch(ttnn_output).reshape(expected.shape)
    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), PCC[ttnn.bfloat16])


def test_rms_norm_rejects_rank_lt_2(device, expect_error):
    torch.manual_seed(42)
    ttnn_input = ttnn.from_torch(
        torch.randn(64, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error((ValueError, RuntimeError), ".*"):
        rms_norm(ttnn_input, compute_kernel_config=_compute_kernel_config())


def test_rms_norm_rejects_gamma_dim_mismatch(device, expect_error):
    torch.manual_seed(42)
    ttnn_input = ttnn.from_torch(
        torch.randn((1, 1, 32, 128), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_gamma = ttnn.from_torch(
        torch.randn(64, dtype=torch.bfloat16).reshape(1, 1, 1, 64),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error((ValueError, RuntimeError), ".*"):
        rms_norm(ttnn_input, gamma=ttnn_gamma, compute_kernel_config=_compute_kernel_config())
