# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for the rms_norm operation — the immutable spec.

DO NOT MODIFY THIS FILE. It defines the contract the kernel must satisfy:
    RMSNorm(x) = x / sqrt(mean(x^2, dim=-1, keepdim=True) + epsilon) * gamma

Covers the mandated Phase-0 capabilities:
  * both layouts (TILE and ROW_MAJOR), native — no host-side layout/pad workaround;
  * both dtypes (bfloat16, float32), with the maxed-out precision corner
    (fp32_dest_acc_en=True) as the default;
  * non-tile-aligned H and/or W, native (result reflects only valid elements);
  * gamma present and absent (optional scale);
  * ranks 2, 3, 4.

PCC thresholds keyed by dtype match the golden suite: f32→0.999, bf16→0.995.
The `device` fixture comes from the root conftest (module-scoped via this
directory's conftest.py) — do NOT open a device manually.
"""

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm


# PCC gate per dtype — identical to the golden suite; not derived from op "complexity".
PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
}

TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
}


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    if torch.allclose(a, b):
        return 1.0
    stacked = torch.stack([a, b])
    return torch.corrcoef(stacked)[0, 1].item()


def rms_norm_reference(x: torch.Tensor, gamma: torch.Tensor | None, epsilon: float) -> torch.Tensor:
    """Golden RMSNorm over the last dim, computed in float32."""
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    out = x * torch.rsqrt(variance + epsilon)
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)  # broadcast on last dim
    return out


def _run_rms_norm(device, shape, layout, dtype, with_gamma, epsilon=1e-6, compute_kernel_config=None):
    torch.manual_seed(42)
    W = shape[-1]

    torch_input = torch.randn(shape, dtype=torch.float32)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_gamma = None
    torch_gamma = None
    if with_gamma:
        torch_gamma = torch.randn(W, dtype=torch.float32)
        # gamma is always provided ROW_MAJOR with shape (1, 1, 1, W).
        ttnn_gamma = ttnn.from_torch(
            torch_gamma.reshape(1, 1, 1, W),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    ttnn_output = rms_norm(
        ttnn_input,
        gamma=ttnn_gamma,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
    )

    # Output layout and shape must match the input.
    assert ttnn_output.layout == layout, f"output layout {ttnn_output.layout} != input layout {layout}"
    torch_output = ttnn.to_torch(ttnn_output).to(torch.float32)
    assert list(torch_output.shape) == list(shape), f"shape mismatch: {tuple(torch_output.shape)} vs {tuple(shape)}"

    expected = rms_norm_reference(torch_input, torch_gamma, epsilon)

    pcc = _pcc(torch_output, expected)
    assert (
        pcc >= PCC[dtype]
    ), f"PCC {pcc:.5f} < {PCC[dtype]} (shape={shape}, layout={layout}, dtype={dtype}, gamma={with_gamma})"


# Shapes exercise: single-tile, multi-tile non-square, multi-batch 4D, 3D,
# W non-aligned, H non-aligned, both non-aligned, wide-hidden. Ranks 2/3/4.
SHAPES = [
    pytest.param((32, 32), id="single_tile_2d"),
    pytest.param((64, 128), id="multi_tile_2d_nonsquare"),
    pytest.param((2, 4, 64, 128), id="multi_batch_4d"),
    pytest.param((2, 64, 128), id="rank3"),
    pytest.param((32, 50), id="w_non_aligned_2d"),
    pytest.param((50, 64), id="h_non_aligned_2d"),
    pytest.param((1, 1, 47, 50), id="hw_non_aligned_4d"),
    pytest.param((1, 1, 32, 256), id="wide_hidden"),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("with_gamma", [True, False], ids=["gamma", "no_gamma"])
def test_rms_norm(device, shape, layout, dtype, with_gamma):
    _run_rms_norm(device, shape, layout, dtype, with_gamma)


@pytest.mark.parametrize("shape", [(64, 128), (1, 1, 32, 256)], ids=["2d", "4d"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
def test_rms_norm_custom_epsilon(device, shape, layout):
    _run_rms_norm(device, shape, layout, ttnn.bfloat16, with_gamma=True, epsilon=1e-5)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "f32"])
def test_rms_norm_maxed_precision_config(device, dtype):
    """Phase-0 maxed-out precision corner: fp32_dest_acc_en=True, HiFi4."""
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
    )
    _run_rms_norm(
        device,
        (64, 512),
        ttnn.TILE_LAYOUT,
        dtype,
        with_gamma=True,
        compute_kernel_config=config,
    )
