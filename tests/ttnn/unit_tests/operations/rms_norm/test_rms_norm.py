# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for rms_norm (Phase 0 contract).

Phase 0: TILE layout, interleaved DRAM, bfloat16 / float32 input at fp32_dest_acc_en=True,
tile-aligned and non-tile-aligned H / W, ranks 2-4, optional ROW_MAJOR gamma of shape (1, 1, 1, W).
"""

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm

PCC_BY_DTYPE = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
}


def reference_rms_norm(x, gamma=None, epsilon=1e-6):
    x32 = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(x32 * x32, dim=-1, keepdim=True) + epsilon)
    out = x32 / rms
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out


def pcc(expected, actual):
    expected = expected.to(torch.float64).flatten()
    actual = actual.to(torch.float64).flatten()
    if torch.equal(expected, actual):
        return 1.0
    return torch.corrcoef(torch.stack([expected, actual]))[0, 1].item()


def make_gamma(width, dtype, device):
    torch_gamma = torch.randn(width, dtype=torch.float32).to(TORCH_DTYPE[dtype])
    ttnn_gamma = ttnn.from_torch(
        torch_gamma.reshape(1, 1, 1, width),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return torch_gamma, ttnn_gamma


def run_and_check(device, shape, dtype, *, gamma_dtype=None, epsilon=None, compute_kernel_config=None):
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32).to(TORCH_DTYPE[dtype])
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    kwargs = {}
    torch_gamma = None
    if gamma_dtype is not None:
        torch_gamma, kwargs["gamma"] = make_gamma(shape[-1], gamma_dtype, device)
    if epsilon is not None:
        kwargs["epsilon"] = epsilon
    if compute_kernel_config is not None:
        kwargs["compute_kernel_config"] = compute_kernel_config

    output = rms_norm(ttnn_input, **kwargs)

    assert output.layout == ttnn.TILE_LAYOUT
    assert output.dtype == dtype
    assert list(output.shape) == list(shape)

    expected = reference_rms_norm(torch_input, torch_gamma, 1e-6 if epsilon is None else epsilon)
    actual = ttnn.to_torch(output).to(torch.float32)
    assert list(actual.shape) == list(shape)
    assert torch.isfinite(actual).all()

    measured = pcc(expected, actual)
    assert measured >= PCC_BY_DTYPE[dtype], f"PCC {measured} < {PCC_BY_DTYPE[dtype]} for {shape} {dtype}"


SHAPES = [
    pytest.param((1, 1, 32, 32), id="single_tile"),
    pytest.param((1, 1, 64, 128), id="multi_tile"),
    pytest.param((1, 1, 32, 256), id="non_square"),
    pytest.param((2, 4, 64, 96), id="multi_batch"),
    pytest.param((128, 512), id="rank2"),
    pytest.param((4, 128, 256), id="rank3"),
    pytest.param((1, 1, 1024, 256), id="many_rows"),
    pytest.param((1, 1, 32, 50), id="w_non_aligned"),
    pytest.param((1, 1, 17, 64), id="h_non_aligned"),
    pytest.param((2, 1, 100, 47), id="hw_non_aligned"),
    pytest.param((32, 17), id="w_lt_tile"),
    pytest.param((1, 1, 32, 4096), id="wide"),
]

DTYPES = [
    pytest.param(ttnn.bfloat16, id="bf16"),
    pytest.param(ttnn.float32, id="fp32"),
]


@pytest.mark.parametrize("with_gamma", [True, False], ids=["gamma", "no_gamma"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_rms_norm(device, shape, dtype, with_gamma):
    run_and_check(device, shape, dtype, gamma_dtype=dtype if with_gamma else None)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rms_norm_custom_epsilon_and_config(device, dtype):
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
    )
    run_and_check(
        device,
        (2, 1, 64, 160),
        dtype,
        gamma_dtype=dtype,
        epsilon=1e-5,
        compute_kernel_config=config,
    )


@pytest.mark.parametrize("shape", [(1, 1, 64, 128), (1, 1, 32, 72)], ids=["aligned", "w_non_aligned"])
def test_rms_norm_mixed_gamma_dtype(device, shape):
    run_and_check(device, shape, ttnn.bfloat16, gamma_dtype=ttnn.float32)


def test_rms_norm_rejects_rank1(device, expect_error):
    torch_input = torch.randn(64, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    # Error wording is not part of the contract; only the exception type is.
    with expect_error((ValueError, RuntimeError), ".*"):
        rms_norm(ttnn_input)


def test_rms_norm_rejects_gamma_width_mismatch(device, expect_error):
    torch.manual_seed(42)
    torch_input = torch.randn((1, 1, 32, 64), dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    _, ttnn_gamma = make_gamma(96, ttnn.bfloat16, device)
    # Error wording is not part of the contract; only the exception type is.
    with expect_error((ValueError, RuntimeError), ".*"):
        rms_norm(ttnn_input, gamma=ttnn_gamma)
