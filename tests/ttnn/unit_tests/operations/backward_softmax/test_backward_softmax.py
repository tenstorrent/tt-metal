# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Acceptance test for backward_softmax — VJP of softmax.

Math under test:
    grad_input = output * (grad_output - sum(output * grad_output, dim))

This test is the immutable spec for the operation. The implementer must NOT
modify this file. If a parametrized case is impossible under the Phase 0
constraints (fp32, TILE_LAYOUT, 4D, H/W tile-aligned, dim ∈ {-1, -2}), the
implementer should fix the kernel rather than relax the test.

Coverage:
- Multiple shapes (single-tile, multi-tile, non-square, multi-batch).
- Both supported reduce dimensions (-1 and -2).
- A PyTorch reference that re-implements the formula directly (we do NOT use
  torch.autograd, because the spec describes a formula evaluated against
  given (grad_output, output) tensors — `output` is already a softmax result
  the caller provides).
- Negative tests: invalid dtype, layout, rank, dim, shape mismatch, dtype
  mismatch.
"""

import pytest
import torch
import ttnn

from ttnn.operations.backward_softmax import backward_softmax


# Phase-0 tolerances. The op chains a multiply, a reduction, a broadcast
# subtract, and another multiply, all in fp32 with HiFi4 + fp32 dest acc, so
# we are in the "multi-step compute" tolerance band.
RTOL = 0.05
ATOL = 0.01


def _torch_reference(grad_output: torch.Tensor, output: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Reference implementation of softmax-VJP, computed in fp32 to match the
    device-side precision policy.

        grad_input = output * (grad_output - sum(output * grad_output, dim))
    """
    grad_output = grad_output.float()
    output = output.float()
    s = (output * grad_output).sum(dim=dim, keepdim=True)
    return output * (grad_output - s)


# -----------------------------------------------------------------------------
# Positive cases
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        # Single tile per (n, c) plane — minimal case.
        pytest.param((1, 1, 32, 32), id="single_tile"),
        # Multi-tile along the reduction axis (dim=-1) and the orthogonal axis (dim=-2).
        pytest.param((1, 1, 32, 256), id="multi_tile_W"),
        pytest.param((1, 1, 256, 32), id="multi_tile_H"),
        # Non-square HxW.
        pytest.param((1, 1, 64, 128), id="non_square_64x128"),
        pytest.param((1, 1, 128, 64), id="non_square_128x64"),
        # Multi-batch — exercises the work distribution over (N*C) lanes-per-NC.
        pytest.param((2, 4, 64, 128), id="multi_batch"),
    ],
)
@pytest.mark.parametrize("dim", [-1, -2], ids=["dim=-1", "dim=-2"])
def test_backward_softmax_correctness(device, shape, dim):
    """
    Backward-softmax matches the reference formula across shapes and both
    supported reduce dimensions.
    """
    torch.manual_seed(42)

    # We don't require `output` to be a real softmax distribution — the formula
    # is well-defined for arbitrary inputs and exercising it on randn forces
    # the device path to exercise both signs and a mix of magnitudes. Using
    # randn for both inputs (rather than a real softmax) also keeps the test
    # invariant to forward-softmax precision.
    torch_grad_output = torch.randn(shape, dtype=torch.float32)
    torch_output = torch.randn(shape, dtype=torch.float32)

    torch_expected = _torch_reference(torch_grad_output, torch_output, dim=dim)

    ttnn_grad_output = ttnn.from_torch(
        torch_grad_output,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = ttnn.from_torch(
        torch_output,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_grad_input = backward_softmax(ttnn_grad_output, ttnn_output, dim=dim)

    # Sanity on metadata.
    assert tuple(ttnn_grad_input.shape) == tuple(
        shape
    ), f"Output shape {tuple(ttnn_grad_input.shape)} does not match input shape {shape}"
    assert ttnn_grad_input.dtype == ttnn.float32, f"Output dtype {ttnn_grad_input.dtype} != float32"
    assert ttnn_grad_input.layout == ttnn.TILE_LAYOUT

    actual = ttnn.to_torch(ttnn_grad_input).float()
    expected = torch_expected.float()

    max_abs = (actual - expected).abs().max().item()
    max_rel = ((actual - expected).abs() / (expected.abs() + 1e-6)).max().item()
    assert torch.allclose(actual, expected, rtol=RTOL, atol=ATOL), (
        f"Mismatch for shape={shape}, dim={dim}:\n"
        f"  max abs diff = {max_abs:.6f}\n"
        f"  max rel diff = {max_rel:.6f}\n"
        f"  actual.flat[:6]   = {actual.flatten()[:6].tolist()}\n"
        f"  expected.flat[:6] = {expected.flatten()[:6].tolist()}"
    )


def test_backward_softmax_default_dim_is_minus_one(device):
    """
    Calling backward_softmax(dy, y) without dim must default to dim=-1.
    """
    torch.manual_seed(42)
    shape = (1, 1, 32, 64)
    torch_grad_output = torch.randn(shape, dtype=torch.float32)
    torch_output = torch.randn(shape, dtype=torch.float32)

    torch_expected = _torch_reference(torch_grad_output, torch_output, dim=-1)

    ttnn_grad_output = ttnn.from_torch(
        torch_grad_output,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = ttnn.from_torch(
        torch_output,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # NOTE: no `dim=` argument — must behave like dim=-1.
    ttnn_grad_input = backward_softmax(ttnn_grad_output, ttnn_output)

    actual = ttnn.to_torch(ttnn_grad_input).float()
    assert torch.allclose(actual, torch_expected, rtol=RTOL, atol=ATOL)


def test_backward_softmax_against_real_softmax(device):
    """
    End-to-end sanity: when `output` is the actual softmax of some logits, the
    result of backward_softmax(dy, output, dim) matches torch.autograd's
    softmax backward (computed via vjp). This guards against sign or scaling
    bugs that might cancel out on randn inputs.
    """
    torch.manual_seed(42)
    shape = (1, 1, 64, 128)
    dim = -1

    logits = torch.randn(shape, dtype=torch.float32, requires_grad=True)
    softmax_out = torch.nn.functional.softmax(logits, dim=dim)
    grad_output = torch.randn(shape, dtype=torch.float32)
    softmax_out.backward(grad_output)
    torch_expected = logits.grad.detach().float()

    ttnn_grad_output = ttnn.from_torch(
        grad_output,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = ttnn.from_torch(
        softmax_out.detach(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_grad_input = backward_softmax(ttnn_grad_output, ttnn_output, dim=dim)
    actual = ttnn.to_torch(ttnn_grad_input).float()

    max_abs = (actual - torch_expected).abs().max().item()
    max_rel = ((actual - torch_expected).abs() / (torch_expected.abs() + 1e-6)).max().item()
    assert torch.allclose(actual, torch_expected, rtol=RTOL, atol=ATOL), (
        f"Mismatch vs torch.autograd softmax-backward:\n"
        f"  max abs diff = {max_abs:.6f}\n"
        f"  max rel diff = {max_rel:.6f}\n"
    )


# -----------------------------------------------------------------------------
# Negative cases (validation)
# -----------------------------------------------------------------------------


def _make_tensor(device, shape, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    torch.manual_seed(42)
    if dtype == ttnn.float32:
        torch_dtype = torch.float32
    elif dtype == ttnn.bfloat16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    return ttnn.from_torch(
        torch.randn(shape, dtype=torch_dtype),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_backward_softmax_rejects_bf16_dtype(device):
    """Phase 0: float32 only. bfloat16 inputs must be rejected."""
    bf16_grad = _make_tensor(device, (1, 1, 32, 32), dtype=ttnn.bfloat16)
    fp32_out = _make_tensor(device, (1, 1, 32, 32), dtype=ttnn.float32)
    with pytest.raises((ValueError, RuntimeError)):
        backward_softmax(bf16_grad, fp32_out)


def test_backward_softmax_rejects_dtype_mismatch(device):
    """grad_output and output dtypes must match."""
    fp32_grad = _make_tensor(device, (1, 1, 32, 32), dtype=ttnn.float32)
    bf16_out = _make_tensor(device, (1, 1, 32, 32), dtype=ttnn.bfloat16)
    with pytest.raises((ValueError, RuntimeError)):
        backward_softmax(fp32_grad, bf16_out)


def test_backward_softmax_rejects_row_major_layout(device):
    """TILE_LAYOUT only on both inputs."""
    rm_grad = _make_tensor(device, (1, 1, 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT)
    tile_out = _make_tensor(device, (1, 1, 32, 32), layout=ttnn.TILE_LAYOUT)
    with pytest.raises((ValueError, RuntimeError)):
        backward_softmax(rm_grad, tile_out)

    rm_out = _make_tensor(device, (1, 1, 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT)
    tile_grad = _make_tensor(device, (1, 1, 32, 32), layout=ttnn.TILE_LAYOUT)
    with pytest.raises((ValueError, RuntimeError)):
        backward_softmax(tile_grad, rm_out)


def test_backward_softmax_rejects_rank_lt_4(device):
    """Phase 0 requires rank == 4."""
    grad_3d = _make_tensor(device, (1, 32, 32))
    out_3d = _make_tensor(device, (1, 32, 32))
    with pytest.raises((ValueError, RuntimeError)):
        backward_softmax(grad_3d, out_3d)


def test_backward_softmax_rejects_shape_mismatch(device):
    """grad_output and output shapes must match exactly."""
    grad = _make_tensor(device, (1, 1, 32, 64))
    out = _make_tensor(device, (1, 1, 32, 32))
    with pytest.raises((ValueError, RuntimeError)):
        backward_softmax(grad, out)


@pytest.mark.parametrize("bad_dim", [0, 1, -3, 2, -4, 3], ids=lambda d: f"dim={d}")
def test_backward_softmax_rejects_invalid_dim(device, bad_dim):
    """Only dim ∈ {-1, -2} is supported in Phase 0."""
    grad = _make_tensor(device, (1, 1, 32, 32))
    out = _make_tensor(device, (1, 1, 32, 32))
    with pytest.raises((ValueError, RuntimeError)):
        backward_softmax(grad, out, dim=bad_dim)


def test_backward_softmax_rejects_non_tile_aligned(device):
    """
    Phase 0 requires H and W divisible by 32. Non-aligned inputs must be
    rejected. Note: ttnn.from_torch with TILE_LAYOUT will pad internally, so
    we provide a shape that the validator should still flag as having an
    underlying logical shape that is not tile-aligned.
    """
    grad = _make_tensor(device, (1, 1, 30, 32))
    out = _make_tensor(device, (1, 1, 30, 32))
    with pytest.raises((ValueError, RuntimeError)):
        backward_softmax(grad, out)
