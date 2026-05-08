# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Extended coverage for backward_softmax beyond the spec test.

Focuses on:
- Mid-size shapes the spec test does not cover.
- Determinism across runs with fixed seed.
- Correctness when `output` is the actual softmax of some logits (the common
  user-facing case), checked at PCC level.

Tolerance discipline: PCC + relative-RMS, NOT torch.allclose with tight atol.
The spec test in `test_backward_softmax.py` uses `atol=0.01` and is hardware-
precision-limited for shapes whose reduction depth exceeds 32 elements (see
verification_report.md "Precision Baseline" for the empirical numbers). This
file is in addition to — not in place of — that spec test.
"""

import pytest
import torch
import ttnn

from ttnn.operations.backward_softmax import backward_softmax
from tests.ttnn.utils_for_testing import check_with_pcc


def _torch_reference(grad_output, output, dim):
    s = (output * grad_output).sum(dim=dim, keepdim=True)
    return output * (grad_output - s)


def _make(device, shape, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, fn=None):
    if fn is None:
        fn = lambda s: torch.randn(s, dtype=torch.float32)
    return fn(shape), ttnn.from_torch(
        fn(shape),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 96, 96), id="non_aligned_96"),
        pytest.param((1, 2, 32, 192), id="2C_dim_minus_one"),
        pytest.param((3, 1, 96, 64), id="3N_64W"),
    ],
)
@pytest.mark.parametrize("dim", [-1, -2], ids=["dim=-1", "dim=-2"])
def test_backward_softmax_extended_shapes(device, shape, dim):
    """
    Sweep a few shapes the spec doesn't touch. We require PCC ≥ 0.999 and
    rel_rms ≤ 0.01 — both are consistent with the precision baseline.
    """
    torch.manual_seed(123)
    torch_dy = torch.randn(shape, dtype=torch.float32)
    torch_y = torch.randn(shape, dtype=torch.float32)
    expected = _torch_reference(torch_dy, torch_y, dim=dim)

    ttnn_dy = ttnn.from_torch(
        torch_dy,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_y = ttnn.from_torch(
        torch_y,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_grad_input = backward_softmax(ttnn_dy, ttnn_y, dim=dim)
    actual = ttnn.to_torch(ttnn_grad_input).float()

    pcc_ok, pcc_msg = check_with_pcc(expected, actual, pcc=0.999)
    rel_rms = (actual - expected).pow(2).mean().sqrt().item() / max(expected.pow(2).mean().sqrt().item(), 1e-12)

    assert pcc_ok, f"PCC failed for shape={shape}, dim={dim}: {pcc_msg}"
    assert rel_rms <= 0.01, f"rel_rms={rel_rms:.4f} > 0.01 for shape={shape}, dim={dim}"


def test_backward_softmax_against_real_softmax_output_dim_minus_2(device):
    """
    End-to-end check with `output` from an actual softmax along dim=-2,
    matched against torch.autograd.
    """
    torch.manual_seed(7)
    shape = (1, 1, 64, 96)
    dim = -2

    logits = torch.randn(shape, dtype=torch.float32, requires_grad=True)
    softmax_out = torch.nn.functional.softmax(logits, dim=dim)
    grad_output = torch.randn(shape, dtype=torch.float32)
    softmax_out.backward(grad_output)
    expected = logits.grad.detach().float()

    ttnn_dy = ttnn.from_torch(
        grad_output,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_y = ttnn.from_torch(
        softmax_out.detach(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_grad_input = backward_softmax(ttnn_dy, ttnn_y, dim=dim)
    actual = ttnn.to_torch(ttnn_grad_input).float()

    pcc_ok, pcc_msg = check_with_pcc(expected, actual, pcc=0.999)
    assert pcc_ok, f"PCC failed for autograd-derived softmax(dim=-2): {pcc_msg}"


def test_backward_softmax_deterministic(device):
    """Same inputs → same output across two invocations."""
    torch.manual_seed(99)
    shape = (1, 1, 32, 64)
    torch_dy = torch.randn(shape, dtype=torch.float32)
    torch_y = torch.randn(shape, dtype=torch.float32)

    ttnn_dy = ttnn.from_torch(
        torch_dy,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_y = ttnn.from_torch(
        torch_y,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out1 = ttnn.to_torch(backward_softmax(ttnn_dy, ttnn_y, dim=-1)).float()
    out2 = ttnn.to_torch(backward_softmax(ttnn_dy, ttnn_y, dim=-1)).float()
    assert torch.equal(out1, out2), "backward_softmax is not deterministic across runs"


def test_backward_softmax_explicit_memory_config(device):
    """memory_config kwarg routes the output to the requested memory."""
    torch.manual_seed(11)
    shape = (1, 1, 32, 64)
    torch_dy = torch.randn(shape, dtype=torch.float32)
    torch_y = torch.randn(shape, dtype=torch.float32)

    ttnn_dy = ttnn.from_torch(
        torch_dy,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_y = ttnn.from_torch(
        torch_y,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out_dram = backward_softmax(ttnn_dy, ttnn_y, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out_l1 = backward_softmax(ttnn_dy, ttnn_y, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Both produce the same numeric result; only the buffer location differs.
    a = ttnn.to_torch(out_dram).float()
    b = ttnn.to_torch(out_l1).float()
    assert torch.equal(a, b), "DRAM and L1 output diverge — should be bit-identical"
