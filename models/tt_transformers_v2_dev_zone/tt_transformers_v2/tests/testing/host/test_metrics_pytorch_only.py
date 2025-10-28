#!/usr/bin/env python3
"""PyTorch-only metric tests that do not require TTNN hardware."""

import torch
from tt_transformers_v2.src.testing import comp_allclose, compute_max_abs_error, compute_mean_abs_error, compute_pcc


def test_metrics_pytorch_only():
    """Verify fallback metric implementations using pure PyTorch tensors."""

    torch_a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    torch_b = torch.tensor([[1.0, 2.0, 3.5], [4.0, 5.0, 6.0]], dtype=torch.float32)

    max_error = compute_max_abs_error(torch_a, torch_b)
    mean_error = compute_mean_abs_error(torch_a, torch_b)
    pcc = compute_pcc(torch_a, torch_b)

    expected_max = 0.5
    expected_mean = (torch_a - torch_b).abs().mean().item()

    assert abs(max_error - expected_max) < 1e-6, f"Expected {expected_max}, got {max_error}"
    assert abs(mean_error - expected_mean) < 1e-6, "Mean error mismatch"
    assert 0.99 <= pcc <= 1.0, f"PCC should be high (~1.0) for similar tensors, got {pcc}"


def test_comp_allclose_pytorch_only():
    """PyTorch-only tests for comp_allclose covering pass/fail and edge cases."""

    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    b = a.clone()
    passed, msg = comp_allclose(a, b)
    assert passed, f"Expected pass for exact equality. Got: {msg}"
    assert "Max ATOL Delta" in msg and "Max RTOL Delta" in msg

    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    b = torch.tensor([1.0 + 1e-7, 2.0 - 1e-7, 3.0], dtype=torch.float32)
    passed, msg = comp_allclose(a, b)
    assert passed, f"Expected pass within default tolerance. Got: {msg}"

    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    b = torch.tensor([1.0, 2.1, 3.0], dtype=torch.float32)
    passed, msg = comp_allclose(a, b)
    assert not passed and "Allclose check failed" in msg

    a = torch.tensor([float("nan"), 1.0, 2.0], dtype=torch.float32)
    b = torch.tensor([float("nan"), 1.0, 2.0], dtype=torch.float32)
    passed, _ = comp_allclose(a, b)
    assert passed, "Both NaNs at same positions should pass"

    a = torch.tensor([float("inf"), -float("inf"), 1.0])
    b = torch.tensor([float("inf"), -float("inf"), 1.0])
    passed, _ = comp_allclose(a, b)
    assert passed, "Same sign infinities should pass"

    a = torch.tensor([float("inf"), -float("inf"), 1.0])
    b = torch.tensor([float("inf"), float("inf"), 1.0])
    passed, msg = comp_allclose(a, b)
    assert not passed and "Allclose check failed" in msg
