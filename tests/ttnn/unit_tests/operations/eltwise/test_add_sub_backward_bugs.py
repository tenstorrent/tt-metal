# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "are_required_outputs",
    [
        [True, False],
        [False, True],
        [True, True],
    ],
)
def test_add_bw_respects_are_required_outputs(device, are_required_outputs):
    """Test that add_bw respects are_required_outputs when optional grads are not preallocated."""
    a = torch.randn(1, 1, 32, 32)
    b = torch.randn(1, 1, 32, 32)
    grad = torch.randn(1, 1, 32, 32)

    a_tt = ttnn.from_torch(a, device=device)
    b_tt = ttnn.from_torch(b, device=device)
    grad_tt = ttnn.from_torch(grad, device=device)

    result = ttnn.add_bw(grad_tt, a_tt, b_tt, are_required_outputs=are_required_outputs)

    assert len(result) == 2
    if are_required_outputs[0]:
        assert result[0] is not None, "grad_a should be computed when are_required_outputs[0]=True"
    else:
        assert result[0] is None, "grad_a should be None when are_required_outputs[0]=False"

    if are_required_outputs[1]:
        assert result[1] is not None, "grad_b should be computed when are_required_outputs[1]=True"
    else:
        assert result[1] is None, "grad_b should be None when are_required_outputs[1]=False"


@pytest.mark.parametrize(
    "are_required_outputs",
    [
        [True, False],
        [False, True],
        [True, True],
    ],
)
def test_sub_bw_respects_are_required_outputs(device, are_required_outputs):
    """Test that sub_bw respects are_required_outputs when optional grads are not preallocated."""
    a = torch.randn(1, 1, 32, 32)
    b = torch.randn(1, 1, 32, 32)
    grad = torch.randn(1, 1, 32, 32)

    a_tt = ttnn.from_torch(a, device=device)
    b_tt = ttnn.from_torch(b, device=device)
    grad_tt = ttnn.from_torch(grad, device=device)

    result = ttnn.sub_bw(grad_tt, a_tt, b_tt, are_required_outputs=are_required_outputs)

    assert len(result) == 2
    if are_required_outputs[0]:
        assert result[0] is not None, "grad_a should be computed when are_required_outputs[0]=True"
    else:
        assert result[0] is None, "grad_a should be None when are_required_outputs[0]=False"

    if are_required_outputs[1]:
        assert result[1] is not None, "grad_b should be computed when are_required_outputs[1]=True"
    else:
        assert result[1] is None, "grad_b should be None when are_required_outputs[1]=False"


@pytest.mark.parametrize(
    "input_shape, other_shape, grad_shape",
    [
        ((1, 1, 1, 32), (1, 1, 32, 32), (1, 1, 32, 32)),
        ((1, 1, 32, 32), (1, 1, 1, 32), (1, 1, 32, 32)),
        ((1, 1, 1, 1), (1, 1, 32, 32), (1, 1, 32, 32)),
    ],
)
def test_addalpha_bw_reduces_broadcast_dims(device, input_shape, other_shape, grad_shape):
    """Test that addalpha_bw reduces gradients over broadcasted dimensions."""
    a = torch.randn(*input_shape)
    b = torch.randn(*other_shape)
    grad = torch.randn(*grad_shape)
    alpha = 2.0

    a_tt = ttnn.from_torch(a, device=device)
    b_tt = ttnn.from_torch(b, device=device)
    grad_tt = ttnn.from_torch(grad, device=device)

    result = ttnn.addalpha_bw(grad_tt, a_tt, b_tt, alpha=alpha)

    assert len(result) == 2
    assert result[0] is not None
    assert result[1] is not None

    # Check shapes match input shapes (not grad shape)
    grad_a_tt = result[0]
    grad_b_tt = result[1]

    grad_a = ttnn.to_torch(grad_a_tt)
    grad_b = ttnn.to_torch(grad_b_tt)

    assert grad_a.shape == torch.Size(input_shape), f"grad_a shape {grad_a.shape} != input shape {input_shape}"
    assert grad_b.shape == torch.Size(other_shape), f"grad_b shape {grad_b.shape} != other shape {other_shape}"

    # Verify numerical correctness against PyTorch
    a.requires_grad_(True)
    b.requires_grad_(True)
    out = a + alpha * b
    out.backward(grad)

    assert_with_pcc(grad_a, a.grad, pcc=0.999)
    assert_with_pcc(grad_b, b.grad, pcc=0.999)


@pytest.mark.parametrize(
    "input_shape, other_shape, grad_shape",
    [
        ((1, 1, 1, 32), (1, 1, 32, 32), (1, 1, 32, 32)),
        ((1, 1, 32, 32), (1, 1, 1, 32), (1, 1, 32, 32)),
        ((1, 1, 1, 1), (1, 1, 32, 32), (1, 1, 32, 32)),
    ],
)
def test_subalpha_bw_reduces_broadcast_dims(device, input_shape, other_shape, grad_shape):
    """Test that subalpha_bw reduces gradients over broadcasted dimensions."""
    a = torch.randn(*input_shape)
    b = torch.randn(*other_shape)
    grad = torch.randn(*grad_shape)
    alpha = 2.0

    a_tt = ttnn.from_torch(a, device=device)
    b_tt = ttnn.from_torch(b, device=device)
    grad_tt = ttnn.from_torch(grad, device=device)

    result = ttnn.subalpha_bw(grad_tt, a_tt, b_tt, alpha=alpha)

    assert len(result) == 2
    assert result[0] is not None
    assert result[1] is not None

    grad_a_tt = result[0]
    grad_b_tt = result[1]

    grad_a = ttnn.to_torch(grad_a_tt)
    grad_b = ttnn.to_torch(grad_b_tt)

    assert grad_a.shape == torch.Size(input_shape), f"grad_a shape {grad_a.shape} != input shape {input_shape}"
    assert grad_b.shape == torch.Size(other_shape), f"grad_b shape {grad_b.shape} != other shape {other_shape}"

    # Verify numerical correctness against PyTorch
    a.requires_grad_(True)
    b.requires_grad_(True)
    out = a - alpha * b
    out.backward(grad)

    assert_with_pcc(grad_a, a.grad, pcc=0.999)
    assert_with_pcc(grad_b, b.grad, pcc=0.999)
