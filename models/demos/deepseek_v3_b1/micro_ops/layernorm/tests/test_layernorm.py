# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LayerNorm single-core generic op."""

import pytest
import torch

from models.demos.deepseek_v3_b1.micro_ops.layernorm.op import LayerNormSingleCore


@pytest.mark.parametrize(
    "shape",
    [
        [1, 32],
        [1, 64],
        [4, 128],
    ],
)
def test_golden_vs_torch(shape):
    """
    Test that LayerNormSingleCore.golden() matches torch.nn.functional.layer_norm().

    Pass criteria: torch.allclose(golden, torch_ref, rtol=1e-4, atol=1e-4) for all shapes.
    """
    torch.manual_seed(42)

    # Create input tensor with random values
    input_tensor = torch.randn(shape, dtype=torch.float32)

    # Create gamma (scale) and beta (shift) parameters with shape matching last dimension
    W = shape[-1]
    gamma_tensor = torch.randn(W, dtype=torch.float32)
    beta_tensor = torch.randn(W, dtype=torch.float32)

    epsilon = 1e-6

    # Compute using our golden implementation
    golden_output = LayerNormSingleCore.golden(input_tensor, gamma_tensor, beta_tensor, epsilon)

    # Compute using torch.nn.functional.layer_norm (reference)
    torch_output = torch.nn.functional.layer_norm(
        input_tensor,
        normalized_shape=[W],
        weight=gamma_tensor,
        bias=beta_tensor,
        eps=epsilon,
    )

    # Verify outputs match
    assert torch.allclose(golden_output, torch_output, rtol=1e-4, atol=1e-4), (
        f"Golden output does not match torch.nn.functional.layer_norm for shape {shape}\n"
        f"Max absolute difference: {(golden_output - torch_output).abs().max().item()}"
    )
