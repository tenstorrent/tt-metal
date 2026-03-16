# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TDD Stage 4: affine
Full layer normalization with optional gamma (scale) and beta (bias) affine transform.

Reference: torch.nn.functional.layer_norm(input, [W], weight=gamma, bias=beta, eps=epsilon)
Tolerances: rtol=0.05, atol=0.2
"""

import pytest
import torch
import torch.nn.functional as F
import ttnn

from ttnn.operations.layer_norm_rm import layer_norm_rm


def pytorch_reference(input_tensor, gamma_val, beta_val, epsilon=1e-5):
    """Stage 4 reference: full layer norm with affine."""
    return torch.nn.functional.layer_norm(
        input_tensor.to(torch.float32),
        [input_tensor.shape[-1]],
        weight=gamma_val.to(torch.float32).squeeze(),
        bias=beta_val.to(torch.float32).squeeze(),
        eps=epsilon,
    ).to(torch.bfloat16)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="minimal_1x1x32x32"),
        pytest.param((1, 1, 64, 128), id="multi_tile_1x1x64x128"),
        pytest.param((1, 1, 32, 256), id="non_square_1x1x32x256"),
        pytest.param((4, 2, 64, 64), id="multi_batch_4x2x64x64"),
    ],
)
def test_stage_affine(device, shape):
    """Test full layer norm with gamma and beta affine transform."""
    epsilon = 1e-5
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    gamma_val = torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16)
    beta_val = torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_gamma = ttnn.from_torch(
        gamma_val,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_beta = ttnn.from_torch(
        beta_val,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm_rm(ttnn_input, ttnn_gamma, ttnn_beta, epsilon=epsilon)

    # Verify output shape
    assert list(ttnn_output.shape) == list(shape), f"Shape mismatch: {ttnn_output.shape} vs {shape}"

    # Compare with reference
    torch_output = ttnn.to_torch(ttnn_output)
    torch_expected = pytorch_reference(torch_input, gamma_val, beta_val, epsilon)

    assert torch.allclose(
        torch_output.float(),
        torch_expected.float(),
        rtol=0.05,
        atol=0.2,
    ), f"Output mismatch. Max diff: {(torch_output.float() - torch_expected.float()).abs().max()}"
