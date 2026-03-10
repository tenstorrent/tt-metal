# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TDD Stage 5: affine

Reader adds gamma/beta stick reads. Compute adds gamma/beta tilize,
mul<ROW>(gamma), add<ROW>(beta). Full layer normalization with affine transform.

Reference: torch.nn.functional.layer_norm(x, [x.shape[-1]], weight=gamma, bias=beta, eps=1e-5)
Tolerances: rtol=0.05, atol=0.2
"""

import pytest
import torch
import torch.nn.functional
import ttnn

from ttnn.operations.layer_norm_rm import layer_norm_rm


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="minimal"),
        pytest.param((1, 1, 64, 128), id="multi_tile_hw"),
        pytest.param((1, 1, 32, 256), id="wide"),
        pytest.param((4, 2, 64, 64), id="multi_batch"),
    ],
)
def test_stage_affine(device, shape):
    """Stage 5: Full layer normalization with affine transform (gamma + beta)."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    gamma_tensor = torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16)
    beta_tensor = torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gamma_ttnn = ttnn.from_torch(
        gamma_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_ttnn = ttnn.from_torch(
        beta_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm_rm(ttnn_input, gamma_ttnn, beta_ttnn, epsilon=1e-5)

    # Verify shape
    assert list(ttnn_output.shape) == list(shape), f"Shape mismatch: {ttnn_output.shape} vs {shape}"

    # Reference: layer norm with affine
    expected = torch.nn.functional.layer_norm(
        torch_input.float(),
        [shape[-1]],
        weight=gamma_tensor.squeeze().float(),
        bias=beta_tensor.squeeze().float(),
        eps=1e-5,
    )
    actual = ttnn.to_torch(ttnn_output)

    assert torch.allclose(
        actual.float(), expected.float(), rtol=0.05, atol=0.2
    ), f"Stage 5 affine failed. Max diff: {(actual.float() - expected.float()).abs().max()}"
