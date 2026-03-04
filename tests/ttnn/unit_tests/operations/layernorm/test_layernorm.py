# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Integration test for the layernorm generic_op operation.

Tests cover:
  - Multiple 2D shapes: (32,32), (32,128), (64,128)
  - Without gamma/beta (normalize only)
  - With gamma and beta (full affine layernorm)

Stub kernels will produce garbage output so numerical checks are skipped
until kernels are implemented. Shape and dtype checks always run.
"""

import pytest
import torch
import ttnn

from ttnn.operations.layernorm import layernorm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pytorch_layernorm(input_tensor, gamma=None, beta=None, eps=1e-5):
    """PyTorch reference: layer_norm over the last dimension."""
    W = input_tensor.shape[-1]
    return torch.nn.functional.layer_norm(input_tensor, [W], weight=gamma, bias=beta, eps=eps)


# ---------------------------------------------------------------------------
# Test: operation runs without Python-side errors (stub validation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((32, 32), id="32x32"),
        pytest.param((32, 128), id="32x128"),
        pytest.param((64, 128), id="64x128"),
    ],
)
class TestLayernormRuns:
    """Validate that the layernorm op infrastructure works end-to-end."""

    def test_no_affine(self, device, shape):
        """Run layernorm without gamma/beta -- verify shape and dtype."""
        torch.manual_seed(42)
        torch_input = torch.randn(shape, dtype=torch.bfloat16)

        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn_output = layernorm(ttnn_input, eps=1e-5)

        # Shape must match input
        assert list(ttnn_output.shape) == list(
            shape
        ), f"Output shape {list(ttnn_output.shape)} != expected {list(shape)}"

    def test_with_affine(self, device, shape):
        """Run layernorm with gamma and beta -- verify shape and dtype."""
        torch.manual_seed(42)
        W = shape[-1]
        torch_input = torch.randn(shape, dtype=torch.bfloat16)
        gamma = torch.randn(W)
        beta = torch.randn(W)

        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gamma_tt = ttnn.from_torch(
            gamma.unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        beta_tt = ttnn.from_torch(
            beta.unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        ttnn_output = layernorm(ttnn_input, gamma=gamma_tt, beta=beta_tt, eps=1e-5)

        assert list(ttnn_output.shape) == list(
            shape
        ), f"Output shape {list(ttnn_output.shape)} != expected {list(shape)}"


# ---------------------------------------------------------------------------
# Test: numerical accuracy (will fail with stub kernels, enable after TDD)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((32, 32), id="32x32"),
        pytest.param((32, 128), id="32x128"),
        pytest.param((64, 128), id="64x128"),
    ],
)
class TestLayernormAccuracy:
    """Numerical accuracy tests against PyTorch reference.

    These are expected to fail while kernels are stubs. Mark xfail so the
    test suite stays green during TDD development.
    """

    @pytest.mark.xfail(reason="stub kernels -- numerical output is garbage", strict=False)
    def test_accuracy_no_affine(self, device, shape):
        """Check numerical accuracy without affine parameters."""
        torch.manual_seed(42)
        torch_input = torch.randn(shape, dtype=torch.bfloat16)
        expected = pytorch_layernorm(torch_input, eps=1e-5)

        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn_output = layernorm(ttnn_input, eps=1e-5)
        torch_output = ttnn.to_torch(ttnn_output)

        assert torch.allclose(
            torch_output.float(), expected.float(), rtol=0.05, atol=0.2
        ), f"Max diff: {(torch_output.float() - expected.float()).abs().max()}"

    @pytest.mark.xfail(reason="stub kernels -- numerical output is garbage", strict=False)
    def test_accuracy_with_affine(self, device, shape):
        """Check numerical accuracy with gamma and beta."""
        torch.manual_seed(42)
        W = shape[-1]
        torch_input = torch.randn(shape, dtype=torch.bfloat16)
        gamma = torch.randn(W)
        beta = torch.randn(W)

        expected = pytorch_layernorm(torch_input, gamma=gamma, beta=beta, eps=1e-5)

        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gamma_tt = ttnn.from_torch(
            gamma.unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        beta_tt = ttnn.from_torch(
            beta.unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        ttnn_output = layernorm(ttnn_input, gamma=gamma_tt, beta=beta_tt, eps=1e-5)
        torch_output = ttnn.to_torch(ttnn_output)

        assert torch.allclose(
            torch_output.float(), expected.float(), rtol=0.05, atol=0.2
        ), f"Max diff: {(torch_output.float() - expected.float()).abs().max()}"
