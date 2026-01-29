# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""LayerNorm single-core generic op implementation using ProgramDescriptor API."""

import torch


class LayerNormSingleCore:
    """
    Single-core LayerNorm implementation using generic_op infrastructure.

    Performs row-wise layer normalization: output = (x - mean) * rsqrt(var + eps) * gamma + beta
    """

    @staticmethod
    def golden(input_tensor, gamma_tensor, beta_tensor, epsilon=1e-6):
        """
        Golden reference implementation using PyTorch.

        Args:
            input_tensor: Input tensor (torch.Tensor)
            gamma_tensor: Scale parameter tensor (torch.Tensor)
            beta_tensor: Shift parameter tensor (torch.Tensor)
            epsilon: Small constant for numerical stability

        Returns:
            torch.Tensor: Normalized output
        """
        # Compute mean along the last dimension (row-wise)
        mean = input_tensor.mean(dim=-1, keepdim=True)

        # Compute variance along the last dimension (unbiased=False for population variance)
        var = input_tensor.var(dim=-1, unbiased=False, keepdim=True)

        # Standardize: (x - mean) / sqrt(var + epsilon)
        normalized = (input_tensor - mean) / torch.sqrt(var + epsilon)

        # Apply affine transformation: gamma * normalized + beta
        output = normalized * gamma_tensor + beta_tensor

        return output

    @staticmethod
    def op(input_tensor, gamma_tensor, beta_tensor, output_tensor, epsilon=1e-6):
        """
        Execute LayerNorm on device using generic_op.

        Args:
            input_tensor: Input tensor on device (row-major, interleaved, DRAM)
            gamma_tensor: Scale parameter tensor on device
            beta_tensor: Shift parameter tensor on device
            output_tensor: Pre-allocated output tensor on device
            epsilon: Small constant for numerical stability

        Returns:
            output_tensor with results
        """
        raise NotImplementedError("Op implementation pending - Step 1.5")
