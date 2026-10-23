# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


class DeepseekMoeGateOp:
    """
    DeepSeek MoE gate routing via TTNN ``ttnn.experimental.deepseek.moe.deepseek_moe_gate``.

    This class retains a PyTorch reference (``golden``) and delegates device execution to the C++
    operation (which runs the unified kernel on hardware).
    """

    @staticmethod
    def golden(input_tensor, bias_tensor, eps=1e-20, scaling_factor=2.5, enable_sigmoid=False):
        """
        PyTorch reference implementation of Deepseek Moe Gate for validation.

        Args:
            input_tensor: Input tensor (torch.Tensor) of shape [16, 16]
            bias_tensor: Bias tensor (torch.Tensor) of shape [16, 16]
            eps: Epsilon value for normalization
            scaling_factor: Scaling factor for normalization
            enable_sigmoid: Whether to enable sigmoid activation

        Returns:
            Top8 normalized scores tensor (torch.Tensor) of shape [1, 8]
            Top8 indices tensor (torch.Tensor) of shape [1, 8]
        """
        row_offsets = torch.arange(input_tensor.shape[-2]) * input_tensor.shape[-1]
        batch_idx = torch.arange(input_tensor.shape[0]).unsqueeze(-1)

        scores = torch.sigmoid(input_tensor) if enable_sigmoid else input_tensor
        bias_scores = scores + bias_tensor
        sorted_bias, sorted_indices = torch.sort(bias_scores, dim=-1, descending=True)
        sorted_scores = torch.gather(scores, dim=-1, index=sorted_indices)
        sorted_indices = sorted_indices + row_offsets.view(1, -1, 1)

        top2_sum = sorted_bias[:, :, 0] + sorted_bias[:, :, 1]
        sorted_top2_sum, sorted_top2_indices = torch.sort(top2_sum, dim=-1, descending=True)
        top4_values = sorted_bias[batch_idx, sorted_top2_indices[:, :4]].flatten(1)
        top4_scores = sorted_scores[batch_idx, sorted_top2_indices[:, :4]].flatten(1)
        top4_indices = sorted_indices[batch_idx, sorted_top2_indices[:, :4]].flatten(1)
        top8_values, top8_indices = torch.topk(top4_values, 8, dim=-1, sorted=True)
        top8_values = torch.gather(top4_scores, dim=-1, index=top8_indices)
        top8_indices = torch.gather(top4_indices, dim=-1, index=top8_indices)
        denominator = torch.sum(top8_values, dim=-1, keepdim=True) + eps
        normalized_scores = top8_values / denominator * scaling_factor
        return normalized_scores, top8_indices

    @staticmethod
    def op(
        input_tensor,
        bias_tensor,
        output_tensor,
        input_indices_tensor,
        output_indices_tensor,
        eps=1e-20,
        scaling_factor=2.5,
        enable_sigmoid=True,
    ):
        """
        Execute DeepSeek MoE gate on device.

        Writes into ``output_tensor`` and ``output_indices_tensor`` and returns them.

        See also: ``ttnn.experimental.deepseek.moe.deepseek_moe_gate`` (C++ implementation).
        """
        return ttnn.experimental.deepseek.moe.deepseek_moe_gate(
            input_tensor,
            bias_tensor=bias_tensor,
            input_indices_tensor=input_indices_tensor,
            output_tensor=output_tensor,
            output_indices_tensor=output_indices_tensor,
            eps=eps,
            scaling_factor=scaling_factor,
            enable_sigmoid=enable_sigmoid,
        )
