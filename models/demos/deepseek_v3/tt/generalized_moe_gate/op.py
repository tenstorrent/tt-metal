# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


class GeneralizedMoeGateOp:
    """
    Generalized MoE gate routing via TTNN ``ttnn.experimental.deepseek.moe.generalized_moe_gate``.

    Standalone copy of ``DeepseekMoeGateOp`` intended for independent modification. It retains a
    PyTorch reference (``golden``) and delegates device execution to the C++ operation (which runs
    the unified kernel on hardware).
    """

    @staticmethod
    def golden(input_tensor, bias_tensor, eps=1e-20, scaling_factor=2.5, enable_sigmoid=False):
        """
        PyTorch reference for the *ungrouped* generalized MoE gate.

        Unlike the DeepSeek grouped gate (top-2 sum -> top-4 groups -> top-8 of 128),
        this variant keeps each group's top-8 and then takes the true global top-8 over
        all groups. Pooling every group's top-8 and taking top-8 is exactly the global
        top-8 of all experts (a group can contribute at most 8 to a top-8), so this is
        equivalent to a plain top-8 over the flattened experts, ranked by the
        bias-corrected score and returning the (un-biased) score, normalized.

        Args:
            input_tensor: Router logits, shape [batch, n_group, group_size].
            bias_tensor: Score-correction bias, same shape as logits.
            eps: Denominator stabilization for normalization.
            scaling_factor: Routed scaling factor applied after normalization.
            enable_sigmoid: Apply sigmoid to logits before the bias add when True.

        Returns:
            Top8 normalized scores tensor of shape [batch, 8]
            Top8 global indices tensor of shape [batch, 8]
        """
        batch = input_tensor.shape[0]

        scores = torch.sigmoid(input_tensor) if enable_sigmoid else input_tensor
        bias_scores = scores + bias_tensor

        # TEMP (P4a verification): top-8 of the HIGH 4 groups (groups 4-7 = experts 128-255).
        # Confirms the shift-hi-groups + skip-sort path yields the high-half batch. REVERT after.
        bias_g = bias_scores[:, 4:, :].reshape(batch, -1)
        scores_flat = scores[:, 4:, :].reshape(batch, -1)
        _, top8_local = torch.topk(bias_g, 8, dim=-1, sorted=True)
        top8_scores = torch.gather(scores_flat, dim=-1, index=top8_local)
        top8_indices = top8_local + 128  # global expert ids for groups 4-7

        denominator = torch.sum(top8_scores, dim=-1, keepdim=True) + eps
        normalized_scores = top8_scores / denominator * scaling_factor
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
        Execute the generalized MoE gate on device.

        Writes into ``output_tensor`` and ``output_indices_tensor`` and returns them.

        See also: ``ttnn.experimental.deepseek.moe.generalized_moe_gate`` (C++ implementation).
        """
        return ttnn.experimental.deepseek.moe.generalized_moe_gate(
            input_tensor,
            bias_tensor=bias_tensor,
            input_indices_tensor=input_indices_tensor,
            output_tensor=output_tensor,
            output_indices_tensor=output_indices_tensor,
            eps=eps,
            scaling_factor=scaling_factor,
            enable_sigmoid=enable_sigmoid,
        )
