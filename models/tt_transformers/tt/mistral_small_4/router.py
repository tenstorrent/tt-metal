# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Routed-expert router logits matching Hugging Face ``Mistral4TopkRouter``."""

from __future__ import annotations

import torch

from models.tt_transformers.tt.mistral_small_4.linear import linear_bf16_no_bias


def router_logits_bf16(
    mesh_device,
    hidden_states_bsh: torch.Tensor,
    weight_experts_hidden: torch.Tensor,
) -> torch.Tensor:
    """
    Compute pre-softmax router logits on device.

    Same math as ``Mistral4TopkRouter.forward``: flatten tokens, then ``F.linear(x, W)``.

    Args:
        hidden_states_bsh: ``[B, S, hidden_size]``.
        weight_experts_hidden: HF ``gate.weight``, shape ``[n_routed_experts, hidden_size]``.

    Returns:
        Host bf16 tensor ``[B * S, n_routed_experts]`` (same layout as HF before ``route_tokens_to_experts``).

    ``hidden_size`` and ``n_routed_experts`` should be multiples of 32 for current TILE linear paths.
    """
    logits_bsh = linear_bf16_no_bias(mesh_device, hidden_states_bsh, weight_experts_hidden)
    b, s, e = logits_bsh.shape
    return logits_bsh.reshape(b * s, e)
