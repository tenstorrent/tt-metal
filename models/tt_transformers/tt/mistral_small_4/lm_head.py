# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LM head logits (bias-free linear): ``Mistral4ForCausalLM.lm_head`` / ``nn.Linear(hidden, vocab)``."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.tt_transformers.tt.mistral_small_4.linear import linear_bf16_no_bias


def lm_head_logits_reference_torch(hidden_states_bsh: torch.Tensor, weight_vocab_hidden: torch.Tensor) -> torch.Tensor:
    """
    CPU reference: ``F.linear(x, W)`` with HF ``lm_head.weight`` layout ``[vocab_size, hidden_size]``.
    """
    return F.linear(
        hidden_states_bsh.to(torch.bfloat16),
        weight_vocab_hidden.to(torch.bfloat16),
        bias=None,
    )


def lm_head_logits_bf16(
    mesh_device, hidden_states_bsh: torch.Tensor, weight_vocab_hidden: torch.Tensor
) -> torch.Tensor:
    """
    Project last hidden states to vocabulary on device; returns host bf16 ``[B, S, vocab_size]``.

    Same math as :func:`linear_bf16_no_bias` with ``out_features = vocab_size``. Both dimensions
    should be multiples of **32** for the current TILE linear path.
    """
    return linear_bf16_no_bias(mesh_device, hidden_states_bsh, weight_vocab_hidden)
