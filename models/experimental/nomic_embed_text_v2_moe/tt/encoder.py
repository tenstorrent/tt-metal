# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The 12 blocks in sequence, the TTNN form of reference.NomicBertEncoder.

Every block preserves its input shape, so the stack does too: (B, 1, S, H) in, (B, 1, S, H) out.
The dense and MoE FFNs alternate; config.is_moe_layer owns the predicate.
"""

from __future__ import annotations

from typing import Optional

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.nomic_embed_text_v2_moe.tt.block import TtNomicBertBlock


class TtNomicBertEncoder(LightweightModule):
    """The encoder stack.

    The rotary tables and the additive mask are arguments rather than members: both depend only
    on S, which is the batch's longest sequence and so varies per call, and building them once
    per forward pass instead of once per block saves 11 repetitions of the same host work.
    """

    def __init__(self, device, config, tt_config, state_dict, state_dict_prefix="encoder."):
        super().__init__()
        # The trailing dot is part of the prefix, as it is for every module here. Inserting the
        # separator instead would make "encoder." build "encoder..layers.0." and raise KeyError.
        self.layers = [
            TtNomicBertBlock(
                device,
                config,
                tt_config,
                state_dict,
                f"{state_dict_prefix}layers.{idx}.",
                moe=config.is_moe_layer(idx),
            )
            for idx in range(config.num_hidden_layers)
        ]

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        attn_mask: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Run every block in order.

        Args:
            hidden_states: (B, 1, S, H) post-embedding input.
            rot_mats: (cos, sin), each (1, 1, S, D).
            attn_mask: (B, 1, S, S) additive mask, or None.

        Returns:
            ttnn.Tensor: (B, 1, S, H).
        """
        for idx, layer in enumerate(self.layers):
            next_hidden_states = layer(hidden_states, rot_mats, attn_mask)
            # Every intermediate is freed as soon as the next block has consumed it, but not the
            # caller's own input: the caller allocated it and may still need it.
            if idx > 0:
                ttnn.deallocate(hidden_states)
            hidden_states = next_hidden_states
        return hidden_states
