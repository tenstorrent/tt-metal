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
from models.experimental.nomic_embed_text_v2_moe.tt.common import rotary_rotation_matrix


class TtNomicBertEncoder(LightweightModule):
    """The encoder stack.

    The rotary tables and the additive mask are arguments rather than members: both depend only
    on S, which is the batch's longest sequence and so varies per call, and building them once
    per forward pass instead of once per block saves 11 repetitions of the same host work. The
    model keeps the tables across forwards (tt.common.RotaryTables).
    """

    def __init__(self, device, config, tt_config, state_dict, state_dict_prefix="encoder."):
        super().__init__()
        # The trailing dot is part of the prefix, as it is for every module here. Inserting the
        # separator instead would make "encoder." build "encoder..layers.0." and raise KeyError.
        # The last block's output leaves the model and outlives the forward: in L1 it would sit
        # under ops that plan L1 as free, so it goes to DRAM whatever its size.
        last = config.num_hidden_layers - 1
        rotation_matrix = rotary_rotation_matrix(device, tt_config.core_grid)
        self.layers = [
            TtNomicBertBlock(
                device,
                config,
                tt_config,
                state_dict,
                f"{state_dict_prefix}layers.{idx}.",
                moe=config.is_moe_layer(idx),
                output_memory_config=ttnn.DRAM_MEMORY_CONFIG if idx == last else None,
                rotation_matrix=rotation_matrix,
            )
            for idx in range(config.num_hidden_layers)
        ]

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        attn_mask: Optional[ttnn.Tensor] = None,
        free_input: bool = False,
        hidden_sharded: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Run every block in order.

        Args:
            hidden_states: (B, 1, S, H) post-embedding input.
            rot_mats: (cos, sin), each (1, 1, S, D) or longer, from tt.common.RotaryTables.
            attn_mask: (B, 1, S, S) additive mask, or None.
            free_input: free hidden_states once the first block is done with it; otherwise it stays
                the caller's.
            hidden_sharded: hidden_states in the split norms' layout (tt.common.layer_norm), handed to
                the first block, which frees it; each block hands its own to the next.

        Returns:
            ttnn.Tensor: (B, 1, S, H).
        """
        last = len(self.layers) - 1
        for idx, layer in enumerate(self.layers):
            # Each block frees its input as soon as it has read it for the last time (TtNomicBertBlock),
            # the caller's own input only when the caller hands it over.
            free = idx > 0 or free_input
            if idx == last:
                hidden_states = layer(hidden_states, rot_mats, attn_mask, free_input=free, x_sharded=hidden_sharded)
            else:
                hidden_states, hidden_sharded = layer(
                    hidden_states, rot_mats, attn_mask, free_input=free, x_sharded=hidden_sharded, keep_sharded=True
                )
        return hidden_states
