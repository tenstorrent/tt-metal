# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Router plus experts, the TTNN form of reference.NomicMoELayer.

This is the one place the activation layout changes. Blocks pass (B, 1, S, H), because attention
mixes tokens along S and flattening the batch away would let one text attend to another. The
expert matmuls need the opposite: they take every token on one axis against every expert's
weights at once, (1, 1, T, H) x (H, E*F) on a stacked pass. The flatten therefore lives here, at
the same place the reference does its own x.view(-1, H), and nowhere else in the encoder.
"""

from __future__ import annotations

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.nomic_embed_text_v2_moe.tt.common import flatten_tokens, unflatten_tokens
from models.experimental.nomic_embed_text_v2_moe.tt.experts import HELD_MAX_TILES, StackedBuffers, TtNomicExperts
from models.experimental.nomic_embed_text_v2_moe.tt.router import TtNomicRouter


class TtNomicMoELayer(LightweightModule):
    """The FFN used on odd-numbered layers: route each token, then combine its experts.

    Upstream passes an inverted pad mask into this layer and then ignores it. Not threaded
    through here either: applying it would zero the real tokens rather than the padding.
    """

    def __init__(self, device, config, tt_config, state_dict, state_dict_prefix, buffers: StackedBuffers | None = None):
        super().__init__()
        self.router = TtNomicRouter(device, config, tt_config, state_dict, f"{state_dict_prefix}router.")
        self.experts = TtNomicExperts(device, config, tt_config, state_dict, f"{state_dict_prefix}experts.")
        self.buffers = buffers

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Route and combine, flattening the token axis for the duration.

        Args:
            x: (B, 1, S, H) block activations.

        Returns:
            ttnn.Tensor: (B, 1, S, H).
        """
        batch, seqlen = x.shape[0], x.shape[-2]
        # Counted as the blocks count them, each sequence tile-padded: 120 one-token sequences are
        # 120 tokens of the MoE's flat axis but 120 tile rows to every dense layer around it.
        small = batch * ttnn.core.divup(seqlen, ttnn.TILE_SIZE) <= HELD_MAX_TILES

        flat = flatten_tokens(x)
        dense_weights = self.router(flat)
        out = self.experts(flat, dense_weights, buffers=self.buffers if small else None)
        ttnn.deallocate(dense_weights)
        # flat is not freed: the reshape aliases x, which the block still needs as the residual
        # for norm2.

        return unflatten_tokens(out, batch, seqlen)
