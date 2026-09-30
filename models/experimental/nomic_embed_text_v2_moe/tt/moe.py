# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Router plus experts, the TTNN form of reference.NomicMoELayer.

This is the one place the activation layout changes. Blocks pass (B, 1, S, H), because attention
mixes tokens along S and flattening the batch away would let one text attend to another. The
expert matmuls need the opposite: ttnn.matmul broadcasts a weight's batch dims only when every
batch dim of the activation is 1, so (1, 1, T, H) x (1, E, H, F) gives (1, E, T, F) while
(B, 1, S, H) raises. The flatten therefore lives here, at the same place the reference does its
own x.view(-1, H), and nowhere else in the encoder.
"""

from __future__ import annotations

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.nomic_embed_text_v2_moe.tt.common import flatten_tokens, unflatten_tokens
from models.experimental.nomic_embed_text_v2_moe.tt.experts import TtNomicExperts
from models.experimental.nomic_embed_text_v2_moe.tt.router import TtNomicRouter


class TtNomicMoELayer(LightweightModule):
    """The FFN used on odd-numbered layers: route each token, then combine its experts.

    Upstream passes an inverted pad mask into this layer and then ignores it. Not threaded
    through here either: applying it would zero the real tokens rather than the padding.
    """

    def __init__(self, device, config, tt_config, state_dict, state_dict_prefix):
        super().__init__()
        self.router = TtNomicRouter(device, config, tt_config, state_dict, f"{state_dict_prefix}router.")
        self.experts = TtNomicExperts(device, config, tt_config, state_dict, f"{state_dict_prefix}experts.")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Route and combine, flattening the token axis for the duration.

        Args:
            x: (B, 1, S, H) block activations.

        Returns:
            ttnn.Tensor: (B, 1, S, H).
        """
        batch, _, seqlen, _ = x.shape

        flat = flatten_tokens(x)
        dense_weights = self.router(flat)
        out = self.experts(flat, dense_weights)
        ttnn.deallocate(dense_weights)
        # flat is not freed: the reshape aliases x, which the block still needs as the residual
        # for norm2.

        return unflatten_tokens(out, batch, seqlen)
