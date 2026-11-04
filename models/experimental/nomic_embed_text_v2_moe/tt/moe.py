# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Router plus experts, the TTNN form of reference.NomicMoELayer.

This is the one place the activation layout changes. Blocks pass (B, 1, S, H), because attention
mixes tokens along S and flattening the batch away would let one text attend to another. The
expert pass needs the opposite: it gathers each token's row from one (1, 1, T, H) token axis by
index. The flatten therefore lives here, at the same place the reference does its own
x.view(-1, H), and nowhere else in the encoder.
"""

from __future__ import annotations

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.nomic_embed_text_v2_moe.tt.common import (
    flatten_tokens,
    nd_norm_layout,
    sharded_norm_layout,
    unflatten_tokens,
)
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

        The experts' L1 plans are made below x when x is in L1 (l1_floor).

        Args:
            x: (B, 1, S, H) block activations.

        Returns:
            ttnn.Tensor: (B, 1, S, H).
        """
        batch, seqlen = x.shape[0], x.shape[-2]
        nd_layout = nd_norm_layout(x.shape, self.experts.tt_config)
        if nd_layout is not None:
            # On the generated norm's ND layout nothing is reshaped, since a reshape of an ND tensor copies it:
            # each sequence is whole tiles there (nd_norm_layout), so (B, 1, S, H) is the flat token axis page for
            # page. router_front_end reads it as it stands, and the combine writes (B, 1, S, H) in that layout.
            routing, x_rows = self.router(x, self.experts.dispatch_table_config(x))
            out = self.experts(x, routing, nd_layout, x_rows, batch=batch)
            routing.deallocate()
            return out
        flat = flatten_tokens(x)
        routing, x_rows = self.router(flat, self.experts.dispatch_table_config(flat))
        # The combine writes norm2's split layout where unflattening it is a view, not a copy.
        output_layout = None
        if batch == 1 or seqlen % ttnn.TILE_SIZE == 0:
            output_layout = sharded_norm_layout(x.shape, self.experts.tt_config.core_grid)
        out = self.experts(flat, routing, None if output_layout is None else output_layout[0], x_rows)
        routing.deallocate()
        # flat is not freed: the reshape aliases x, which the block still needs as the residual
        # for norm2.

        return unflatten_tokens(out, batch, seqlen)
