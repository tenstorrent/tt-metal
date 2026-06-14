# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Text token embedding for MiniMax-M3 (TP=32, replicated table).

Matches ``reference.functional.embedding_forward``: a plain ``nn.Embedding``
row lookup over a [vocab 200064, hidden 6144] table with NO scaling
(MiniMax-M3 feeds the raw embedding into the decoder).

First-cut TP recipe: REPLICATE the table on every device (~2.4 GB/chip in
bf16, acceptable for the first pass). Sharding the vocab/hidden dim is a
follow-on optimization. The forward is a single ``ttnn.embedding`` op — no
torch fallback.
"""

from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule


class Embedding(LightweightModule):
    def __init__(self, mesh_device, weight: ttnn.Tensor):
        """
        Args:
            mesh_device: open bh_galaxy mesh.
            weight: REPLICATED ttnn embedding table [vocab, hidden], ROW_MAJOR.
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.weight = weight

    def forward(self, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        """Look up token rows. ``input_ids`` is a uint32 ttnn tensor [B, S]."""
        return ttnn.embedding(input_ids, self.weight, layout=ttnn.TILE_LAYOUT)
