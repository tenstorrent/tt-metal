# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Decoder layer whose attention is Gated DeltaNet (linear attention).

48 of the 64 layers are this kind. Identical residual/norm/MLP wrapper to the
full-attention decoder -- only the attention module differs.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.qwen_3_27b.tt.tt_gated_deltanet import TtQwen36GatedDeltaNet
from models.experimental.qwen_3_27b.tt.tt_mlp import TtQwen36MLP
from models.experimental.qwen_3_27b.tt.tt_rms_norm import TtQwen36RmsNorm


class TtQwen36DeltaNetDecoder(LightweightModule):
    """
    input_norm -> gated_deltanet -> +residual -> post_norm -> mlp -> +residual

    Shapes (D = 5120):
        input   [B, T, D]
        output  [B, T, D]

    Every intermediate is [B, T, D] -- the residual stream never changes width.

    To implement:
        1. h = input_norm(x)
        2. h = gated_deltanet(h)
        3. x = x + h                       (residual 1)
        4. h = post_attention_norm(x)
        5. h = mlp(h)
        6. x = x + h                       (residual 2)
      + ttnn.deallocate() each intermediate once consumed -- at 27B this is
        required, not hygiene.
    """

    def __init__(self, device, layer_idx: int):
        self.device = device
        self.layer_idx = layer_idx

        self.input_norm = TtQwen36RmsNorm(device)
        self.gated_deltanet = TtQwen36GatedDeltaNet(device, layer_idx)
        self.post_attention_norm = TtQwen36RmsNorm(device)
        self.mlp = TtQwen36MLP(device, layer_idx)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """[B, T, D] -> [B, T, D]."""
        h = self.input_norm(x)
        h = self.gated_deltanet(h)

        x = ttnn.add(h, x)  # residual 1
        ttnn.deallocate(h)

        h = self.post_attention_norm(x)
        h = self.mlp(h)

        x = ttnn.add(h, x)  # residual 2
        ttnn.deallocate(h)

        return x
