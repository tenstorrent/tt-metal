# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
One repeat of the Qwen3.6-27B hybrid attention pattern.

The checkpoint's `layer_types` is (linear, linear, linear, full) repeated 16
times -- `full_attention_interval = 4`. A block is one such repeat: three Gated
DeltaNet decoders followed by one Gated Attention decoder. Sixteen blocks make
up the 64-layer stack.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.qwen_3_27b.tt.tt_deltanet_decoder import TtQwen36DeltaNetDecoder
from models.experimental.qwen_3_27b.tt.tt_full_attention_decoder import TtQwen36FullAttentionDecoder

LAYERS_PER_BLOCK = 4  # full_attention_interval


class TtQwen36Block(LightweightModule):
    """
    Four decoder layers: three DeltaNet, then one Gated Attention.

    Shapes (D = 5120):
        input   [B, T, D]
        output  [B, T, D]

    Each attribute is a complete decoder layer (norm -> attention -> residual -> norm -> MLP -> residual).
    """

    def __init__(self, device, block_idx: int):
        self.device = device
        self.block_idx = block_idx

        # Global layer index -- each layer needs it to find its own weights in the
        # checkpoint (keys look like `model.layers.37.mlp.gate_proj.weight`).
        first_layer_idx = block_idx * LAYERS_PER_BLOCK

        self.delta_net_1 = TtQwen36DeltaNetDecoder(device, layer_idx=first_layer_idx + 0)
        self.delta_net_2 = TtQwen36DeltaNetDecoder(device, layer_idx=first_layer_idx + 1)
        self.delta_net_3 = TtQwen36DeltaNetDecoder(device, layer_idx=first_layer_idx + 2)
        self.gated_attention = TtQwen36FullAttentionDecoder(device, layer_idx=first_layer_idx + 3)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """[B, T, D] -> [B, T, D]."""
        x = self.delta_net_1(x)
        x = self.delta_net_2(x)
        x = self.delta_net_3(x)
        x = self.gated_attention(x)
        return x
