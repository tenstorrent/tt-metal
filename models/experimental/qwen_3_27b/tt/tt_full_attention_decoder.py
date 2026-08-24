# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Decoder layer whose attention is Gated (full, softmax) Attention.

16 of the 64 layers are this kind -- every 4th one. Identical residual/norm/MLP
wrapper to the DeltaNet decoder; only the attention module differs.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.qwen_3_27b.tt.tt_gated_attention import TtQwen36GatedAttention
from models.experimental.qwen_3_27b.tt.tt_mlp import TtQwen36MLP
from models.experimental.qwen_3_27b.tt.tt_rms_norm import TtQwen36RmsNorm


class TtQwen36FullAttentionDecoder(LightweightModule):
    """
    input_norm -> gated_attention -> +residual -> post_norm -> mlp -> +residual

    Shapes (D = 5120):
        input   [B, T, D]
        output  [B, T, D]

    Same structure as the DeltaNet decoder, but this layer is also where RoPE and
    the KV cache enter the picture -- so `forward` will grow cos/sin and cache
    arguments that the DeltaNet decoder never needs.

    To implement:
        1. h = input_norm(x)
        2. h = gated_attention(h)          (later: + cos, sin, kv_cache)
        3. x = x + h                       (residual 1)
        4. h = post_attention_norm(x)
        5. h = mlp(h)
        6. x = x + h                       (residual 2)
    """

    def __init__(self, device, layer_idx: int):
        self.device = device
        self.layer_idx = layer_idx

        self.input_norm = TtQwen36RmsNorm(device)
        self.gated_attention = TtQwen36GatedAttention(device, layer_idx)
        self.post_attention_norm = TtQwen36RmsNorm(device)
        self.mlp = TtQwen36MLP(device, layer_idx)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """[B, T, D] -> [B, T, D]."""
        return x
