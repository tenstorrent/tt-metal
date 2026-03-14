# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Swin Transformer block for Swin-L backbone.
Adapted from models/experimental/swin_s/tt/tt_swin_transformer_block.py.
"""

import ttnn
from models.experimental.swin_l.tt.tt_swin_attention import TtSwinAttention
from models.experimental.swin_l.tt.tt_swin_mlp import TtSwinMLP


class TtSwinBlock:
    """Pre-norm Swin Transformer block: LN -> Attention -> residual -> LN -> MLP -> residual."""

    def __init__(self, device, parameters, dim, num_heads, window_size, shift_size, mlp_ratio=4.0, attn_mask=None):
        self.device = device
        self.parameters = parameters
        self.attn = TtSwinAttention(
            device, parameters["attn"], dim, window_size, shift_size, num_heads, attn_mask=attn_mask
        )
        self.mlp = TtSwinMLP(device, parameters["mlp"], dim, mlp_ratio=mlp_ratio)

    def __call__(self, input_tensor):
        # LN1 -> Attention
        attn_out = self.attn(
            ttnn.layer_norm(
                input_tensor,
                weight=self.parameters["norm1"]["weight"],
                bias=self.parameters["norm1"]["bias"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
        output = input_tensor + attn_out
        ttnn.deallocate(attn_out)

        # LN2 -> MLP
        mlp_out = self.mlp(
            ttnn.layer_norm(
                output,
                weight=self.parameters["norm2"]["weight"],
                bias=self.parameters["norm2"]["bias"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
        result = output + mlp_out
        ttnn.deallocate(mlp_out)
        return result
