# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import List
from models.experimental.swin_v2.tt.tt_swin_transformer_block import TtSwinTransformerBlock


class TtSwinTransformerBlockV2(TtSwinTransformerBlock):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        device=None,
        parameters=None,
        attn_mask=None,
    ):
        super().__init__(
            device, parameters, dim, num_heads, window_size, shift_size, mlp_ratio=mlp_ratio, attn_mask=attn_mask
        )
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attn_mask_shape = attn_mask.shape

    def forward(self, x):
        attn = self.attn.forward(x)
        norm1 = ttnn.layer_norm(
            attn,
            weight=self.parameters.norm1.weight,
            bias=self.parameters.norm1.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        x = ttnn.add(x, norm1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(norm1)
        mlp = self.mlp(x)
        norm2 = ttnn.layer_norm(
            mlp,
            weight=self.parameters.norm2.weight,
            bias=self.parameters.norm2.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn)
        x = ttnn.add(x, norm2, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(norm2)
        ttnn.deallocate(mlp)
        return x
