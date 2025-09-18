# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch.nn as nn
from models.experimental.swin_s.tt.tt_mlp import TtMLP
from models.experimental.swin_s.tt.tt_shifted_window_attention import TtShiftedWindowAttention


class TtSwinTransformerBlock(nn.Module):
    def __init__(
        self,
        device,
        parameters,
        dim,
        num_heads,
        window_size,
        shift_size,
        mlp_ratio=4.0,
        attn_mask=None,
    ):
        super().__init__()
        self.device = device
        self.parameters = parameters
        self.attn = TtShiftedWindowAttention(
            parameters.attn,
            device,
            dim,
            window_size,
            shift_size,
            num_heads,
            attn_mask=attn_mask,
        )

        self.mlp = TtMLP(
            [int(dim * mlp_ratio), dim],
            device,
            parameters.mlp,
            activation_layer=ttnn.gelu,
            inplace=None,
        )

    def __call__(self, input_tensor):
        norm1 = ttnn.layer_norm(
            input_tensor,
            weight=self.parameters.norm1.weight,
            bias=self.parameters.norm1.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        attn = self.attn(norm1)
        output_tensor = input_tensor + attn
        ttnn.deallocate(norm1)
        norm2 = ttnn.layer_norm(
            output_tensor,
            weight=self.parameters.norm2.weight,
            bias=self.parameters.norm2.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn)
        mlp = self.mlp(norm2)
        output_tensor = output_tensor + mlp
        ttnn.deallocate(norm2)
        ttnn.deallocate(mlp)
        return output_tensor
