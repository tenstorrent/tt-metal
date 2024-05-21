# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
from models.experimental.functional_swin_s.tt.tt_mlp import TtMLP
from models.experimental.functional_swin_s.tt.tt_shifted_window_attention import TtShiftedWindowAttention


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
        dropout=0.0,
        attention_dropout=0.0,
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
            attention_dropout=attention_dropout,
            dropout=dropout,
        )

        self.mlp = TtMLP(
            [int(dim * mlp_ratio), dim],
            device,
            parameters.mlp,
            activation_layer=ttnn.gelu,
            inplace=None,
            dropout=dropout,
        )

        # for m in self.mlp.modules():
        #     if isinstance(m, ttnn.linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             nn.init.normal_(m.bias, std=1e-6)

    def __call__(self, x):
        x = x.to(self.device)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        norm1 = ttnn.layer_norm(x, weight=self.parameters.norm1.weight, bias=self.parameters.norm1.bias)
        norm1 = ttnn.to_layout(norm1, ttnn.ROW_MAJOR_LAYOUT)
        attn = self.attn(norm1)
        attn = ttnn.to_layout(attn, ttnn.TILE_LAYOUT)
        x = x + attn
        norm2 = ttnn.layer_norm(x, weight=self.parameters.norm2.weight, bias=self.parameters.norm2.bias)
        mlp = self.mlp(norm2)
        mlp = mlp.to(self.device)
        x = x + mlp
        return ttnn.from_device(x)
