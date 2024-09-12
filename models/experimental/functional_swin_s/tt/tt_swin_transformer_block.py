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

        # for m in self.mlp.modules():
        #     if isinstance(m, ttnn.linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             nn.init.normal_(m.bias, std=1e-6)

    def __call__(self, x):
        norm1 = ttnn.layer_norm(x, weight=self.parameters.norm1.weight, bias=self.parameters.norm1.bias)
        attn = self.attn(norm1)
        x = x + attn
        norm2 = ttnn.layer_norm(x, weight=self.parameters.norm2.weight, bias=self.parameters.norm2.bias)
        mlp = self.mlp(norm2)
        x = x + mlp
        return x
