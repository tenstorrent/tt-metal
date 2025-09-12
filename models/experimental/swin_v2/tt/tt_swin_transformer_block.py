# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch.nn as nn
from models.experimental.swin_v2.tt.tt_mlp import TtMLP
from models.experimental.swin_v2.tt.tt_shifted_window_attention_v2 import TtShiftedWindowAttentionV2

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


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
        if use_signpost:
            signpost(header="swin_transformer_block")

        self.device = device
        self.parameters = parameters
        self.attn = TtShiftedWindowAttentionV2(
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
