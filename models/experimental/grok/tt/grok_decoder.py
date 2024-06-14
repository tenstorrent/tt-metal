# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.experimental.grok.tt.grok_attention import TtGrokAttention
from models.experimental.grok.tt.grok_mlp import TtGrokMLP
from models.experimental.grok.tt.grok_rms_norm import TtRMSNormSharded, TtRMSNorm
from models.experimental.grok.tt.grok_moe import TtMoeLayer
from models.experimental.grok.tt.grok_common import LightweightModule


class TtTransformerBlock(LightweightModule):
    def __init__(
        self,
        device_mesh,
        state_dict,
        args,
        layer_num,
        dtype,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device_mesh = device_mesh

        self.args = args

        self.layer_num = layer_num
        self.attention = TtGrokAttention(
            device_mesh=device_mesh,
            state_dict=state_dict,
            args=args,
            layer_num=layer_num,
            dtype=dtype,
        )

        self.feed_forward = TtMoeLayer(
            device_mesh=device_mesh,
            state_dict=state_dict,
            experts=TtGrokMLP(
                device_mesh=device_mesh,
                state_dict=state_dict,
                args=args,
                layer_num=layer_num,
                dtypes={
                    "w1": ttnn.bfloat4_b,
                    "w2": ttnn.bfloat8_b,
                    "w3": ttnn.bfloat4_b,
                },
            ),
            args=args,
            layer_num=layer_num,
            dtype=dtype,
        )
        self.attention_norm = TtRMSNorm(
            device_mesh=device_mesh,
            state_dict=state_dict,
            args=args,
            dtype=ttnn.bfloat16,
            layer_num=layer_num,
            weight_key="attention_norm",
        )

        self.ffn_norm = TtRMSNorm(
            device_mesh=device_mesh,
            state_dict=state_dict,
            args=args,
            dtype=ttnn.bfloat16,
            layer_num=layer_num,
            weight_key="ffn_norm",
        )

    def forward(
        self,
        xs_1SBH,
        start_pos,
        current_pos,
        attn_masks,
        rot_mats,
    ) -> ttnn.Tensor:
        """
        Tensors are postfixed with 4 characters that represent their 4-D shape:
        B: batch dim (32)
        S: seq dim (1)
        1: unary dim
        H: hidden dim (4096)
        """
        attn_norm_1SBH = self.attention_norm(xs_1SBH)

        attn_1SBH = self.attention(
            attn_norm_1SBH,
            start_pos,
            current_pos,
            attn_masks,
            rot_mats,
        )
        hs_1SBH = ttnn.add(xs_1SBH, attn_1SBH)
        ffn_norm_1SBH = self.ffn_norm(hs_1SBH)
        ffn_1SBH = self.feed_forward(ffn_norm_1SBH)
        out_1SBH = ttnn.add(hs_1SBH, ffn_1SBH)
        return out_1SBH
