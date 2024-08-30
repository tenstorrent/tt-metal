# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.experimental.grok.tt.grok_attention import TtGrokAttention
from models.experimental.grok.tt.grok_mlp import TtGrokMLP
from models.experimental.grok.tt.grok_rms_norm import TtRMSNormSharded, TtRMSNorm
from models.experimental.grok.tt.grok_moe import TtMoeLayer
from models.experimental.grok.tt.grok_common import LightweightModule
from models.experimental.grok.scripts.tlog import tlog, tlog_mesh_device


class TtTransformerBlock(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        args,
        layer_num,
        dtype,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        tlog_mesh_device = mesh_device

        self.args = args

        self.layer_num = layer_num
        self.attention = TtGrokAttention(
            mesh_device=mesh_device,
            state_dict=state_dict,
            args=args,
            layer_num=layer_num,
            dtype=dtype,
        )

        self.feed_forward = TtMoeLayer(
            mesh_device=mesh_device,
            state_dict=state_dict,
            experts=TtGrokMLP(
                mesh_device=mesh_device,
                state_dict=state_dict,
                args=args,
                layer_num=layer_num,
                dtypes={
                    "linear": ttnn.bfloat8_b,
                    "linear_1": ttnn.bfloat8_b,
                    "linear_v": ttnn.bfloat8_b,
                },
            ),
            args=args,
            layer_num=layer_num,
            dtype=dtype,
        )
        make_norm = lambda name: TtRMSNormSharded(
            mesh_device=mesh_device,
            state_dict=state_dict,
            args=args,
            dtype=ttnn.bfloat16,
            layer_num=layer_num,
            weight_key=name,
        )

        self.pre_attn_norm = make_norm("pre_attn_norm")
        self.post_attn_norm = make_norm("post_attn_norm")
        self.pre_moe_norm = make_norm("pre_moe_norm")
        self.post_moe_norm = make_norm("post_moe_norm")

    def forward(
        self,
        xs_1SBH,
        current_pos,
        attn_masks,
        rot_mats,
    ) -> ttnn.Tensor:
        """
        Tensors are postfixed with 4 characters that represent their 4-D shape:
        B: batch dim (32)
        S: seq dim (1)
        1: unary dim
        H: hidden dim (6144)
        """
        # # tlog('our_decoder_input', xs_1SBH)
        hidden_1SBH = self.pre_attn_norm(xs_1SBH)
        # tlog('our_decoder_pre_attn_norm', hidden_1SBH)
        hidden_1SBH = self.attention(
            hidden_1SBH,
            current_pos,
            attn_masks,
            rot_mats,
        )
        # tlog('our_decoder_attention', hidden_1SBH)
        hidden_1SBH = self.post_attn_norm(hidden_1SBH)
        # tlog('our_decoder_post_attn_norm', hidden_1SBH)
        hidden_1SBH = ttnn.add(xs_1SBH, hidden_1SBH)
        # tlog('our_decoder_add', hidden_1SBH)
        residual_1SBH = hidden_1SBH

        hidden_1SBH = self.pre_moe_norm(hidden_1SBH)
        # tlog('our_decoder_pre_moe_norm', hidden_1SBH)
        hidden_1SBH = self.feed_forward(hidden_1SBH)
        # tlog('our_decoder_feed_forward', hidden_1SBH)
        hidden_1SBH = self.post_moe_norm(hidden_1SBH)
        # tlog('our_decoder_post_moe_norm', hidden_1SBH)
        hidden_1SBH = ttnn.add(residual_1SBH, hidden_1SBH)
        # tlog('our_decoder_output', hidden_1SBH)
        return hidden_1SBH
