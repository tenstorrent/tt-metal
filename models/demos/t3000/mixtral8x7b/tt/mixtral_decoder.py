# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.demos.t3000.mixtral8x7b.tt.mixtral_attention import TtMixtralAttention
from models.demos.t3000.mixtral8x7b.tt.mixtral_mlp import TtMixtralMLP
from models.demos.t3000.mixtral8x7b.tt.mixtral_moe import TtMoeLayer
from models.common.rmsnorm import RMSNorm
from models.common.lightweightmodule import LightweightModule


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

        self.args = args

        self.layer_num = layer_num
        self.attention = TtMixtralAttention(
            mesh_device=mesh_device,
            state_dict=state_dict,
            args=args,
            layer_num=layer_num,
            dtype=dtype,
        )

        self.feed_forward = TtMoeLayer(
            mesh_device=mesh_device,
            state_dict=state_dict,
            experts=TtMixtralMLP(
                mesh_device=mesh_device,
                state_dict=state_dict,
                args=args,
                layer_num=layer_num,
                dtypes={
                    "w1": ttnn.bfloat8_b,
                    "w2": ttnn.bfloat8_b,
                    "w3": ttnn.bfloat8_b,
                },
            ),
            args=args,
            layer_num=layer_num,
            dtype=dtype,
        )
        self.attention_norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            layer_num=layer_num,
            weight_dtype=ttnn.bfloat16,
            weight_key="attention_norm",
        )

        self.ffn_norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            layer_num=layer_num,
            weight_dtype=ttnn.bfloat16,
            weight_key="ffn_norm",
        )

    def forward(
        self,
        xs_1SBH,
        start_pos_ids,
        attn_masks,
        rot_mat,
        transformation_mats=None,
        user_id=0,
        mode="decode",
        start_pos_ids_tensor=None,
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
            start_pos_ids,
            attn_masks,
            rot_mat,
            transformation_mats,
            user_id,
            mode,
            start_pos_ids_tensor=start_pos_ids_tensor,
        )
        hs_1SBH = ttnn.add(xs_1SBH, attn_1SBH)
        # xs_1SBH.deallocate(True)
        # attn_1SBH.deallocate(True)
        ffn_norm_1SBH = self.ffn_norm(hs_1SBH)
        ffn_1SBH = self.feed_forward(ffn_norm_1SBH, mode=mode)
        out_1SBH = ttnn.add(hs_1SBH, ffn_1SBH)
        # hs_1SBH.deallocate(True)
        # ffn_1SBH.deallocate(True)
        return out_1SBH
