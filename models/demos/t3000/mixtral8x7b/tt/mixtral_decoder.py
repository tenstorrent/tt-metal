# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn
from models.demos.t3000.mixtral8x7b.tt.mixtral_attention import TtMixtralAttention
from models.demos.t3000.mixtral8x7b.tt.mixtral_mlp import TtMixtralMLP
from models.demos.t3000.mixtral8x7b.tt.mixtral_rms_norm import TtRMSNormSharded
from models.demos.t3000.mixtral8x7b.tt.mixtral_moe import TtMoeLayer


class TtTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        devices,
        state_dict,
        args,
        layer_num,
        dtype,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)

        self.args = args
        self.hidden_size = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = args.max_seq_len
        self.dim = args.dim
        self.max_batch_size = args.max_batch_size
        self.n_kv_heads = args.n_kv_heads
        self.sliding_window = args.sliding_window

        self.layer_num = layer_num
        self.n_local_heads = self.n_heads // len(devices)
        self.n_local_kv_heads = self.n_kv_heads // len(devices)

        self.attention = TtMixtralAttention(
            devices=devices,
            state_dict=state_dict,
            args=args,
            layer_num=layer_num,
            dtype=dtype,
        )

        self.feed_forward = TtMoeLayer(
            devices=devices,
            state_dict=state_dict,
            experts=[
                TtMixtralMLP(
                    device=devices[i],
                    state_dict=state_dict,
                    args=args,
                    layer_num=layer_num,
                    expert_num=i,
                    dtypes={
                        "w1": ttnn.bfloat4_b,
                        "w2": ttnn.bfloat8_b,
                        "w3": ttnn.bfloat4_b,
                    },
                )
                for i in range(args.num_experts)
            ],
            args=args,
            layer_num=layer_num,
            dtype=dtype,
        )
        self.attention_norm = [
            TtRMSNormSharded(
                device=dev,
                state_dict=state_dict,
                args=args,
                dtype=ttnn.bfloat16,
                layer_num=layer_num,
                weight_key="attention_norm",
            )
            for dev in self.devices
        ]
        self.ffn_norm = [
            TtRMSNormSharded(
                device=dev,
                state_dict=state_dict,
                args=args,
                dtype=ttnn.bfloat16,
                layer_num=layer_num,
                weight_key="ffn_norm",
            )
            for dev in self.devices
        ]

    def forward(
        self,
        xs_1SBH,
        start_pos,
        current_pos,
        rot_mats,
    ) -> ttnn.Tensor:
        """
        Tensors are postfixed with 4 characters that represent their 4-D shape:
        B: batch dim (32)
        S: seq dim (1)
        1: unary dim
        H: hidden dim (4096)
        """
        assert isinstance(xs_1SBH, list)

        attn_norm_1SBH = [self.attention_norm[i](xs_1SBH[i]) for i in range(self.num_devices)]

        attn_1SBH = self.attention(
            attn_norm_1SBH,
            start_pos,
            current_pos,
            rot_mats,
        )
        hs_1SBH = [ttnn.add(xs_1SBH[i], attn_1SBH[i]) for i in range(self.num_devices)]

        ffn_norm_1SBH = [self.ffn_norm[i](hs_1SBH[i]) for i in range(self.num_devices)]
        ffn_1SBH = self.feed_forward(ffn_norm_1SBH)
        out_1SBH = [ttnn.add(hs_1SBH[i], ffn_1SBH[i]) for i in range(self.num_devices)]

        return out_1SBH
