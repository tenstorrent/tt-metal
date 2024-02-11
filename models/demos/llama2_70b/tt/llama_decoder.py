# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import tt_lib
import ttnn
from models.utility_functions import torch2tt_tensor, pad_by_zero, tt2torch_tensor
from models.demos.llama2_70b.tt.llama_attention import TtLlamaAttention
from models.demos.llama2_70b.tt.llama_mlp import TtLlamaMLP


class TtLlamaDecoder(nn.Module):
    def __init__(self, device, state_dict, base_url, layer_num, model_config, configuration, batch):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.hidden_size = configuration.dim
        self.model_config = model_config

        layer_name = f"{base_url}.{layer_num}"

        attn_norm_str = f"{layer_name}.attention_norm.weight"
        ffn_norm_str = f"{layer_name}.ffn_norm.weight"

        self.norm_eps = configuration.norm_eps

        self.attn_norm = torch2tt_tensor(
            # Expand to size of input since we decomped norm
            self.state_dict[attn_norm_str].unsqueeze(0).expand(batch, -1),
            self.device,
        )

        self.ffn_norm = torch2tt_tensor(
            # Expand to size of input since we decomped norm
            self.state_dict[ffn_norm_str].unsqueeze(0).expand(batch, -1),
            self.device,
        )

        self.attention = TtLlamaAttention(device, state_dict, base_url, layer_num, model_config, configuration)

        self.mlp = TtLlamaMLP(
            device,
            state_dict,
            base_url,
            layer_num,
            self.hidden_size,
            model_config,
        )

    def rms_decomp(self, x, norm_weight):
        squared = tt_lib.tensor.pow(x, 2)
        # mean_squared = tt_lib.tensor.mean(squared, )
        sum_squared = tt_lib.tensor.reduce(
            squared, tt_lib.tensor.ReduceOpMath.SUM, tt_lib.tensor.ReduceOpDim.W, scaler=1.0
        )
        # Tensor is 1,1,32,1+31 now
        mean_squared = tt_lib.tensor.div_unary(sum_squared, x.shape()[-1])
        mean_squared_eps = tt_lib.tensor.add_unary(mean_squared, self.norm_eps)
        rms = tt_lib.tensor.pow(mean_squared_eps, 0.5)
        rms_recip = tt_lib.tensor.recip(rms)
        normed_x = tt_lib.tensor.bcast(
            x, rms_recip, math_op=tt_lib.tensor.BcastOpMath.MUL, dim=tt_lib.tensor.BcastOpDim.W
        )
        norm_out = tt_lib.tensor.mul(normed_x, norm_weight)
        return norm_out

    def prepare_inputs(self, x, start_pos):
        # Pass through to attention layer
        return self.attention.prepare_inputs(x, start_pos)

    def forward(
        self,
        x: tt_lib.tensor.Tensor,
        rot_mat: tt_lib.tensor.Tensor,
        start_pos: int,
        attn_mask: tt_lib.tensor.Tensor,
    ) -> tt_lib.tensor.Tensor:
        """
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
        """

        # tt_lib.tensor.rmsnorm(hidden_states, self.variance_epsilon, self.weight)

        x_attn_norm = self.rms_decomp(x, self.attn_norm)
        attn_out = self.attention(x_attn_norm, rot_mat, start_pos, attn_mask)

        attn_resid = tt_lib.tensor.add(attn_out, x)

        # x_ffn_norm = tt_lib.tensor.rmsnorm(
        #     attn_resid, self.norm_eps, self.ffn_norm
        # )
        x_ffn_norm = self.rms_decomp(attn_resid, self.ffn_norm)
        ffn_out = self.mlp(x_ffn_norm)
        ffn_resid = tt_lib.tensor.add(ffn_out, attn_resid)

        return ffn_resid
