# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import tt_lib
import ttnn
from models.utility_functions import torch2tt_tensor, pad_by_zero, tt2torch_tensor
from models.demos.llama2_70b.tt.llama_attention import TtLlamaAttention
from models.demos.llama2_70b.tt.llama_mlp import TtLlamaMLP
from models.demos.llama2_70b.tt.llama_common import rms_decomp


class TtLlamaDecoder(nn.Module):
    def __init__(self, devices, state_dict, base_url, layer_num, model_config, configuration, batch):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)
        self.hidden_size = configuration.dim
        self.model_config = model_config

        layer_name = f"{base_url}.{layer_num}"

        attn_norm_str = f"{layer_name}.attention_norm.weight"
        ffn_norm_str = f"{layer_name}.ffn_norm.weight"

        self.norm_eps = configuration.norm_eps

        self.attn_norm_list = []
        self.ffn_norm_list = []
        for i in range(self.num_devices):
            attn_norm = torch2tt_tensor(
                # Expand to size of input since we decomped norm
                self.state_dict[attn_norm_str].unsqueeze(0).expand(batch, -1),
                self.devices[i],
            )
            ffn_norm = torch2tt_tensor(
                # Expand to size of input since we decomped norm
                self.state_dict[ffn_norm_str].unsqueeze(0).expand(batch, -1),
                self.devices[i],
            )
            self.attn_norm_list.append(attn_norm)
            self.ffn_norm_list.append(ffn_norm)

        self.attention = TtLlamaAttention(devices, state_dict, base_url, layer_num, model_config, configuration)

        self.mlp = TtLlamaMLP(
            devices,
            state_dict,
            base_url,
            layer_num,
            self.hidden_size,
            model_config,
        )

    def prepare_inputs(self, x, start_pos):
        # Pass through to attention layer
        return self.attention.prepare_inputs(x, start_pos)

    def forward(
        self,
        xs: list,
        rot_mats: list,
        start_pos: int,
        attn_masks: list,
    ) -> tt_lib.tensor.Tensor:
        ### Duplicate layernorm
        attn_norm_replicated = []
        for i in range(self.num_devices):
            x = xs[i]
            x_attn_norm = rms_decomp(x, self.attn_norm_list[i], self.norm_eps)
            attn_norm_replicated.append(x_attn_norm)

        attn_outs = self.attention(attn_norm_replicated, rot_mats, start_pos, attn_masks)

        ### Duplicate residual
        attn_resid_replicated = []
        for i in range(self.num_devices):
            attn_resid = tt_lib.tensor.add(attn_outs[i], xs[i])
            attn_resid_replicated.append(attn_resid)

        ### Duplicate layernorm
        ffn_norm_replicated = []
        for i in range(self.num_devices):
            x = attn_resid_replicated[i]
            x_ffn_norm = rms_decomp(x, self.ffn_norm_list[i], self.norm_eps)
            ffn_norm_replicated.append(x_ffn_norm)

        ffn_out = self.mlp(ffn_norm_replicated)

        ### Duplicate residual
        ffn_resid_replicated = []
        for i in range(self.num_devices):
            ffn_resid = tt_lib.tensor.add(ffn_out[i], attn_resid_replicated[i])
            ffn_resid_replicated.append(ffn_resid)

        return ffn_resid_replicated
