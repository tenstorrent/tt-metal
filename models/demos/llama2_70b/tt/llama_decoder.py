# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import tt_lib
import ttnn
from models.utility_functions import torch2tt_tensor, pad_by_zero, tt2torch_tensor
from models.demos.llama2_70b.tt.llama_attention import TtLlamaAttention

# from models.demos.llama2_70b.tt.llama_mlp import TtLlamaMLP
from models.demos.llama2_70b.tt.llama_mlp_optimized import TtLlamaMLP_optimized
from models.demos.llama2_70b.tt.llama_common import tt_all_gather


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
            attn_norm = tt_lib.tensor.Tensor(
                # Expand to size of input since we decomped norm
                self.state_dict[attn_norm_str].reshape([1, 1, -1, 32]),
                self.model_config["LN_ATTN_WEIGHTS_DTYPE"],
            ).to(devices[i], self.model_config["LN_ATTN_WEIGHTS_MEMCFG"])

            ffn_norm = tt_lib.tensor.Tensor(
                # Expand to size of input since we decomped norm
                self.state_dict[ffn_norm_str].reshape([1, 1, -1, 32]),
                self.model_config["LN_MLP_WEIGHTS_DTYPE"],
            ).to(devices[i], self.model_config["LN_MLP_WEIGHTS_MEMCFG"])

            self.attn_norm_list.append(attn_norm)
            self.ffn_norm_list.append(ffn_norm)

        self.attention = TtLlamaAttention(devices, state_dict, base_url, layer_num, model_config, configuration)

        self.mlp = TtLlamaMLP_optimized(
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
        ### xs (residual stream) is fractured on all chips

        ### Duplicate inputs for layernorm
        xs_replicated = tt_all_gather(xs, dim=-1)

        attn_norm_replicated = []
        for i in range(self.num_devices):
            x = xs_replicated[i]
            # RMSNorm must execute on sharded input
            x = tt_lib.tensor.interleaved_to_sharded(
                x, sharded_mem_config=self.model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"]
            )
            x_attn_norm = tt_lib.operations.primary.rmsnorm(
                x,
                self.norm_eps,
                self.attn_norm_list[i],
                output_mem_config=self.model_config["LN_ATTN_OUTPUT_MEMCFG"],
                program_config=self.model_config["LN_ATTN_PROGCFG"],
            )
            # Spill input and output back to DRAM
            x = tt_lib.tensor.sharded_to_interleaved(x, output_mem_config=self.model_config["DEFAULT_MEMCFG"])
            x_attn_norm = tt_lib.tensor.sharded_to_interleaved(
                x_attn_norm, output_mem_config=self.model_config["DEFAULT_MEMCFG"]
            )
            attn_norm_replicated.append(x_attn_norm)

        # attn_outs is fractured
        attn_outs = self.attention(attn_norm_replicated, rot_mats, start_pos, attn_masks)

        ### Fractured residual add
        attn_resid_fractures = []
        for i in range(self.num_devices):
            attn_resid = tt_lib.tensor.add(attn_outs[i], xs[i])
            attn_resid_fractures.append(attn_resid)

        ### Duplicate attention residual on all chips
        attn_resid_replicated = tt_all_gather(attn_resid_fractures, dim=-1)

        ### Duplicate layernorm
        ffn_norm_replicated = []
        for i in range(self.num_devices):
            x = attn_resid_replicated[i]
            # RMSNorm must execute on sharded input
            x = tt_lib.tensor.interleaved_to_sharded(
                x, sharded_mem_config=self.model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"]
            )
            x_ffn_norm = tt_lib.operations.primary.rmsnorm(
                x,
                self.norm_eps,
                self.ffn_norm_list[i],
                output_mem_config=self.model_config["LN_MLP_OUTPUT_MEMCFG"],
                program_config=self.model_config["LN_MLP_PROGCFG"],
            )
            # Spill input and output back to DRAM
            x = tt_lib.tensor.sharded_to_interleaved(x, output_mem_config=self.model_config["DEFAULT_MEMCFG"])
            x_ffn_norm = tt_lib.tensor.sharded_to_interleaved(
                x_ffn_norm, output_mem_config=self.model_config["DEFAULT_MEMCFG"]
            )
            ffn_norm_replicated.append(x_ffn_norm)

        ffn_out = self.mlp(ffn_norm_replicated)

        ### Duplicate residual
        ffn_resid_fractured = []
        for i in range(self.num_devices):
            ffn_resid = tt_lib.tensor.add(ffn_out[i], attn_resid_fractures[i])
            ffn_resid_fractured.append(ffn_resid)

        return ffn_resid_fractured
