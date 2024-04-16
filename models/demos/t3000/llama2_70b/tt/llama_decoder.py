# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import tt_lib
import ttnn
from models.utility_functions import torch2tt_tensor, pad_by_zero, tt2torch_tensor, nearest_32
from models.demos.llama2_70b.tt.llama_attention import TtLlamaAttention
from models.demos.llama2_70b.tt.llama_attention_optimized import TtLlamaAttention_optimized

# from models.demos.llama2_70b.tt.llama_mlp import TtLlamaMLP
from models.demos.llama2_70b.tt.llama_mlp_optimized import TtLlamaMLP_optimized
from models.demos.llama2_70b.tt.llama_mlp import TtLlamaMLP
from models.demos.llama2_70b.tt.llama_common import tt_all_gather_torch


class TtLlamaDecoder:
    def __init__(self, devices, state_dict, base_url, layer_num, model_config, configuration, batch):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)
        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.n_local_heads = self.n_heads // self.num_devices
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = configuration.max_seq_len
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

        # self.attention = TtLlamaAttention_optimized(
        self.attention = TtLlamaAttention(devices, state_dict, base_url, layer_num, model_config, configuration)

        # self.mlp = TtLlamaMLP_optimized(
        self.mlp = TtLlamaMLP(
            devices,
            state_dict,
            base_url,
            layer_num,
            self.hidden_size,
            model_config,
        )

    def prepare_inputs(self, x, start_pos):
        # Only called by decoder tests
        assert x.size(2) == self.hidden_size
        assert len(x.size()) == 3

        batch = x.size(0)
        seq_len = x.size(1)
        assert seq_len == 1, "Only supporting decode mode"
        x = x.transpose(0, 1).unsqueeze(1)  # [seq_len, 1, batch, hidden_dim]

        position_ids = torch.ones(seq_len, batch, dtype=torch.long) * start_pos
        from models.demos.llama2_70b.tt.llama_common import generate_rot_emb, gather_rotary_emb

        rot_emb = generate_rot_emb(self.head_dim, self.max_seq_len * 2)
        rot_mat = gather_rotary_emb(rot_emb, position_ids)

        padded_layer_past_len = nearest_32(start_pos + 1)
        attn_mask = torch.zeros(seq_len, 1, batch, padded_layer_past_len)
        attn_mask[:, :, :, start_pos + 1 :] = torch.finfo(attn_mask.dtype).min
        attn_mask = attn_mask.expand(-1, self.n_local_heads, -1, -1)

        # expected shapes:
        # x: (seq_len, 1, batch, hidden_dim)
        # start_pos: int
        # rot_mat: [1, bsz, head_dim, head_dim]
        # attn_mask: [seq_len, n_heads, batch, padded_layer_past_len]
        assert x.size() == (seq_len, 1, batch, self.hidden_size)
        assert rot_mat.size() == (1, batch, self.head_dim, self.head_dim)
        assert attn_mask.size() == (seq_len, self.n_local_heads, batch, padded_layer_past_len)

        x_fractured = torch.chunk(x, self.num_devices, dim=-1)
        xs, rot_mats, attn_masks = [], [], []
        for i in range(self.num_devices):
            device = self.devices[i]
            xs.append(torch2tt_tensor(x_fractured[i], device))
            rot_mats.append(torch2tt_tensor(rot_mat.clone(), device))
            attn_masks.append(torch2tt_tensor(attn_mask.clone(), device))
        return (
            xs,
            start_pos,
            rot_mats,
            attn_masks,
        )

    def __call__(
        self,
        xs: list,
        rot_mats: list,
        start_pos: int,
        attn_masks: list,
    ) -> tt_lib.tensor.Tensor:
        ### xs (residual stream) is fractured on all chips

        ### Duplicate inputs for layernorm
        xs_replicated = tt_all_gather_torch(xs, dim=-1)

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
        attn_resid_replicated = tt_all_gather_torch(attn_resid_fractures, dim=-1)

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
