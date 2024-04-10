# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import tt_lib
import ttnn
from models.utility_functions import torch2tt_tensor, pad_by_zero, tt2torch_tensor, nearest_32
from models.demos.llama2_70b.tt.llama_decoder import TtLlamaDecoder
from models.demos.llama2_70b.tt.llama_common import generate_rot_emb, gather_rotary_emb, rms_decomp, tt_all_gather_torch


class TtLlamaModel(nn.Module):
    def __init__(self, devices, state_dict, base_url, n_layers, model_config, configuration, batch):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)
        self.model_config = model_config

        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = configuration.max_seq_len
        self.n_kv_heads = configuration.n_kv_heads
        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices

        emb_str = "tok_embeddings.weight"

        norm_str = "norm.weight"
        lm_head_str = "output.weight"

        self.norm_eps = configuration.norm_eps

        self.tok_embeddings = torch.nn.Embedding(configuration.vocab_size, self.hidden_size)
        self.tok_embeddings.weight = torch.nn.Parameter(self.state_dict[emb_str])

        self.norm_list = []
        for i in range(self.num_devices):
            output_norm = tt_lib.tensor.Tensor(
                # Expand to size of input since we decomped norm
                self.state_dict[norm_str].reshape([1, 1, -1, 32]),
                self.model_config["LN_F_WEIGHTS_DTYPE"],
            ).to(devices[i], self.model_config["LN_F_WEIGHTS_MEMCFG"])

            self.norm_list.append(output_norm)

        self.layers = [
            TtLlamaDecoder(
                devices,
                state_dict,
                base_url,
                i,
                model_config,
                configuration,
                batch,
            )
            for i in range(n_layers)
        ]

        self.lm_head_list = []
        for i in range(self.num_devices):
            lm_head = torch2tt_tensor(
                torch.chunk(torch.transpose(self.state_dict[lm_head_str], -2, -1), self.num_devices, -1)[i],
                self.devices[i],
            )
            self.lm_head_list.append(lm_head)

        self.rot_emb = generate_rot_emb(self.head_dim, self.max_seq_len * 2)

    def prepare_inputs(self, inp_ids, start_pos):
        """
        Prepare inputs for decode mode. Assume that current token is at
        start_pos, and KV cache has valid data up to start_pos.
        x: (batch, seq)
        start_pos: int

        returns
        xs: [(seq, batch, hidden_dim)] * num_devices
        start_pos: int
        rot_mats: [(1, batch, head_dim, head_dim)] * num_devices
        attn_masks: [(seq, n_local_heads, batch, max_seq_len)] * num_devices
        """
        x = self.tok_embeddings(inp_ids)  # [batch, seq, hidden]
        assert x.size(2) == self.hidden_size
        assert len(x.size()) == 3

        batch = x.size(0)
        seq_len = x.size(1)
        assert seq_len == 1, "Only supporting decode mode"
        x = x.transpose(0, 1).unsqueeze(1)  # [seq_len, 1, batch, hidden_dim]

        position_ids = torch.ones(seq_len, batch, dtype=torch.long) * start_pos
        rot_mat = gather_rotary_emb(self.rot_emb, position_ids)

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

    def forward(
        self,
        xs: tt_lib.tensor.Tensor,
        rot_mats: tt_lib.tensor.Tensor,
        start_pos: int,
        attn_masks: tt_lib.tensor.Tensor,
    ) -> tt_lib.tensor.Tensor:
        """
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output
        """

        ### Run all layers
        for layer in self.layers:
            xs = layer(xs, rot_mats, start_pos, attn_masks)

        ### Gather fractured layers output
        xs_replicated = tt_all_gather_torch(xs, dim=-1)

        ### Duplicate layernorm
        norm_out_replicated = []
        for i in range(self.num_devices):
            # TODO: Delete xs after consuming by rmsnorm
            x = xs_replicated[i]
            # RMSNorm must execute on sharded input
            x = tt_lib.tensor.interleaved_to_sharded(
                x, sharded_mem_config=self.model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"]
            )
            x_norm = tt_lib.operations.primary.rmsnorm(
                x,
                self.norm_eps,
                self.norm_list[i],
                output_mem_config=self.model_config["LN_F_OUTPUT_MEMCFG"],
                program_config=self.model_config["LN_F_PROGCFG"],
            )
            # Spill input and output back to DRAM
            x = tt_lib.tensor.sharded_to_interleaved(x, output_mem_config=self.model_config["DEFAULT_MEMCFG"])
            x_norm = tt_lib.tensor.sharded_to_interleaved(x_norm, output_mem_config=self.model_config["DEFAULT_MEMCFG"])
            norm_out_replicated.append(x_norm)

        ### Each device does an LM head fracture
        lm_head_out = []
        for i in range(self.num_devices):
            # out = tt_lib.tensor.matmul(norm_out_replicated[i], self.lm_head_list[i])
            out = tt_lib.operations.primary.matmul_1d(
                norm_out_replicated[i], self.lm_head_list[i], fp32_dest_acc_en=True, packer_l1_acc=True
            )
            lm_head_out.append(out)

        ret = tt_all_gather_torch(lm_head_out, dim=-1)
        ret = ret[0]
        ### Concat LM head results to return 1 tensor
        # ret = tt_lib.tensor.concat(lm_head_out, dim=-1)
        return ret
