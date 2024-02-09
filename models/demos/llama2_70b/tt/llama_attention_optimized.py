# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import math
import torch
from torch import nn
import tt_lib
import ttnn
from models.utility_functions import torch2tt_tensor
from models.demos.llama2_70b.tt.llama_common import (
    precompute_freqs as tt_precompute_freqs,
    freqs_to_rotation_matrix,
    gather_rotary_emb as tt_gather_rotary_emb,
)


class TtLlamaAttention(nn.Module):
    def __init__(self, device, state_dict, base_url, layer_num, model_config, layer_past, configuration):
        super().__init__()

        self.state_dict = state_dict
        self.device = device

        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = configuration.max_seq_len
        self.n_kv_heads = configuration.n_kv_heads

        self.model_config = model_config

        layer_name = f"{base_url}.{layer_num}"

        wq_str = f"{layer_name}.attention.wq.weight"
        wk_str = f"{layer_name}.attention.wk.weight"
        wv_str = f"{layer_name}.attention.wv.weight"
        wo_str = f"{layer_name}.attention.wo.weight"

        self.wq = torch2tt_tensor(
            torch.transpose(
                self.state_dict[wq_str],
                -2,
                -1,
            ),
            self.device,
        )
        self.wk = torch2tt_tensor(
            torch.transpose(
                self.state_dict[wk_str],
                -2,
                -1,
            ),
            self.device,
        )
        self.wv = torch2tt_tensor(
            torch.transpose(
                self.state_dict[wv_str],
                -2,
                -1,
            ),
            self.device,
        )
        self.wo = torch2tt_tensor(
            torch.transpose(
                self.state_dict[wo_str],
                -2,
                -1,
            ),
            self.device,
        )

        layer_past = [torch2tt_tensor(lp, device) for lp in layer_past]
        self.layer_past = layer_past

    def get_rotation_mat(self, dhead, end, start_pos, seqlen, batch):
        cos, sin = tt_precompute_freqs(dhead, end)
        rot_mat = freqs_to_rotation_matrix(cos, sin)
        position_ids = torch.ones(seqlen, batch, dtype=torch.long) * start_pos
        rot_emb = tt_gather_rotary_emb(rot_mat, position_ids)
        return rot_emb

    def prepare_inputs(self, x, start_pos):
        """
        Prepare inputs for decode mode. Assume that current token is at
        start_pos, and KV cache has valid data up to start_pos.
        x: (batch, seq, hidden_dim)
        start_pos: int
        """
        assert x.size(2) == self.hidden_size
        assert len(x.size()) == 3

        batch = x.size(0)
        seq_len = x.size(1)
        x = x.transpose(0, 1).unsqueeze(1)  # [seq_len, 1, batch, hidden_dim]
        rot_mat = self.get_rotation_mat(
            dhead=self.head_dim, end=self.max_seq_len * 2, start_pos=start_pos, seqlen=seq_len, batch=batch
        )

        attn_mask = torch.zeros(seq_len, 1, batch, self.max_seq_len)
        # attn_mask[:, :, :, : start_pos + 1] = -1e9
        attn_mask[:, :, :, start_pos + 1 :] = torch.finfo(attn_mask.dtype).min
        attn_mask = attn_mask.expand(-1, self.n_heads, -1, -1)

        # expected shapes:
        # x: (seq_len, 1, batch, hidden_dim)
        # start_pos: int
        # rot_mat: [1, bsz, head_dim, head_dim]
        # attn_mask: [seq_len, n_heads, batch, self.max_seq_len]
        assert x.size() == (seq_len, 1, batch, self.hidden_size)
        assert rot_mat.size() == (1, batch, self.head_dim, self.head_dim)
        assert attn_mask.size() == (seq_len, self.n_heads, batch, self.max_seq_len)

        device = self.device
        return (
            torch2tt_tensor(x, device),
            start_pos,
            torch2tt_tensor(rot_mat, device),
            torch2tt_tensor(attn_mask, device),
        )

    def forward(
        self,
        x: tt_lib.tensor.Tensor,
        rot_mat: tt_lib.tensor.Tensor,
        start_pos: int,
        attn_mask: tt_lib.tensor.Tensor,
    ) -> tt_lib.tensor.Tensor:
        """
        x: (seq_len, 1, batch, hidden_dim)
        rot_mat: ???
        start_pos: the length of the KV cache. Same as current token's index.
        attn_mask: (seq_len, n_heads, batch, cache_len + seqlen
        """

        ###
        # QKV matmuls
        ###
        xq = tt_lib.tensor.matmul(
            x,
            self.wq,
        )

        xk = tt_lib.tensor.matmul(
            x,
            self.wk,
        )

        xv = tt_lib.tensor.matmul(
            x,
            self.wv,
        )

        ###
        # Reshape and rotary embeddings
        ###

        xqkv_fused = tt_lib.tensor.concat([xq, xk, xv], dim=-1)
        (
            q_heads,  # [seqlen, n_heads, bsz, head_dim]
            k_heads,  # [seqlen, n_kv_heads, bsz, head_dim]
            v_heads,  # [seqlen, n_kv_heads, bsz, head_dim]
        ) = tt_lib.tensor.nlp_create_qkv_heads(
            xqkv_fused, num_heads=self.n_heads, num_kv_heads=self.n_kv_heads, transpose_k_heads=False
        )

        # Have to put bsz back in dim 1 to match rot_mat shape
        q_heads = tt_lib.tensor.transpose(q_heads, 1, 2)
        k_heads = tt_lib.tensor.transpose(k_heads, 1, 2)

        q_heads = tt_lib.tensor.bmm(
            q_heads, rot_mat  # [seqlen, bsz, n_heads, head_dim]  # [1, bsz, head_dim, head_dim]
        )
        k_heads = tt_lib.tensor.bmm(
            k_heads, rot_mat  # [seqlen, bsz, n_kv_heads, head_dim]  # [1, bsz, head_dim, head_dim]
        )

        q_heads = tt_lib.tensor.transpose(q_heads, 1, 2)
        k_heads = tt_lib.tensor.transpose(k_heads, 1, 2)

        ###
        # KV update
        ###
        keys = self.layer_past[0]
        values = self.layer_past[1]
        tt_lib.tensor.update_cache(keys, k_heads, start_pos)
        tt_lib.tensor.update_cache(values, v_heads, start_pos)

        ###
        # Attention
        ###
        keys = tt_lib.tensor.transpose(keys, -1, -2)  #  [batch, num_kv_heads, dhead, cache_len + seqlen]
        attn = tt_lib.operations.primary.transformers.group_attn_matmul(
            q_heads,
            keys,
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            # output_mem_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
            # output_dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
        )  # seqlen, n_heads, batch, cache_len + seqlen

        scale = 1 / math.sqrt(self.head_dim)
        attn = tt_lib.tensor.mul_unary(attn, scale)
        attn = tt_lib.tensor.add(attn, attn_mask)
        attn = tt_lib.tensor.softmax(attn)

        attn_output = tt_lib.operations.primary.transformers.group_attn_matmul(
            attn,
            values,
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            # output_mem_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
            # output_dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
        )  # seqlen, n_heads, batch, dhead

        attn_output = tt_lib.tensor.nlp_concat_heads(
            attn_output,
            # output_mem_config=self.model_config["CONCAT_HEADS_OUTPUT_MEMCFG"],
        )  # seqlen, 1, batch, hidden_size

        dense_out = tt_lib.tensor.matmul(
            attn_output,
            self.wo,
        )  # seqlen, 1, batch, hidden_size

        return dense_out
