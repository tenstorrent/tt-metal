# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
import torch
from ttnn.model_preprocessing import ParameterDict, fold_batch_norm2d_into_conv2d
from torch import nn
from ttnn.model_preprocessing import preprocess_linear_weight, preprocess_linear_bias
import math
import torch.nn.functional as F
from tt_lib.utils import (
    _nearest_y,
)
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores


class PETRMultiheadAttention:
    def __init__(self, device, torch_model, embed_dims=256, num_heads=8):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.device = device

        self.attn_in_proj__eight = torch_model.attn.in_proj_weight
        self.attn_in_proj__ias = torch_model.attn.in_proj_bias
        self.attn_out_proj_weight = torch_model.attn.out_proj.weight
        self.attn_out_proj_bias = torch_model.attn.out_proj.bias

    def __call__(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        batch_first=False,
    ):
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos

        if query_pos is not None:
            if query.get_layout() != ttnn.TILE_LAYOUT:
                query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
            if query_pos.get_layout() != ttnn.TILE_LAYOUT:
                query_pos = ttnn.to_layout(query_pos, ttnn.TILE_LAYOUT)
            query = query + query_pos
        if key_pos is not None:
            if key.get_layout() != ttnn.TILE_LAYOUT:
                key = ttnn.to_layout(key, ttnn.TILE_LAYOUT)
            if key_pos.get_layout() != ttnn.TILE_LAYOUT:
                key_pos = ttnn.to_layout(key_pos, ttnn.TILE_LAYOUT)
            key = key + key_pos

        if batch_first:
            query = ttnn.permute(query, (1, 0))
            key = ttnn.permute(key, (1, 0))
            value = ttnn.permute(value, (1, 0))

        in_proj_bias = self.attn_in_proj__ias
        in_proj_weight = self.attn_in_proj__eight

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        q_weight = in_proj_weight[: self.embed_dims, :]  # Query weights
        k_weight = in_proj_weight[self.embed_dims : 2 * self.embed_dims, :]  # Key weights
        v_weight = in_proj_weight[2 * self.embed_dims :, :]  # Value weights

        q_bias = in_proj_bias[: self.embed_dims]  # Query biases
        k_bias = in_proj_bias[self.embed_dims : 2 * self.embed_dims]  # Key biases
        v_bias = in_proj_bias[2 * self.embed_dims :]  # Value biases

        q_batch_size, q_sequence_size, q_hidden_size = query.shape
        q_head_size = q_hidden_size // self.num_heads

        k_batch_size, k_sequence_size, k_hidden_size = key.shape
        k_head_size = k_hidden_size // self.num_heads

        v_batch_size, v_sequence_size, v_hidden_size = value.shape
        v_head_size = v_hidden_size // self.num_heads

        q_weight = preprocess_linear_weight(q_weight, dtype=ttnn.bfloat16)
        q_bias = preprocess_linear_bias(q_bias, dtype=ttnn.bfloat16)

        k_weight = preprocess_linear_weight(k_weight, dtype=ttnn.bfloat16)
        k_bias = preprocess_linear_bias(k_bias, dtype=ttnn.bfloat16)

        v_weight = preprocess_linear_weight(v_weight, dtype=ttnn.bfloat16)
        v_bias = preprocess_linear_bias(v_bias, dtype=ttnn.bfloat16)

        q_weight = ttnn.to_device(q_weight, self.device)
        q_bias = ttnn.to_device(q_bias, self.device)

        k_weight = ttnn.to_device(k_weight, self.device)
        k_bias = ttnn.to_device(k_bias, self.device)

        v_weight = ttnn.to_device(v_weight, self.device)
        v_bias = ttnn.to_device(v_bias, self.device)

        query = ttnn.linear(query, q_weight, bias=q_bias)
        key = ttnn.linear(key, k_weight, bias=k_bias)

        if value.get_layout() != ttnn.TILE_LAYOUT:
            value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        value = ttnn.linear(value, v_weight, bias=v_bias)

        query = ttnn.reshape(query, (tgt_len, bsz * self.num_heads, q_head_size))
        query = ttnn.permute(query, (1, 0, 2))

        key = ttnn.reshape(key, (k_batch_size, bsz * self.num_heads, q_head_size))
        key = ttnn.permute(key, (1, 0, 2))

        value = ttnn.reshape(value, (v_batch_size, bsz * self.num_heads, q_head_size))
        value = ttnn.permute(value, (1, 0, 2))

        src_len = key.shape[1]

        B, Nt, E = query.shape
        q_scaled = query * math.sqrt(1.0 / float(E))
        key_transposed = ttnn.permute(key, (0, 2, 1))

        if attn_mask is not None:
            attn_output_weights = ttnn.matmul(q_scaled, key_transposed)
            attn_output_weights = attn_output_weights + attn_mask
        else:
            attn_output_weights = ttnn.matmul(q_scaled, key_transposed)

        attn_output_weights = ttnn.softmax(attn_output_weights, dim=-1)

        # attn_output_weights = torch.load("softmax_out.pt")
        # attn_output_weights = ttnn.from_torch(attn_output_weights, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device = self.device)

        attn_output = ttnn.matmul(attn_output_weights, value)

        attn_output = ttnn.permute(attn_output, (1, 0, 2))
        attn_output = ttnn.reshape(attn_output, (tgt_len * bsz, embed_dim))

        self.attn_out_proj_weight = preprocess_linear_weight(self.attn_out_proj_weight, dtype=ttnn.bfloat16)
        self.attn_out_proj_bias = preprocess_linear_bias(self.attn_out_proj_bias, dtype=ttnn.bfloat16)

        self.attn_out_proj_weight = ttnn.to_device(self.attn_out_proj_weight, self.device)
        self.attn_out_proj_bias = ttnn.to_device(self.attn_out_proj_bias, self.device)

        attn_output = ttnn.linear(attn_output, self.attn_out_proj_weight, bias=self.attn_out_proj_bias)
        attn_output = ttnn.reshape(attn_output, (tgt_len, bsz, attn_output.shape[1]))
        attn_output_weights = ttnn.reshape(attn_output_weights, (bsz, self.num_heads, tgt_len, src_len))
        attn_output_weights = ttnn.to_layout(attn_output_weights, ttnn.ROW_MAJOR_LAYOUT)
        attn_output_weights = ttnn.mean(attn_output_weights, dim=1)
        identity = ttnn.to_layout(identity, ttnn.TILE_LAYOUT)

        return attn_output + identity, attn_output_weights
