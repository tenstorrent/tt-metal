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


def input_preprocessing(x, N, C, H, W):
    x = ttnn.to_torch(x)
    x = torch.permute(x, (0, 3, 1, 2))
    x = x.reshape(N, C, H, W)
    return x


class FFN:
    def __init__(self, device, parameter):
        super().__init__()
        self.device = device
        self.l1_weight = parameter[0][0].weight
        self.l1_bias = parameter[0][0].bias
        self.l2_weight = parameter[1].weight
        self.l2_bias = parameter[1].bias

    def __call__(
        self,
        x,
    ):
        input = x

        self.l1_weight = preprocess_linear_weight(self.l1_weight, dtype=ttnn.bfloat16)
        self.l1_bias = preprocess_linear_bias(self.l1_bias, dtype=ttnn.bfloat16)
        self.l1_weight = ttnn.to_device(self.l1_weight, self.device)
        self.l1_bias = ttnn.to_device(self.l1_bias, self.device)

        x = ttnn.linear(x, self.l1_weight, bias=self.l1_bias)
        x = ttnn.relu(x)

        self.l2_weight = preprocess_linear_weight(self.l2_weight, dtype=ttnn.bfloat16)
        self.l2_bias = preprocess_linear_bias(self.l2_bias, dtype=ttnn.bfloat16)
        self.l2_weight = ttnn.to_device(self.l2_weight, self.device)
        self.l2_bias = ttnn.to_device(self.l2_bias, self.device)

        x = ttnn.linear(x, self.l2_weight, bias=self.l2_bias)

        return input + x


class PETRMultiheadAttention:
    def __init__(self, device, parameter, embed_dims=256, num_heads=8):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.device = device

        self.attn_in_proj__eight = parameter.attn.in_proj_weight
        self.attn_in_proj__ias = parameter.attn.in_proj_bias
        self.attn_out_proj_weight = parameter.attn.out_proj.weight
        self.attn_out_proj_bias = parameter.attn.out_proj.bias

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


class PETRTransformerDecoderLayer(nn.Module):
    def __init__(self, device, parameter):
        super().__init__()
        self.mha = PETRMultiheadAttention(device, parameter.attentions[0])
        self.petr_mha = PETRMultiheadAttention(device, parameter.attentions[1])
        self.ffns = FFN(device, parameter.ffns[0].layers)

        self.norm1_weight = ttnn.from_torch(parameter.norms[0].weight, layout=ttnn.TILE_LAYOUT, device=device)
        self.norm1_bias = ttnn.from_torch(parameter.norms[0].bias, layout=ttnn.TILE_LAYOUT, device=device)

        self.norm2_weight = ttnn.from_torch(parameter.norms[1].weight, layout=ttnn.TILE_LAYOUT, device=device)
        self.norm2_bias = ttnn.from_torch(parameter.norms[1].bias, layout=ttnn.TILE_LAYOUT, device=device)

        self.norm3_weight = ttnn.from_torch(parameter.norms[2].weight, layout=ttnn.TILE_LAYOUT, device=device)
        self.norm3_bias = ttnn.from_torch(parameter.norms[2].bias, layout=ttnn.TILE_LAYOUT, device=device)

    def __call__(self, device, query, key, value, key_pos, query_pos, key_padding_mask):
        x, weight = self.mha(query, query, query, query_pos=query_pos, key_pos=query_pos)
        x = ttnn.layer_norm(x, weight=self.norm1_weight, bias=self.norm1_bias)

        x, weight = self.petr_mha(
            x, key, value, query_pos=query_pos, key_pos=key_pos, key_padding_mask=key_padding_mask
        )
        x = ttnn.layer_norm(x, weight=self.norm2_weight, bias=self.norm2_bias)

        x = self.ffns(x)
        x = ttnn.layer_norm(x, weight=self.norm3_weight, bias=self.norm3_bias)

        return x


class PETRTransformerDecoder(nn.Module):
    def __init__(self, device, parameter):
        super().__init__()
        self.decoder0 = PETRTransformerDecoderLayer(device, parameter.decoder.module.layers[0])
        self.decoder1 = PETRTransformerDecoderLayer(device, parameter.decoder.module.layers[1])
        self.decoder2 = PETRTransformerDecoderLayer(device, parameter.decoder.module.layers[2])
        self.decoder3 = PETRTransformerDecoderLayer(device, parameter.decoder.module.layers[3])
        self.decoder4 = PETRTransformerDecoderLayer(device, parameter.decoder.module.layers[4])
        self.decoder5 = PETRTransformerDecoderLayer(device, parameter.decoder.module.layers[5])

        self.post_norm_weight = ttnn.from_torch(
            parameter.decoder.module.post_norm.weight, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.post_norm_bias = ttnn.from_torch(
            parameter.decoder.module.post_norm.bias, layout=ttnn.TILE_LAYOUT, device=device
        )

    def __call__(self, device, query, key, value, key_pos, query_pos, key_padding_mask):
        x = self.decoder0(device, query, key, value, key_pos, query_pos, key_padding_mask)
        x1 = ttnn.layer_norm(x, weight=self.post_norm_weight, bias=self.post_norm_bias)

        x = self.decoder1(device, x, key, value, key_pos, query_pos, key_padding_mask)
        x2 = ttnn.layer_norm(x, weight=self.post_norm_weight, bias=self.post_norm_bias)

        x = self.decoder2(device, x, key, value, key_pos, query_pos, key_padding_mask)
        x3 = ttnn.layer_norm(x, weight=self.post_norm_weight, bias=self.post_norm_bias)

        x = self.decoder3(device, x, key, value, key_pos, query_pos, key_padding_mask)
        x4 = ttnn.layer_norm(x, weight=self.post_norm_weight, bias=self.post_norm_bias)

        x = self.decoder4(device, x, key, value, key_pos, query_pos, key_padding_mask)
        x5 = ttnn.layer_norm(x, weight=self.post_norm_weight, bias=self.post_norm_bias)

        x = self.decoder5(device, x, key, value, key_pos, query_pos, key_padding_mask)
        x6 = ttnn.layer_norm(x, weight=self.post_norm_weight, bias=self.post_norm_bias)

        x1 = ttnn.reshape(x1, (1, x1.shape[0], x1.shape[1], x1.shape[2]))
        x2 = ttnn.reshape(x2, (1, x2.shape[0], x2.shape[1], x2.shape[2]))
        x3 = ttnn.reshape(x3, (1, x3.shape[0], x3.shape[1], x3.shape[2]))
        x4 = ttnn.reshape(x4, (1, x4.shape[0], x4.shape[1], x4.shape[2]))
        x5 = ttnn.reshape(x5, (1, x5.shape[0], x5.shape[1], x5.shape[2]))
        x6 = ttnn.reshape(x6, (1, x6.shape[0], x6.shape[1], x6.shape[2]))

        x = ttnn.concat([x1, x2, x3, x4, x5, x6], dim=0)
        return x


class PETRTransformer:
    def __init__(self, device, parameter):
        super().__init__()
        self.decoder = PETRTransformerDecoder(device, parameter)

    def __call__(
        self,
        device,
        x,
        mask,
        query_embed,
        pos_embed,
    ):
        bs, n, c, h, w = x.shape
        memory = ttnn.to_torch(x)
        memory = memory.permute(1, 3, 4, 0, 2)
        memory = ttnn.from_torch(memory, dtype=ttnn.bfloat16, device=device)
        memory = ttnn.reshape(memory, (-1, bs, c))

        pos_embed = ttnn.to_torch(pos_embed)
        pos_embed = pos_embed.permute(1, 3, 4, 0, 2)
        pos_embed = ttnn.from_torch(pos_embed, dtype=ttnn.bfloat16, device=device)
        pos_embed = ttnn.reshape(pos_embed, (-1, bs, c))

        query_embed = ttnn.reshape(query_embed, (query_embed.shape[0], 1, query_embed.shape[1]))
        query_embed = ttnn.repeat(query_embed, ttnn.Shape([1, bs, 1]))

        mask = ttnn.reshape(mask, (bs, -1))

        target = torch.zeros((query_embed.shape[0], query_embed.shape[1], query_embed.shape[2]))

        target = ttnn.from_torch(target, dtype=ttnn.bfloat16, device=device)

        out_dec = self.decoder(
            device,
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
        )

        out_dec = ttnn.permute(out_dec, (0, 2, 1, 3))
        memory = ttnn.reshape(memory, (n, h, w, bs, c))
        memory = ttnn.to_torch(memory)
        memory = memory.permute(3, 0, 4, 1, 2)

        return out_dec, memory
