# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch
import math


class PETRMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dims,
        num_heads,
        attn_drop=0.1,
        proj_drop=0.0,
        batch_first=False,
        **kwargs,
    ):
        super(PETRMultiheadAttention, self).__init__()

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop, **kwargs)

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        dropout_p=0.1,
        **kwargs,
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
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        in_proj_bias = self.attn.in_proj_bias
        in_proj_weight = self.attn.in_proj_weight

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

        query = torch.nn.functional.linear(query, q_weight, bias=q_bias)
        key = torch.nn.functional.linear(key, k_weight, bias=k_bias)
        value = torch.nn.functional.linear(value, v_weight, bias=v_bias)

        query = torch.reshape(query, (tgt_len, bsz * self.num_heads, q_head_size)).transpose(0, 1)

        key = torch.reshape(key, (k_batch_size, bsz * self.num_heads, q_head_size)).transpose(0, 1)

        value = torch.reshape(value, (v_batch_size, bsz * self.num_heads, q_head_size)).transpose(0, 1)

        src_len = key.size(1)

        B, Nt, E = query.shape
        q_scaled = query * math.sqrt(1.0 / float(E))

        if attn_mask is not None:
            attn_output_weights = torch.baddbmm(attn_mask, q_scaled, key.transpose(-2, -1))
        else:
            attn_output_weights = torch.bmm(q_scaled, key.transpose(-2, -1))

        attn_output_weights = torch.nn.functional.softmax(attn_output_weights, dim=-1)

        attn_output = torch.bmm(attn_output_weights, value)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        attn_output = torch.nn.functional.linear(attn_output, self.attn.out_proj.weight, self.attn.out_proj.bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        # optionally average attention weights over heads
        attn_output_weights = torch.reshape(attn_output_weights, (bsz, self.num_heads, tgt_len, src_len))
        attn_output_weights = attn_output_weights.mean(dim=1)

        return attn_output + identity, attn_output_weights


class FFN(nn.Module):
    def __init__(self, in_features, out_features):
        super(FFN, self).__init__()
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Linear(in_features=in_features[0], out_features=out_features[0], bias=True),
                nn.ReLU(inplace=True),
            ),
            nn.Linear(in_features=in_features[1], out_features=out_features[1], bias=True),
        )

    def forward(self, x):
        input = x
        x = self.layers(x)
        return x + input


class PETRTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dims,
        num_heads,
        in_features,
        out_features,
        normalized_shape,
        attn_drop=[0.1, 0.1],
        proj_drop=[0.0, 0.0],
        batch_first=[False, False],
    ):
        super(PETRTransformerDecoderLayer, self).__init__()
        self.attentions = nn.ModuleList(
            [
                PETRMultiheadAttention(
                    embed_dims[0],
                    num_heads[0],
                    attn_drop[0],
                    proj_drop[0],
                    batch_first[0],
                ),
                PETRMultiheadAttention(
                    embed_dims[1],
                    num_heads[1],
                    attn_drop[1],
                    proj_drop[1],
                    batch_first[1],
                ),
            ]
        )
        self.ffns = nn.ModuleList([FFN(in_features, out_features)])
        self.norms = nn.ModuleList(
            [nn.LayerNorm(normalized_shape[0]), nn.LayerNorm(normalized_shape[1]), nn.LayerNorm(normalized_shape[2])]
        )

    def forward(self, query, key, value, key_pos, query_pos, key_padding_mask):
        x, weight = self.attentions[0](query, query, query, query_pos=query_pos, key_pos=query_pos)
        x = self.norms[0](x)

        x, weight = self.attentions[1](
            x, key, value, query_pos=query_pos, key_pos=key_pos, key_padding_mask=key_padding_mask
        )
        x = self.norms[1](x)

        x = self.ffns[0](x)
        x = self.norms[2](x)

        return x


class PETRTransformerDecoder(nn.Module):
    def __init__(
        self,
    ):
        super(PETRTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                PETRTransformerDecoderLayer(
                    [256, 256],
                    [8, 8],
                    [256, 2048],
                    [2048, 256],
                    [256, 256, 256],
                ),
                PETRTransformerDecoderLayer(
                    [256, 256],
                    [8, 8],
                    [256, 2048],
                    [2048, 256],
                    [256, 256, 256],
                ),
                PETRTransformerDecoderLayer(
                    [256, 256],
                    [8, 8],
                    [256, 2048],
                    [2048, 256],
                    [256, 256, 256],
                ),
                PETRTransformerDecoderLayer(
                    [256, 256],
                    [8, 8],
                    [256, 2048],
                    [2048, 256],
                    [256, 256, 256],
                ),
                PETRTransformerDecoderLayer(
                    [256, 256],
                    [8, 8],
                    [256, 2048],
                    [2048, 256],
                    [256, 256, 256],
                ),
                PETRTransformerDecoderLayer(
                    [256, 256],
                    [8, 8],
                    [256, 2048],
                    [2048, 256],
                    [256, 256, 256],
                ),
            ]
        )
        self.post_norm = nn.LayerNorm((256))

    def forward(self, query, key, value, key_pos, query_pos, key_padding_mask):
        x = self.layers[0](query, key, value, key_pos, query_pos, key_padding_mask)
        x1 = self.post_norm(x)
        x = self.layers[1](x, key, value, key_pos, query_pos, key_padding_mask)
        x2 = self.post_norm(x)
        x = self.layers[2](x, key, value, key_pos, query_pos, key_padding_mask)
        x3 = self.post_norm(x)
        x = self.layers[3](x, key, value, key_pos, query_pos, key_padding_mask)
        x4 = self.post_norm(x)
        x = self.layers[4](x, key, value, key_pos, query_pos, key_padding_mask)
        x5 = self.post_norm(x)
        x = self.layers[5](x, key, value, key_pos, query_pos, key_padding_mask)
        x6 = self.post_norm(x)
        x = torch.stack((x1, x2, x3, x4, x5, x6))
        return x


class PETRTransformer(nn.Module):
    def __init__(self):
        super(PETRTransformer, self).__init__()
        self.decoder = PETRTransformerDecoder()

    def forward(
        self,
        x,
        mask,
        query_embed,
        pos_embed,
    ):
        bs, n, c, h, w = x.shape
        memory = x.permute(1, 3, 4, 0, 2).reshape(-1, bs, c)  # [bs, n, c, h, w] -> [n*h*w, bs, c]
        pos_embed = pos_embed.permute(1, 3, 4, 0, 2).reshape(-1, bs, c)  # [bs, n, c, h, w] -> [n*h*w, bs, c]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
        mask = mask.view(bs, -1)  # [bs, n, h, w] -> [bs, n*h*w]
        target = torch.zeros_like(query_embed)

        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
        )
        out_dec = out_dec.transpose(1, 2)
        memory = memory.reshape(n, h, w, bs, c).permute(3, 0, 4, 1, 2)
        return out_dec, memory
