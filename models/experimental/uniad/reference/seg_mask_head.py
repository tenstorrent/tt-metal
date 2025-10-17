# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import copy
from torch import Tensor
from typing import Optional


class AttentionTail(nn.Module):
    def __init__(self, cfg, dim, num_heads=2, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_l1 = nn.Sequential(
            nn.Linear(self.num_heads, self.num_heads),
            nn.ReLU(),
        )

        self.linear = nn.Sequential(
            nn.Linear(self.num_heads, 1),
            nn.ReLU(),
        )

    def forward(self, query, key, key_padding_mask, hw_lvl=None):
        B, N, C = query.shape
        _, L, _ = key.shape
        q = self.q(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        k = self.k(key).reshape(B, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale

        attn = attn.permute(0, 2, 3, 1)

        new_feats = self.linear_l1(attn)
        mask = self.linear(new_feats)

        return mask


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, cfg, dim, num_heads=2, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class Attention(nn.Module):
    def __init__(self, cfg, dim, num_heads=2, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.linear_l1 = nn.Sequential(
            nn.Linear(self.num_heads, self.num_heads),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(self.num_heads, 1),
            nn.ReLU(),
        )

    def forward(self, query, key, value, key_padding_mask, hw_lvl):
        B, N, C = query.shape
        _, L, _ = key.shape
        q = (
            self.q(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        )  # .permute(2, 0, 3, 1, 4)
        k = (
            self.k(key).reshape(B, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        )  # .permute(2, 0, 3, 1, 4)

        v = (
            self.v(value).reshape(B, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        )  # .permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale

        attn = attn.permute(0, 2, 3, 1)

        new_feats = self.linear_l1(attn)
        mask = self.linear(new_feats)

        attn = attn.permute(0, 3, 1, 2)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)

        return x, mask


class Block(nn.Module):
    def __init__(
        self,
        cfg,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        self_attn=False,
    ):
        super().__init__()
        self.head_norm1 = norm_layer(dim)
        self.self_attn = self_attn
        self.attn = Attention(cfg, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)

        self.head_norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        if self.self_attn:
            self.self_attention = SelfAttention(
                cfg,
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
            )
            self.norm3 = norm_layer(dim)

    def forward(self, query, key, value, key_padding_mask=None, hw_lvl=None):
        if self.self_attn:
            query = query + self.self_attention(query)
            query = self.norm3(query)
        x, mask = self.attn(query, key, value, key_padding_mask, hw_lvl=hw_lvl)
        query = query + x
        query = self.head_norm1(query)

        query = query + self.mlp(query)
        query = self.head_norm2(query)
        return query, mask


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class SegMaskHead(nn.Module):
    def __init__(
        self,
        cfg=None,
        d_model=16,
        nhead=2,
        num_encoder_layers=6,
        num_decoder_layers=1,
        dim_feedforward=64,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        self_attn=False,
    ):
        super().__init__()

        mlp_ratio = 4
        qkv_bias = True
        qk_scale = None
        block = Block(
            cfg,
            dim=d_model,
            num_heads=nhead,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            self_attn=self_attn,
        )
        self.blocks = _get_clones(block, num_decoder_layers)
        self.attnen = AttentionTail(
            cfg,
            d_model,
            num_heads=nhead,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
        )

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        if pos is None:
            return tensor
        else:
            return tensor + pos

    def forward(self, memory, mask_memory, pos_memory, query_embed, mask_query, pos_query, hw_lvl):
        if mask_memory is not None and isinstance(mask_memory, torch.Tensor):
            mask_memory = mask_memory.to(torch.bool)
        masks = []
        inter_query = []
        for i, block in enumerate(self.blocks):
            query_embed, mask = block(
                self.with_pos_embed(query_embed, pos_query),
                self.with_pos_embed(memory, pos_memory),
                memory,
                key_padding_mask=mask_memory,
                hw_lvl=hw_lvl,
            )
            masks.append(mask)
            inter_query.append(query_embed)
        attn = self.attnen(
            self.with_pos_embed(query_embed, pos_query),
            self.with_pos_embed(memory, pos_memory),
            key_padding_mask=mask_memory,
            hw_lvl=hw_lvl,
        )
        return attn, masks, inter_query
