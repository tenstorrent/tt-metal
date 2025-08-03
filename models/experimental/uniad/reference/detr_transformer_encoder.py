# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from models.experimental.uniad.reference.utils import multi_scale_deformable_attn_pytorch
from models.experimental.uniad.reference.ffn import FFN

import warnings
from typing import Optional


class MultiScaleDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        im2col_step: int = 64,
        dropout: float = 0.1,
        batch_first: bool = False,
        norm_cfg=None,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, " f"but got {embed_dims} and {num_heads}")
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                "MultiScaleDeformAttention to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_levels, self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be" f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations, attention_weights, self.im2col_step
            )
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, level_start_index, sampling_locations, attention_weights, self.im2col_step
            )

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


class DetrTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers=6,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        im2col_step=64,
        dropout=0.1,
        post_norm=False,
        feedforward_channels=512,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            attentions = nn.ModuleList(
                [MultiScaleDeformableAttention(embed_dims, num_heads, num_levels, num_points, dropout)]
            )
            ffns = nn.ModuleList([FFN(embed_dims)])
            norms = nn.ModuleList([nn.LayerNorm(embed_dims), nn.LayerNorm(embed_dims)])
            layer = nn.ModuleDict({"attentions": attentions, "ffns": ffns, "norms": norms})
            self.layers.append(layer)

        self.post_norm = False
        self.embed_dims = embed_dims

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_padding_mask=None,
        attn_masks=None,
        query_key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        for i, layer in enumerate(self.layers):
            temp_key = temp_value = query
            query = layer["attentions"][0](
                query,
                temp_key,
                temp_value,
                identity=None,
                query_pos=query_pos,
                key_pos=query_pos,
                attn_mask=None,
                spatial_shapes=spatial_shapes,
                reference_points=reference_points,
                key_padding_mask=query_key_padding_mask,
                **kwargs,
            )
            identity = query

            query = layer["norms"][0](query)

            query = layer["ffns"][0](query)

            query = layer["norms"][1](query)

        return query
