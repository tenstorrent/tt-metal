# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""PyTorch reference of BEVFormer's detection decoder.

Based on https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/decoder.py

Each layer runs ``self_attn -> norm -> cross_attn -> norm -> ffn -> norm`` in the
sequence-first ``(num_query, bs, embed_dims)`` layout. The cross-attention
(``CustomMSDeformableAttention`` upstream) is ``MSDeformableAttention`` with one level
and one reference point per query, over the BEV feature map.
"""

import torch
import torch.nn as nn

from models.experimental.bevformer.config import DeformableAttentionConfig
from models.experimental.bevformer.config.decoder_config import REG_XY, REG_Z
from models.experimental.bevformer.reference.ms_deformable_attention import MSDeformableAttention


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dims, num_heads):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dims, num_heads)

    def forward(self, query, query_pos=None):
        identity = query
        if query_pos is not None:
            query = query + query_pos
        out = self.attn(query=query, key=query, value=identity)[0]
        return identity + out


class FFN(nn.Module):
    def __init__(self, embed_dims=256, feedforward_channels=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(embed_dims, feedforward_channels), nn.ReLU(inplace=True)),
            nn.Linear(feedforward_channels, embed_dims),
        )

    def forward(self, x):
        return x + self.layers(x)


class DetrTransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dims, num_heads, feedforward_channels, num_points):
        super().__init__()
        cross_attn_config = DeformableAttentionConfig(
            embed_dims=embed_dims, num_heads=num_heads, num_levels=1, num_points=num_points, batch_first=False
        )
        self.attentions = nn.ModuleList(
            [MultiheadAttention(embed_dims, num_heads), MSDeformableAttention(cross_attn_config)]
        )
        self.ffns = nn.ModuleList([FFN(embed_dims, feedforward_channels)])
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(3)])

    def forward(self, query, value, query_pos, reference_points, spatial_shapes):
        """``reference_points`` is ``(bs, nq, 1, 2)`` in [0, 1]; ``spatial_shapes`` is ``[[bev_h, bev_w]]``."""
        query = self.norms[0](self.attentions[0](query, query_pos=query_pos))
        query = self.attentions[1](
            query, value=value, query_pos=query_pos, reference_points=reference_points, spatial_shapes=spatial_shapes
        )
        query = self.norms[1](query)
        return self.norms[2](self.ffns[0](query))


class DetectionTransformerDecoder(nn.Module):
    def __init__(self, num_layers=6, embed_dims=256, num_heads=8, feedforward_channels=512, num_points=4):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DetrTransformerDecoderLayer(embed_dims, num_heads, feedforward_channels, num_points)
                for _ in range(num_layers)
            ]
        )

    def forward(self, query, value, query_pos, reference_points, spatial_shapes, reg_branches):
        """Returns every layer's output ``(L, nq, bs, C)`` and refined reference points ``(L, bs, nq, 3)``.

        ``spatial_shapes`` is the ``(1, 2)`` integer tensor ``[[bev_h, bev_w]]``.
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(2)
            output = layer(output, value, query_pos, reference_points_input, spatial_shapes)

            tmp = reg_branches[lid](output.permute(1, 0, 2))
            new_reference_points = torch.cat(
                [
                    tmp[..., REG_XY] + inverse_sigmoid(reference_points[..., :2]),
                    tmp[..., REG_Z] + inverse_sigmoid(reference_points[..., 2:3]),
                ],
                dim=-1,
            )
            reference_points = new_reference_points.sigmoid()

            intermediate.append(output)
            intermediate_reference_points.append(reference_points)

        return torch.stack(intermediate), torch.stack(intermediate_reference_points)
