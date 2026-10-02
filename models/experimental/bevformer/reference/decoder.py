# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Detection transformer decoder in PyTorch.

This module implements the decoder of BEVFormer's detection head, which refines the 900
object queries against the encoder's BEV features (50x50 for BEVFormer-tiny, 200x200 for
BEVFormer-base, same decoder otherwise) and returns every layer's queries and 3D reference
points for the classification and regression branches. It is the reference the TTNN
decoder in ``tt/tt_decoder.py`` is checked against. Parameter names follow mmcv's modules,
so the ``pts_bbox_head.transformer.decoder`` weights of a BEVFormer checkpoint load into
it with that prefix stripped.

Each of the six layers performs, in the sequence-first ``(num_query, bs, embed_dims)``
layout:
1. Self-attention over the queries, ``query_pos`` added to Q and K, then a residual add
2. LayerNorm
3. Deformable cross-attention, every query sampling the BEV map around its reference point,
   with a residual add
4. LayerNorm
5. FFN (Linear-ReLU-Linear) with a residual add
6. LayerNorm

After each layer the regression branch refines the reference points in logit space:
``sigmoid(delta + inverse_sigmoid(points))``, with ``delta`` the (x, y, z) entries of the
predicted box code.

Only inference is kept: dropout, attention and key padding masks are dropped, and every
layer's output is returned (upstream ``return_intermediate=True``). The cross-attention,
``CustomMSDeformableAttention`` upstream, is ``reference/ms_deformable_attention.py``'s
``MSDeformableAttention`` with one level and one reference point per query.

Adapted from the UniAD port in ``models/experimental/uniad/reference/decoder.py``, which
is based on the BEVFormer decoder and the mmdetection and mmcv versions BEVFormer is built
on:
https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/decoder.py
https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/utils/transformer.py
https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/cnn/bricks/transformer.py

BEVFormer decoder configurations:
https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_base.py
https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_tiny.py
"""

import torch
import torch.nn as nn

from models.experimental.bevformer.config import DeformableAttentionConfig
from models.experimental.bevformer.config.decoder_config import CODE_XY, CODE_Z
from models.experimental.bevformer.reference.ms_deformable_attention import MSDeformableAttention


def inverse_sigmoid(x, eps=1e-5):
    """mmdetection's ``inverse_sigmoid``: ``log(x / (1 - x))`` with ``x`` and ``1 - x`` clamped
    to at least ``eps``, so points on the [0, 1] border give finite logits."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def reg_branch(embed_dims, code_size, num_reg_fcs=2):
    """A layer's box regression branch, ``(Linear-ReLU) x num_reg_fcs`` then ``Linear(code_size)``,
    as BEVFormerHead builds it; the decoder refines its reference points with it."""
    layers = []
    for _ in range(num_reg_fcs):
        layers += [nn.Linear(embed_dims, embed_dims), nn.ReLU()]
    return nn.Sequential(*layers, nn.Linear(embed_dims, code_size))


class MultiheadAttention(nn.Module):
    """
    mmcv's ``MultiheadAttention`` reduced to the decoder's self-attention.

    ``query_pos`` is added to the query and key but not to the value, and the input is added
    back as the residual. The ``nn.MultiheadAttention`` is named ``attn``, as in mmcv, so
    checkpoint keys match.
    """

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
    """mmcv's ``FFN`` with two fully connected layers and no dropout: ``x + Linear(ReLU(Linear(x)))``.
    ``layers`` keeps mmcv's nesting, so checkpoint keys match."""

    def __init__(self, embed_dims=256, feedforward_channels=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(embed_dims, feedforward_channels), nn.ReLU(inplace=True)),
            nn.Linear(feedforward_channels, embed_dims),
        )

    def forward(self, x):
        return x + self.layers(x)


class DetrTransformerDecoderLayer(nn.Module):
    """
    One decoder layer with mmdetection's ``DetrTransformerDecoderLayer`` operation order,
    ``("self_attn", "norm", "cross_attn", "norm", "ffn", "norm")``.

    Args:
        embed_dims (int): Channels of the queries and of the BEV features.
        num_heads (int): Heads of both attentions.
        feedforward_channels (int): Hidden channels of the FFN.
        num_points (int): Sampling points per head of the cross-attention.

    ``attentions``, ``ffns`` and ``norms`` are named and ordered as in mmcv's
    ``BaseTransformerLayer``, so checkpoint keys match.
    """

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
    """
    BEVFormer's ``DetectionTransformerDecoder``: ``num_layers`` decoder layers, each followed
    by a reference-point refinement through that layer's regression branch.

    The defaults are BEVFormer's: six layers of 256 channels, 8 heads, 512 FFN channels and
    4 sampling points.
    """

    def __init__(self, num_layers=6, embed_dims=256, num_heads=8, feedforward_channels=512, num_points=4):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DetrTransformerDecoderLayer(embed_dims, num_heads, feedforward_channels, num_points)
                for _ in range(num_layers)
            ]
        )

    def forward(self, query, value, query_pos, reference_points, spatial_shapes, reg_branches):
        """Sequence-first ``query``/``query_pos`` ``(nq, bs, C)`` and ``value`` ``(bev_h * bev_w, bs, C)``;
        ``reference_points`` ``(bs, nq, 3)`` in [0, 1]; ``spatial_shapes`` the ``(1, 2)`` integer
        tensor ``[[bev_h, bev_w]]``.

        Returns every layer's output ``(L, nq, bs, C)`` and refined reference points ``(L, bs, nq, 3)``.
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(2)
            output = layer(output, value, query_pos, reference_points_input, spatial_shapes)

            box_delta = reg_branches[lid](output.permute(1, 0, 2))
            new_reference_points = torch.cat(
                [
                    box_delta[..., CODE_XY] + inverse_sigmoid(reference_points[..., :2]),
                    box_delta[..., CODE_Z] + inverse_sigmoid(reference_points[..., 2:3]),
                ],
                dim=-1,
            )
            reference_points = new_reference_points.sigmoid()

            intermediate.append(output)
            intermediate_reference_points.append(reference_points)

        return torch.stack(intermediate), torch.stack(intermediate_reference_points)
