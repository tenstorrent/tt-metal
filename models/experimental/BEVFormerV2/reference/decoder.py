# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

##########################################################################
# Adapted from BEVFormer (https://github.com/fundamentalvision/BEVFormer).
# Original work Copyright (c) OpenMMLab.
# Modified by Zhiqi Li.
# Licensed under the Apache License, Version 2.0.
##########################################################################

import torch.nn as nn
from torch.nn.modules import ModuleList
import torch
import copy

from .utils import inverse_sigmoid
from .ffn import FFN
from .multihead_attention import MultiheadAttention, CustomMSDeformableAttention


class DetrTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        attn_cfgs=None,
        ffn_cfgs=dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=512,
            num_fcs=2,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        operation_order=None,
        norm_cfg=dict(type="LN"),
        batch_first=False,
        **kwargs,
    ):
        super(DetrTransformerDecoderLayer, self).__init__()

        self.batch_first = batch_first

        assert set(operation_order) & set(["self_attn", "norm", "ffn", "cross_attn"]) == set(operation_order), (
            f"The operation_order of"
            f" {self.__class__.__name__} should "
            f"contains all four operation type "
            f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"
        )

        num_attn = operation_order.count("self_attn") + operation_order.count("cross_attn")
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), (
                f"The length "
                f"of attn_cfg {num_attn} is "
                f"not consistent with the number of attention"
                f"in operation_order {operation_order}."
            )

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == "norm"
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                if "batch_first" in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]["batch_first"]
                else:
                    attn_cfgs[index]["batch_first"] = self.batch_first
                if attn_cfgs[index]["type"] == "MultiheadAttention":
                    type = attn_cfgs[index].pop("type")
                    attention = MultiheadAttention(**attn_cfgs[index])
                    attn_cfgs[index]["type"] = "MultiheadAttention"
                elif attn_cfgs[index]["type"] == "CustomMSDeformableAttention":
                    type = attn_cfgs[index].pop("type")
                    attn_cfg_copy = attn_cfgs[index].copy()
                    if "dropout" not in attn_cfg_copy:
                        attn_cfg_copy["dropout"] = 0.1
                    attention = CustomMSDeformableAttention(**attn_cfg_copy)
                    attn_cfgs[index]["type"] = "CustomMSDeformableAttention"

                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.pre_norm = operation_order[0] == "norm"

        self.embed_dims = self.attentions[0].embed_dims

        num_attn = operation_order.count("self_attn") + operation_order.count("cross_attn")

        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), (
                f"The length "
                f"of attn_cfg {num_attn} is "
                f"not consistent with the number of attention"
                f"in operation_order {operation_order}."
            )

        self.num_attn = num_attn
        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count("ffn")

        for ffn_index in range(num_ffns):
            self.ffns.append(FFN(self.embed_dims, ffn_cfgs.get("feedforward_channels", 512)))

        self.norms = nn.ModuleList()
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(nn.LayerNorm(self.embed_dims))

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
        else:
            pass

        for layer in self.operation_order:
            if layer == "self_attn":
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


class DetectionTransformerDecoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, return_intermediate=True):
        super(DetectionTransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.layers = ModuleList(
            [
                DetrTransformerDecoderLayer(
                    attn_cfgs=[
                        {"type": "MultiheadAttention", "embed_dims": embed_dim, "num_heads": num_heads},
                        {
                            "type": "CustomMSDeformableAttention",
                            "embed_dims": embed_dim,
                            "num_levels": 1,
                            "dropout": 0.1,
                        },
                    ],
                    ffn_cfgs=dict(
                        type="FFN",
                        embed_dims=embed_dim,
                        feedforward_channels=512,
                        num_fcs=2,
                        ffn_dropout=0.1,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
                    operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                    norm_cfg=dict(type="LN"),
                    batch_first=False,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        reg_branches=None,
        cls_branches=None,
        **kwargs,
    ):
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(2)
            output = layer(
                output,
                key=key,
                value=value,
                query_pos=query_pos,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
            )
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points
