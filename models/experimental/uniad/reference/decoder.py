# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from torch.nn.modules import ModuleList
import torch
import warnings
import copy

from models.experimental.uniad.reference.utils import inverse_sigmoid, multi_scale_deformable_attn_pytorch
from models.experimental.uniad.reference.ffn import FFN


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dims,
        num_heads,
        init_cfg=None,
        batch_first=False,
        **kwargs,
    ):
        super(MultiheadAttention, self).__init__()

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, **kwargs)

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
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f"position encoding of key is" f"missing in {self.__class__.__name__}.")
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(query=query, key=key, value=value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1)
        return identity + out


class CustomMSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=1,
        num_points=4,
        im2col_step=192,
        batch_first=False,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, " f"but got {embed_dims} and {num_heads}")
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.fp16_enabled = False

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
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        flag="decoder",
        **kwargs,
    ):
        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
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

        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, level_start_index, sampling_locations, attention_weights, self.im2col_step
        )

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output + identity


class DetrTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        attn_cfgs=None,
        ffn_cfgs=dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        operation_order=None,
        norm_cfg=dict(type="LN"),
        init_cfg=None,
        batch_first=False,
        **kwargs,
    ):
        deprecated_args = dict(feedforward_channels="feedforward_channels", ffn_num_fcs="num_fcs")
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f"The arguments `{ori_name}` in BaseTransformerLayer "
                    f"has been deprecated, now you should set `{new_name}` "
                    f"and other FFN related arguments "
                    f"to a dict named `ffn_cfgs`. "
                )
                ffn_cfgs[new_name] = kwargs[ori_name]

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
                    attention = CustomMSDeformableAttention(**attn_cfgs[index])
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
            self.ffns.append(FFN(self.embed_dims))

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
            warnings.warn(f"Use same attn_mask in all attentions in " f"{self.__class__.__name__} ")
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

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
                        {"type": "CustomMSDeformableAttention", "embed_dims": embed_dim, "num_levels": 1},
                    ],
                    ffn_cfgs=dict(
                        type="FFN",
                        embed_dims=embed_dim,
                        feedforward_channels=512,
                        num_fcs=2,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
                    operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                    norm_cfg=dict(type="LN"),
                    init_cfg=None,
                    batch_first=False,
                    kwargs={
                        "feedforward_channels": 512,
                        "act_cfg": {"type": "ReLU", "inplace": True},
                        "ffn_num_fcs": 2,
                    },
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
            reference_points_input = reference_points[..., :2].unsqueeze(2)  # BS NUM_QUERY NUM_LEVEL 2
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

                assert reference_points.shape[-1] == 3

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
