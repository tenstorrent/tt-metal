# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.uniad.tt.ttnn_utils import multi_scale_deformable_attn_pytorch
from models.experimental.uniad.tt.ttnn_ffn import TtFFN


class TtMultiScaleDeformableAttention:
    def __init__(
        self,
        params,
        device,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        im2col_step=64,
        batch_first=False,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, " f"but got {embed_dims} and {num_heads}")
        dim_per_head = embed_dims // num_heads
        self.params = params
        self.device = device
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.fp16_enabled = False

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

    def __call__(
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
        **kwargs,
    ):
        params = self.params
        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = ttnn.permute(query, (1, 0, 2))
            value = ttnn.permute(value, (1, 0, 2))

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (ttnn.sum(spatial_shapes[:, 0] * spatial_shapes[:, 1])) == num_value
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        value = ttnn.linear(value, params.value_proj.weight, bias=params.value_proj.bias)
        if key_padding_mask is not None:
            mask = ttnn.unsqueeze(key_padding_mask, dim=-1)
            mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)
            value = ttnn.where(mask, ttnn.zeros_like(value, device=self.device, layout=ttnn.TILE_LAYOUT), value)
        value = ttnn.reshape(value, (bs, num_value, self.num_heads, -1))
        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        sampling_offsets = ttnn.linear(query, params.sampling_offsets.weight, bias=params.sampling_offsets.bias)
        sampling_offsets = ttnn.reshape(
            sampling_offsets, (bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        )
        attention_weights = ttnn.linear(query, params.attention_weights.weight, bias=params.attention_weights.bias)
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_levels * self.num_points)
        )

        attention_weights = ttnn.softmax(attention_weights, dim=-1)

        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_levels, self.num_points)
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = ttnn.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1)  # [num_levels, 2]
            offset_normalizer_xy = ttnn.reshape(offset_normalizer, (1, 1, 1, offset_normalizer.shape[0], 1, 2))
            reference_xy = ttnn.reshape(
                reference_points,
                (reference_points.shape[0], reference_points.shape[1], 1, reference_points.shape[2], 1, 2),
            )
            sampling_offsets = ttnn.to_layout(sampling_offsets, ttnn.TILE_LAYOUT)
            offset_normalizer_xy = ttnn.to_layout(offset_normalizer_xy, ttnn.TILE_LAYOUT)
            sampling_offsets = ttnn.div(sampling_offsets, offset_normalizer_xy)

            sampling_locations = ttnn.add(reference_xy, sampling_offsets)
        elif reference_points.shape[-1] == 4:
            reference_points = ttnn.unsqueeze(reference_points, dim=2)
            reference_points = ttnn.unsqueeze(reference_points, dim=2)
            reference_points_reshape = reference_points[:, :, :, :, :, :2]
            sampling_locations = (
                reference_points_reshape + ttnn.div(sampling_offsets, self.num_points) * reference_points_reshape * 0.5
            )

        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, " f"but got {reference_points.shape[-1]} instead."
            )
        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, None, sampling_locations, attention_weights, None, self.device
        )

        output = ttnn.linear(output, params.output_proj.weight, bias=params.output_proj.bias)
        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        output = ttnn.add(output, identity)

        return output


class TtDetrTransformerEncoder:
    def __init__(
        self,
        params,
        device,
        embed_dims=256,
        num_layers=6,
        num_heads=8,
        num_levels=4,
        num_points=4,
        im2col_step=64,
        batch_first=False,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__()
        self.device = device
        self.params = params
        self.num_layers = num_layers
        self.layers = []
        for i in range(num_layers):
            attn = TtMultiScaleDeformableAttention(
                params=params.layers[i].attentions[0],
                device=device,
            )
            ffn = TtFFN(
                params=params.layers[i].ffns[0].ffn.ffn0,
                device=device,
            )
            self.layers.append([attn, ffn])

    def __call__(
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
        query_key_padding_mask=None,
        **kwargs,
    ):
        for i, layer in enumerate(self.layers):
            temp_key = temp_value = query
            query = layer[0](
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
            )
            identity = query

            query = ttnn.layer_norm(
                query,
                weight=self.params.layers[i].norms[0].weight,
                bias=self.params.layers[i].norms[0].bias,
            )
            query = layer[1](query)
            query = ttnn.layer_norm(
                query,
                weight=self.params.layers[i].norms[1].weight,
                bias=self.params.layers[i].norms[1].bias,
            )

        return query
