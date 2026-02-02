# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.BEVFormerV2.tt.ttnn_utils import multi_scale_deformable_attn


class TtSpatialCrossAttention:
    """TTNN implementation of SpatialCrossAttention"""

    def __init__(
        self,
        device,
        params,
        embed_dims=256,
        num_cams=6,
        pc_range=None,
        dropout=0.1,
        init_cfg=None,
        batch_first=False,
        deformable_attention=dict(type="MSDeformableAttention3D", embed_dims=256, num_levels=4),
        **kwargs,
    ):
        super(TtSpatialCrossAttention, self).__init__()

        self.device = device
        self.params = params
        self.init_cfg = init_cfg
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = TtMSDeformableAttention3D(device=self.device, params=params, num_levels=4)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.batch_first = batch_first

    def __call__(
        self,
        query,
        key,
        value,
        residual=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        reference_points_cam=None,
        bev_mask=None,
        level_start_index=None,
        flag="encoder",
        **kwargs,
    ):
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = ttnn.zeros_like(query)
            slots = ttnn.to_torch(slots)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape

        D = reference_points_cam.size(3)

        if isinstance(query, ttnn.Tensor):
            query = ttnn.to_torch(query)
        if isinstance(reference_points_cam, ttnn.Tensor):
            reference_points_cam = ttnn.to_torch(reference_points_cam)
        if isinstance(bev_mask, ttnn.Tensor):
            bev_mask_torch = ttnn.to_torch(bev_mask)
        else:
            bev_mask_torch = bev_mask

        indexes = []
        for i, mask_per_img in enumerate(bev_mask_torch):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)

        max_len = max([len(each) for each in indexes])
        queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, D, 2])

        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, : len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, : len(index_query_per_img)] = reference_points_per_img[
                    j, index_query_per_img
                ]

        queries_rebatch = ttnn.from_torch(queries_rebatch, dtype=ttnn.bfloat16, device=self.device)
        reference_points_rebatch = ttnn.from_torch(reference_points_rebatch, dtype=ttnn.bfloat16, device=self.device)
        num_cams, l, bs, embed_dims = key.shape

        key = ttnn.permute(key, (2, 0, 1, 3))
        key = ttnn.reshape(key, (bs * self.num_cams, l, self.embed_dims))

        value = ttnn.permute(value, (2, 0, 1, 3))
        value = ttnn.reshape(value, (bs * self.num_cams, l, self.embed_dims))
        queries = self.deformable_attention(
            query=ttnn.reshape(queries_rebatch, (bs * self.num_cams, max_len, self.embed_dims)),
            key=key,
            value=value,
            reference_points=ttnn.reshape(reference_points_rebatch, (bs * self.num_cams, max_len, D, 2)),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        ttnn.deallocate(queries_rebatch)
        ttnn.deallocate(reference_points_rebatch)

        queries = ttnn.reshape(queries, (bs, self.num_cams, max_len, self.embed_dims))

        queries = ttnn.to_torch(queries)
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, : len(index_query_per_img)]

        count = ttnn.sum(bev_mask, -1) > 0
        count = ttnn.permute(count, (1, 2, 0))

        count = ttnn.sum(count, -1)
        count = ttnn.clamp(count, min=1.0)
        count = ttnn.unsqueeze(count, -1)
        slots = ttnn.from_torch(slots, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        slots = ttnn.div(slots, count)
        slots = ttnn.linear(slots, self.params.output_proj.weight, bias=self.params.output_proj.bias)
        ttnn.deallocate(count)
        ttnn.deallocate(key)
        ttnn.deallocate(value)

        output = slots + inp_residual
        ttnn.deallocate(slots)
        ttnn.deallocate(inp_residual)

        return output


class TtMSDeformableAttention3D:
    """TTNN implementation of MSDeformableAttention3D"""

    def __init__(
        self,
        device,
        params,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=8,
        im2col_step=64,
        dropout=0.1,
        batch_first=True,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, " f"but got {embed_dims} and {num_heads}")
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False
        self.device = device
        self.params = params

        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
            return (n & (n - 1) == 0) and n != 0

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
            query = ttnn.permute(query, (1, 0, 2))
            value = ttnn.permute(value, (1, 0, 2))

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (ttnn.sum(spatial_shapes[:, 0] * spatial_shapes[:, 1])) == num_value
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)

        value = ttnn.linear(value, params.value_proj.weight, bias=params.value_proj.bias)
        if key_padding_mask is not None:
            mask = key_padding_mask[..., None]
            value = ttnn.where(mask, ttnn.zeros_like(value), value)
        value = ttnn.reshape(value, (bs, num_value, self.num_heads, -1))
        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)

        sampling_offsets = ttnn.linear(query, params.sampling_offsets.weight, bias=params.sampling_offsets.bias)
        num_levels_actual = spatial_shapes.shape[0]
        sampling_offsets_features = sampling_offsets.shape[-1]
        expected_features_per_level = self.num_heads * self.num_points * 2
        if sampling_offsets_features == expected_features_per_level * num_levels_actual:
            sampling_offsets = ttnn.reshape(
                sampling_offsets, (bs, num_query, self.num_heads, num_levels_actual, self.num_points, 2)
            )
        else:
            sampling_offsets = ttnn.reshape(
                sampling_offsets, (bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
            )
        attention_weights = ttnn.linear(query, params.attention_weights.weight, bias=params.attention_weights.bias)
        attention_weights_features = attention_weights.shape[-1]
        expected_attention_features = self.num_heads * self.num_points * num_levels_actual
        if attention_weights_features == expected_attention_features:
            attention_weights = ttnn.reshape(
                attention_weights, (bs, num_query, self.num_heads, num_levels_actual * self.num_points)
            )
        else:
            attention_weights = ttnn.reshape(
                attention_weights, (bs, num_query, self.num_heads, self.num_levels * self.num_points)
            )

        attention_weights = ttnn.softmax(attention_weights, -1)
        attention_weights = ttnn.reallocate(attention_weights)
        if attention_weights_features == expected_attention_features:
            attention_weights = ttnn.reshape(
                attention_weights, (bs, num_query, self.num_heads, num_levels_actual, self.num_points)
            )
        else:
            attention_weights = ttnn.reshape(
                attention_weights, (bs, num_query, self.num_heads, self.num_levels, self.num_points)
            )

        if reference_points.shape[-1] == 2:
            offset_normalizer = ttnn.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1)
            bs_r, num_query, num_Z_anchors, _ = reference_points.shape
            reference_xy = ttnn.reshape(
                reference_points, (bs_r, num_query, 1, 1, 1, reference_points.shape[-2], reference_points.shape[-1])
            )
            offset_normalizer_xy = ttnn.reshape(
                offset_normalizer, (1, 1, 1, offset_normalizer.shape[0], 1, offset_normalizer.shape[1])
            )

            sampling_offsets = ttnn.to_layout(sampling_offsets, ttnn.TILE_LAYOUT)
            offset_normalizer_xy = ttnn.to_layout(offset_normalizer_xy, ttnn.TILE_LAYOUT)

            bs_s, num_query_s, num_heads_s, num_levels_s, num_points_s, xy_s = sampling_offsets.shape
            sampling_offsets_reshaped = ttnn.reshape(
                sampling_offsets, [bs_s, num_query_s * num_heads_s, num_levels_s, num_points_s, xy_s]
            )
            offset_normalizer_xy_reshaped = ttnn.reshape(
                offset_normalizer_xy,
                [
                    offset_normalizer_xy.shape[0],
                    offset_normalizer_xy.shape[1],
                    offset_normalizer_xy.shape[3],
                    offset_normalizer_xy.shape[4],
                    offset_normalizer_xy.shape[5],
                ],
            )

            sampling_locations = ttnn.div(sampling_offsets_reshaped, offset_normalizer_xy_reshaped)
            ttnn.deallocate(sampling_offsets_reshaped)
            ttnn.deallocate(offset_normalizer_xy_reshaped)
            ttnn.deallocate(offset_normalizer_xy)

            sampling_locations = ttnn.reshape(
                sampling_locations, [bs_s, num_query_s, num_heads_s, num_levels_s, num_points_s, xy_s]
            )

            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_locations.shape
            sampling_locations = ttnn.reshape(
                sampling_locations,
                [bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy],
            )
            reference_xy_reshaped = ttnn.reshape(
                reference_xy,
                (
                    reference_xy.shape[0],
                    reference_xy.shape[1],
                    -1,
                    reference_xy.shape[4],
                    reference_xy.shape[5],
                    reference_xy.shape[6],
                ),
            )
            sampling_locations_reshaped = ttnn.reshape(
                sampling_locations,
                (
                    sampling_locations.shape[0],
                    sampling_locations.shape[1],
                    -1,
                    sampling_locations.shape[4],
                    sampling_locations.shape[5],
                    sampling_locations.shape[6],
                ),
            )

            sampling_locations_add = reference_xy_reshaped + sampling_locations_reshaped

            sampling_locations = ttnn.reshape(sampling_locations_add, sampling_locations.shape)

            ttnn.deallocate(reference_xy_reshaped)
            ttnn.deallocate(sampling_locations_reshaped)
            ttnn.deallocate(sampling_locations_add)

            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors
            sampling_locations = ttnn.reshape(
                sampling_locations, (bs, num_query, num_heads, num_levels, num_all_points, xy)
            )

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f"Last dim of reference_points must be" f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )

        output = multi_scale_deformable_attn(value, spatial_shapes, sampling_locations, attention_weights, self.device)
        ttnn.deallocate(value)
        ttnn.deallocate(sampling_locations)
        ttnn.deallocate(attention_weights)
        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        return output
