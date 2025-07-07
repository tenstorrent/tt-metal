import warnings
import ttnn
from models.experimental.vadv2.tt.tt_temporal_self_attention import multi_scale_deformable_attn


class TtSpatialCrossAttention:
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
        self.deformable_attention = TtMSDeformableAttention3D(device=self.device, params=params, num_levels=1)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        # self.output_proj = nn.Linear(embed_dims, embed_dims)
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
            inp_residual = query  # ttnn.from_torch(query, dtype = ttnn.bfloat16, layout = ttnn.ROW_MAJOR_LAYOUT, device = self.device)
            slots = ttnn.zeros_like(query)
            slots = ttnn.to_torch(slots)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape

        D = reference_points_cam.size(3)
        indexes = []
        indexes = []
        # bev_mask = ttnn.to_torch(bev_mask)
        # reference_points_cam = ttnn.to_torch(reference_points_cam)
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = ttnn.sum(mask_per_img[0], -1)
            index_query_per_img = ttnn.to_torch(index_query_per_img)
            index_query_per_img = index_query_per_img.nonzero()
            index_query_per_img = ttnn.from_torch(index_query_per_img, device=self.device, dtype=ttnn.uint32)

            index_query_per_img = ttnn.squeeze(index_query_per_img, -1)

            indexes.append(index_query_per_img)
        max_len = max([each.shape[0] for each in indexes])
        query = ttnn.to_torch(query)
        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, D, 2])

        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[i]
                index_query_per_img = ttnn.to_torch(index_query_per_img)
                queries_rebatch[j, i, : len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, : len(index_query_per_img)] = reference_points_per_img[
                    j, index_query_per_img
                ]
        queries_rebatch = ttnn.from_torch(queries_rebatch, dtype=ttnn.bfloat16, device=self.device)
        reference_points_rebatch = ttnn.from_torch(reference_points_rebatch, dtype=ttnn.bfloat16, device=self.device)
        num_cams, l, bs, embed_dims = key.shape
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
        queries = ttnn.reshape(queries, (bs, self.num_cams, max_len, self.embed_dims))

        queries = ttnn.to_torch(queries)
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                index_query_per_img = ttnn.to_torch(index_query_per_img)

                slots[j, index_query_per_img] += queries[j, i, : len(index_query_per_img)]

        count = ttnn.sum(bev_mask, -1) > 0
        count = ttnn.permute(count, (1, 2, 0))

        count = ttnn.sum(count, -1)
        count = ttnn.clamp(count, min=1.0)
        count = ttnn.unsqueeze(count, -1)
        slots = ttnn.from_torch(slots, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        slots = ttnn.div(slots, count)
        slots = ttnn.linear(slots, self.params.output_proj.weight, bias=self.params.output_proj.bias)
        ttnn.deallocate(self.params.output_proj.weight)
        ttnn.deallocate(self.params.output_proj.bias)

        return slots + inp_residual


class TtMSDeformableAttention3D:
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
        # self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        # self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        # self.value_proj = nn.Linear(embed_dims, embed_dims)

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
            mask = key_padding_mask[..., None]
            value = ttnn.where(mask, ttnn.zeros_like(value), value)
        value = ttnn.reshape(value, (bs, num_value, self.num_heads, -1))
        # query = ttnn.from_torch(query, dtype=ttnn.bfloat16, device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT)
        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)

        sampling_offsets = ttnn.linear(query, params.sampling_offsets.weight, bias=params.sampling_offsets.bias)
        sampling_offsets = ttnn.reshape(
            sampling_offsets, (bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        )
        attention_weights = ttnn.linear(query, params.attention_weights.weight, bias=params.attention_weights.bias)
        ttnn.deallocate(params.attention_weights.weight)
        ttnn.deallocate(params.attention_weights.bias)
        # ttnn.deallocate(query)
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_levels * self.num_points)
        )
        attention_weights = ttnn.to_torch(attention_weights)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = ttnn.from_torch(
            attention_weights, device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
        )

        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_levels, self.num_points)
        )
        # reference_points = ttnn.from_torch(
        #     reference_points, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device
        # )
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
            sampling_offsets = ttnn.to_torch(sampling_offsets)
            offset_normalizer_xy = ttnn.to_torch(offset_normalizer_xy)
            sampling_locations = sampling_offsets / offset_normalizer_xy
            reference_xy = ttnn.to_torch(reference_xy)
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_locations.shape
            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy
            )
            sampling_locations = reference_xy + sampling_locations
            sampling_locations = ttnn.from_torch(sampling_locations, device=self.device, dtype=ttnn.bfloat16)
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

        output = multi_scale_deformable_attn(
            value, spatial_shapes, sampling_locations, attention_weights, self.device, reshape=True
        )

        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        return output
