# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import warnings
import ttnn
from models.experimental.vadv2.tt.tt_utils import multi_scale_deformable_attn, fold_sampling_offsets
from models.experimental.vadv2.tt.matmul_helpers import linear_flatten_batch


class TtSpatialCrossAttention:
    def __init__(
        self,
        device,
        params,
        spatial_shapes_hw,
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
        self.deformable_attention = TtMSDeformableAttention3D(
            device=self.device, params=params, spatial_shapes_hw=spatial_shapes_hw, num_levels=1
        )
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.batch_first = batch_first
        self._plan = None

    def prepare(self, reference_points_cam, bev_mask):
        """Build the per-camera query gather/scatter plan for a fixed camera rig.

        reference_points_cam (num_cams, 1, num_query, D, 2) and bev_mask
        (num_cams, 1, num_query, D) depend only on the calibration, so the
        nonzero search, its host read of the counts, and every derived table
        are built here once and the forward stays on device.
        """
        num_query = bev_mask.shape[2]
        D = reference_points_cam.shape[3]

        indexes = []
        for mask_per_img in bev_mask:
            index_query_per_img = ttnn.sum(mask_per_img[0], -1)
            index_query_per_img = ttnn.to_layout(index_query_per_img, ttnn.ROW_MAJOR_LAYOUT)
            for _ in range(3):
                index_query_per_img = ttnn.unsqueeze(index_query_per_img, 0)
            output_tensor = ttnn.nonzero(index_query_per_img, queue_id=0, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(index_query_per_img)

            no_of_non_zero_indices = output_tensor[0][..., 0].item()
            # ttnn.nonzero returns `count` (b, n, h, w) coordinate 4-tuples as a
            # flat count*4 buffer. The input is (1, 1, 1, num_query), so only the
            # last column carries the BEV-query index.
            coords = ttnn.reshape(output_tensor[1], (output_tensor[1].shape[-1],))
            coords = ttnn.reshape(coords[: no_of_non_zero_indices * 4], (no_of_non_zero_indices, 4))
            indexes.append(coords[:, 3])
            ttnn.deallocate(output_tensor[0])
            ttnn.deallocate(output_tensor[1])

        max_len = max(each.shape[0] for each in indexes)

        # Cameras are padded to max_len. Padded rows scatter into a sink row at
        # index num_query (sliced off after the scatter) and gather from row 0,
        # which is harmless because their results land in the sink.
        scatter_parts = []
        gather_parts = []
        for idx in indexes:
            idx_i = ttnn.typecast(idx, ttnn.uint32)
            pad_len = max_len - idx.shape[0]
            if pad_len > 0:
                sentinel_scatter = ttnn.full(
                    [pad_len], num_query, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
                )
                sentinel_gather = ttnn.full(
                    [pad_len], 0, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
                )
                scatter_parts.append(ttnn.concat([idx_i, sentinel_scatter], dim=0))
                gather_parts.append(ttnn.concat([idx_i, sentinel_gather], dim=0))
            else:
                scatter_parts.append(idx_i)
                gather_parts.append(idx_i)
        all_idx_padded = ttnn.concat(scatter_parts, dim=0)
        all_idx_for_gather_q = ttnn.concat(gather_parts, dim=0)
        all_idx_expanded = ttnn.unsqueeze(all_idx_padded, 1)
        all_idx_expanded = ttnn.expand(all_idx_expanded, [-1, self.embed_dims])

        # reference_points_cam is gathered from a flattened (num_cams * num_query, D*2)
        # table, so each camera's indices are offset by cam * num_query.
        cam_offset = ttnn.arange(0, self.num_cams, dtype=ttnn.uint32, device=self.device) * num_query
        cam_offset = ttnn.unsqueeze(cam_offset, -1)
        cam_offset = ttnn.expand(cam_offset, [-1, max_len])
        cam_offset = ttnn.reshape(cam_offset, (self.num_cams * max_len,))
        all_idx_for_gather_rc = all_idx_for_gather_q + cam_offset
        for idx in indexes:
            ttnn.deallocate(idx)

        ref_cam_table = ttnn.reshape(ttnn.squeeze(reference_points_cam, 1), (self.num_cams * num_query, D * 2))
        ref_cam_gathered = ttnn.embedding(all_idx_for_gather_rc, ref_cam_table, layout=ttnn.TILE_LAYOUT)
        reference_points_rebatch = ttnn.reshape(ref_cam_gathered, (self.num_cams, max_len, D, 2))

        count = ttnn.sum(bev_mask, -1) > 0
        count = ttnn.permute(count, (1, 2, 0))
        count = ttnn.sum(count, -1)
        count = ttnn.clamp(count, min=1.0)
        count = ttnn.unsqueeze(count, -1)

        # The plan lives for the whole model lifetime; keep it out of L1 so it
        # cannot collide with circular buffers of later programs.
        to_dram = lambda t: ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
        self._plan = dict(
            num_query=num_query,
            max_len=max_len,
            all_idx_expanded=to_dram(all_idx_expanded),
            all_idx_for_gather_q=to_dram(all_idx_for_gather_q),
            reference_points_rebatch=to_dram(reference_points_rebatch),
            count=to_dram(count),
            # Cloned per call: zero fills are host writes, and the forward must not issue them.
            slots_template=ttnn.zeros(
                (1, num_query, self.embed_dims), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            ),
            sentinel_row_template=ttnn.zeros(
                (1, self.embed_dims), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            ),
        )

    def __call__(
        self,
        query,
        key,
        value,
        residual=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        flag="encoder",
        **kwargs,
    ):
        assert self._plan is not None, "call prepare() with the rig's reference_points_cam / bev_mask first"
        plan = self._plan
        max_len = plan["max_len"]

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = ttnn.clone(plan["slots_template"])
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape
        assert bs == 1, "the device-side gather/scatter assumes bs=1"
        assert num_query == plan["num_query"]

        query_2d = ttnn.squeeze(query, 0)
        queries_gathered = ttnn.embedding(plan["all_idx_for_gather_q"], query_2d, layout=ttnn.TILE_LAYOUT)
        queries_rebatch = ttnn.reshape(queries_gathered, (bs * self.num_cams, max_len, self.embed_dims))
        reference_points_rebatch = ttnn.clone(plan["reference_points_rebatch"])
        num_cams, l, bs, embed_dims = key.shape

        key = ttnn.permute(key, (2, 0, 1, 3))
        key = ttnn.reshape(key, (bs * self.num_cams, l, self.embed_dims))

        value = ttnn.permute(value, (2, 0, 1, 3))
        value = ttnn.reshape(value, (bs * self.num_cams, l, self.embed_dims))
        queries = self.deformable_attention(
            query=queries_rebatch,
            key=key,
            value=value,
            reference_points=reference_points_rebatch,
        )
        ttnn.deallocate(queries_rebatch)
        ttnn.deallocate(reference_points_rebatch)

        # Padded rows go to the sink row at index num_query through the sentinel
        # in all_idx_expanded and are sliced off, so no mask multiply is needed.
        queries_flat = ttnn.reshape(queries, (self.num_cams * max_len, self.embed_dims))
        slots_2d = ttnn.squeeze(slots, 0)
        slots_2d = ttnn.to_layout(slots_2d, ttnn.TILE_LAYOUT)
        sentinel_row = ttnn.clone(plan["sentinel_row_template"])
        slots_extended = ttnn.concat([slots_2d, sentinel_row], dim=0)
        slots_extended = ttnn.scatter_add(slots_extended, 0, plan["all_idx_expanded"], queries_flat)
        slots = ttnn.slice(slots_extended, [0, 0], [num_query, self.embed_dims])
        slots = ttnn.unsqueeze(slots, 0)

        slots = ttnn.to_layout(slots, layout=ttnn.TILE_LAYOUT)
        slots = ttnn.div(slots, plan["count"])
        slots = linear_flatten_batch(slots, self.params.output_proj.weight, bias=self.params.output_proj.bias)
        ttnn.deallocate(key)
        ttnn.deallocate(value)

        output = slots + inp_residual
        ttnn.deallocate(slots)
        ttnn.deallocate(inp_residual)

        return output


class TtMSDeformableAttention3D:
    def __init__(
        self,
        device,
        params,
        spatial_shapes_hw,
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

        assert len(spatial_shapes_hw) == num_levels, f"need {num_levels} (H, W) pairs, got {spatial_shapes_hw}"
        self.spatial_shapes_hw = [(int(h), int(w)) for h, w in spatial_shapes_hw]
        self._so_w, self._so_b = fold_sampling_offsets(params.sampling_offsets, self.spatial_shapes_hw, device)

    def __call__(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
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
        assert sum(h * w for h, w in self.spatial_shapes_hw) == num_value
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)

        value = linear_flatten_batch(value, params.value_proj.weight, bias=params.value_proj.bias)
        if key_padding_mask is not None:
            mask = key_padding_mask[..., None]
            value = ttnn.where(mask, ttnn.zeros_like(value), value)
        value = ttnn.reshape(value, (bs, num_value, self.num_heads, -1))
        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)

        sampling_offsets = linear_flatten_batch(query, self._so_w, bias=self._so_b)
        sampling_offsets = ttnn.reshape(
            sampling_offsets, (bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        )
        attention_weights = linear_flatten_batch(
            query, params.attention_weights.weight, bias=params.attention_weights.bias
        )
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_levels * self.num_points)
        )

        attention_weights = ttnn.softmax(attention_weights, -1)
        attention_weights = ttnn.reallocate(attention_weights)
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_levels, self.num_points)
        )

        if reference_points.shape[-1] == 2:
            bs_r, num_query, num_Z_anchors, _ = reference_points.shape
            reference_xy = ttnn.reshape(
                reference_points, (bs_r, num_query, 1, 1, 1, reference_points.shape[-2], reference_points.shape[-1])
            )
            sampling_offsets = ttnn.to_layout(sampling_offsets, ttnn.TILE_LAYOUT)
            # sampling_offsets is already divided by offset_normalizer (folded into
            # the sampling_offsets Linear weight).
            sampling_locations = sampling_offsets

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

            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors
            sampling_locations = ttnn.reshape(
                sampling_locations, (bs, num_query, num_heads, num_levels, num_all_points, xy)
            )

        elif reference_points.shape[-1] == 4:
            # Already unused (was `assert False`); also incompatible with the folded
            # offset_normalizer — sampling_offsets is pre-scaled by 1/[W,H] in the
            # Linear weights, so any 4-D path would need unscaled offsets.
            raise NotImplementedError(
                "4-D reference_points is incompatible with the folded offset_normalizer "
                "(see build_folded_sampling_offsets)."
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be" f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )

        output = multi_scale_deformable_attn(
            value, self.spatial_shapes_hw, sampling_locations, attention_weights, self.device
        )
        ttnn.deallocate(value)
        ttnn.deallocate(sampling_locations)
        if reference_points.shape[-1] == 2:
            ttnn.deallocate(sampling_locations_add)
        ttnn.deallocate(attention_weights)
        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        return output
