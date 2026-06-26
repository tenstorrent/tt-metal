# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import warnings
import ttnn
from models.experimental.vadv2.tt.tt_utils import multi_scale_deformable_attn, build_folded_sampling_offsets
from models.experimental.vadv2.tt.matmul_helpers import linear_flatten_batch


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
        self.batch_first = batch_first

        # Cache for batched-scatter & host-gather lookup tables. Built once per
        # session, keyed by id(bev_mask). bev_mask is encoder-cached (see
        # iter#1 `_ref_cache` in TtBEVFormerEncoder), so id() is stable across
        # all 3 layer calls of every forward, giving 100% cache-hit rate after
        # the first call.
        # Entry: (indexes_long, max_len, all_idx_expanded_2d)
        #   - indexes_long: list[torch.LongTensor] of per-camera valid query
        #                   positions; consumed by the host gather that builds
        #                   queries_rebatch (kept on host for now).
        #   - max_len:      max over cameras of len(indexes[i]), used to size
        #                   queries_rebatch.
        #   - all_idx_expanded_2d: (num_cams * max_len, embed_dims) uint32
        #                   indices for the batched device-side scatter_add.
        #                   Padded positions hold sentinel = num_query so
        #                   their contributions land in a sink row that is
        #                   sliced off after scatter.
        self._sca_cache = {}
        # zeros/zeros_like do a host->device write of the fill, forbidden inside
        # trace capture. Cache the slot accumulator and sentinel-row templates;
        # each warm call clones them (kernel-level device copy, trace-safe).
        self._slots_template = None
        self._sentinel_row_template = None

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
            if self._slots_template is None:
                self._slots_template = ttnn.zeros_like(query)
            slots = ttnn.clone(self._slots_template)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape

        D = reference_points_cam.shape[3]  # ttnn tensor since encoder no longer to_torch's it

        # Build / look up the cached index tables. bev_mask is stable
        # across layer calls + warm iterations (encoder ref-cache from iter#1),
        # so id() works as a session-stable key.
        cache_key = id(bev_mask)
        cached = self._sca_cache.get(cache_key)
        if cached is None:
            indexes = []
            for i, mask_per_img in enumerate(bev_mask):
                index_query_per_img = ttnn.sum(mask_per_img[0], -1)
                index_query_per_img = ttnn.to_layout(index_query_per_img, ttnn.ROW_MAJOR_LAYOUT)
                for _ in range(3):  # unsqueeze 3 times
                    index_query_per_img = ttnn.unsqueeze(index_query_per_img, 0)
                output_tensor = ttnn.nonzero(index_query_per_img, queue_id=0, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(index_query_per_img)

                no_of_non_zero_indices = output_tensor[0][..., 0].item()
                # ttnn.nonzero mirrors torch.nonzero(as_tuple=False): out[1] holds
                # `count` 4-tuples (b, n, h, w) of the multi-dim coordinates of the
                # non-zero elements, laid out as a flat count*4 buffer. The input
                # here is (1, 1, 1, num_query), so the meaningful coordinate is the
                # last (w) column; the first three are always 0. Reshape to
                # (count, 4) and take column 3 to get the flat BEV-query indices,
                # all on device. (The old code sliced out[1][:count], which
                # interleaves the b/n/h zeros into the index list and silently
                # drops 3/4 of the indices.)
                coords = ttnn.reshape(output_tensor[1], (output_tensor[1].shape[-1],))
                coords = ttnn.reshape(coords[: no_of_non_zero_indices * 4], (no_of_non_zero_indices, 4))
                index_query_per_img = coords[:, 3]
                indexes.append(index_query_per_img)
                ttnn.deallocate(output_tensor[0])
                ttnn.deallocate(output_tensor[1])

            max_len = max([each.shape[0] for each in indexes])

            # Build TWO padded-index variants. Per-camera valid indices stay
            # the same; only the sentinel (padding) value differs:
            #   scatter:  sentinel = num_query  -> routes padded src into the
            #                                      sink slot in slots_extended
            #   gather:   sentinel = 0          -> padded rows gather from
            #                                      row 0 (a valid query); the
            #                                      result is harmless because
            #                                      the matching scatter
            #                                      routes those rows to the
            #                                      sink and they're sliced
            #                                      off
            scatter_parts = []
            gather_parts = []
            for i in range(self.num_cams):
                len_i = indexes[i].shape[0]
                idx_i = ttnn.typecast(indexes[i], ttnn.uint32)
                pad_len = max_len - len_i
                if pad_len > 0:
                    sentinel_scatter = ttnn.full(
                        [pad_len], num_query, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
                    )
                    sentinel_gather = ttnn.full(
                        [pad_len], 0, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
                    )
                    idx_scatter = ttnn.concat([idx_i, sentinel_scatter], dim=0)
                    idx_gather = ttnn.concat([idx_i, sentinel_gather], dim=0)
                else:
                    idx_scatter = idx_i
                    idx_gather = idx_i
                scatter_parts.append(idx_scatter)
                gather_parts.append(idx_gather)
            all_idx_padded = ttnn.concat(scatter_parts, dim=0)  # (num_cams * max_len,) sentinel=num_query
            all_idx_for_gather_q = ttnn.concat(gather_parts, dim=0)  # (num_cams * max_len,) sentinel=0
            # Expand to (num_cams * max_len, embed_dims) so it broadcasts as
            # the index arg of scatter_add along dim=0 of an (N, embed_dims)
            # slots tensor.
            all_idx_expanded = ttnn.unsqueeze(all_idx_padded, 1)
            all_idx_expanded = ttnn.expand(all_idx_expanded, [-1, self.embed_dims])

            # For the reference_points_cam gather we need indices into a
            # flattened (num_cams * num_query, D*2) embedding table, so add
            # a cam offset c*num_query to each element of all_idx_for_gather_q.
            cam_offset = ttnn.arange(0, self.num_cams, dtype=ttnn.uint32, device=self.device) * num_query  # (num_cams,)
            cam_offset = ttnn.unsqueeze(cam_offset, -1)  # (num_cams, 1)
            cam_offset = ttnn.expand(cam_offset, [-1, max_len])  # (num_cams, max_len)
            cam_offset = ttnn.reshape(cam_offset, (self.num_cams * max_len,))
            all_idx_for_gather_rc = all_idx_for_gather_q + cam_offset  # (num_cams * max_len,)

            # Release device-side per-camera nonzero outputs; everything we
            # need is now in the consolidated tensors above.
            for idx in indexes:
                ttnn.deallocate(idx)
            self._sca_cache[cache_key] = (
                max_len,
                all_idx_expanded,
                all_idx_for_gather_q,
                all_idx_for_gather_rc,
            )
        else:
            max_len, all_idx_expanded, all_idx_for_gather_q, all_idx_for_gather_rc = cached

        # Device-side gather: build queries_rebatch and reference_points_rebatch
        # via ttnn.embedding from the consolidated indices. Replaces the old
        # `query = to_torch + queries_rebatch.new_zeros + per-camera torch
        # advanced-indexing + from_torch(queries_rebatch) + from_torch(
        # reference_points_rebatch)` chain.
        query_2d = ttnn.squeeze(query, 0) if query.shape[0] == 1 else query[0]  # (num_query, embed_dims)
        queries_gathered = ttnn.embedding(
            all_idx_for_gather_q, query_2d, layout=ttnn.TILE_LAYOUT
        )  # (num_cams * max_len, embed_dims)
        queries_rebatch = ttnn.reshape(queries_gathered, (bs, self.num_cams, max_len, self.embed_dims))

        # reference_points_cam is now on device (encoder no longer to_torch's
        # it). Flatten to an embedding table of shape (num_cams * num_query,
        # D*2) for the batched gather.
        ref_cam_dev = ttnn.squeeze(reference_points_cam, 1)  # (num_cams, num_query, D, 2) — bs=1 squeezed
        ref_cam_table = ttnn.reshape(ref_cam_dev, (self.num_cams * num_query, D * 2))
        ref_cam_gathered = ttnn.embedding(
            all_idx_for_gather_rc, ref_cam_table, layout=ttnn.TILE_LAYOUT
        )  # (num_cams * max_len, D*2)
        reference_points_rebatch = ttnn.reshape(ref_cam_gathered, (bs, self.num_cams, max_len, D, 2))
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
        ttnn.deallocate(queries_rebatch)
        ttnn.deallocate(reference_points_rebatch)

        queries = ttnn.reshape(queries, (bs, self.num_cams, max_len, self.embed_dims))

        # Device-side batched scatter_add. Replaces the per-camera torch loop
        # (`slots[j, indexes[i]] += queries[j, i, :len_i]`) and its bracketing
        # to_torch(queries) / from_torch(slots) round trips. Padded src rows
        # in `queries` (attention output for the zero-padded query slots
        # past valid_len_i per camera) get routed to `slots_extended[num_query]`
        # via the sentinel index in `all_idx_expanded`, then sliced off — no
        # mask multiply needed.
        assert bs == 1, "device-side scatter currently assumes bs=1"
        queries_flat = ttnn.reshape(queries, (self.num_cams * max_len, self.embed_dims))
        slots_2d = ttnn.squeeze(slots, 0)
        slots_2d = ttnn.to_layout(slots_2d, ttnn.TILE_LAYOUT)
        if self._sentinel_row_template is None:
            self._sentinel_row_template = ttnn.zeros(
                (1, self.embed_dims), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
        sentinel_row = ttnn.clone(self._sentinel_row_template)
        slots_extended = ttnn.concat([slots_2d, sentinel_row], dim=0)
        slots_extended = ttnn.scatter_add(slots_extended, 0, all_idx_expanded, queries_flat)
        slots = ttnn.slice(slots_extended, [0, 0], [num_query, self.embed_dims])
        slots = ttnn.unsqueeze(slots, 0)

        count = ttnn.sum(bev_mask, -1) > 0
        count = ttnn.permute(count, (1, 2, 0))

        count = ttnn.sum(count, -1)
        count = ttnn.clamp(count, min=1.0)
        count = ttnn.unsqueeze(count, -1)
        slots = ttnn.to_layout(slots, layout=ttnn.TILE_LAYOUT)
        slots = ttnn.div(slots, count)
        slots = linear_flatten_batch(slots, self.params.output_proj.weight, bias=self.params.output_proj.bias)
        ttnn.deallocate(count)
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

        # Cached (H, W) per level for multi_scale_deformable_attn — avoids
        # the host-sync `.item()` cost on warm calls.
        self._hw_cache = None

        # sampling_offsets Linear weight/bias pre-scaled by 1/offset_normalizer,
        # folding the per-call offset_normalizer DIV away. Built once (static).
        self._so_w = None
        self._so_b = None

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

        value = linear_flatten_batch(value, params.value_proj.weight, bias=params.value_proj.bias)
        if key_padding_mask is not None:
            mask = key_padding_mask[..., None]
            value = ttnn.where(mask, ttnn.zeros_like(value), value)
        value = ttnn.reshape(value, (bs, num_value, self.num_heads, -1))
        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)

        if self._so_w is None:
            self._so_w, self._so_b = build_folded_sampling_offsets(params.sampling_offsets, spatial_shapes, self.device)
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

        if self._hw_cache is None:
            self._hw_cache = [
                (int(spatial_shapes[lvl, 0].item()), int(spatial_shapes[lvl, 1].item()))
                for lvl in range(self.num_levels)
            ]
        output = multi_scale_deformable_attn(
            value, spatial_shapes, sampling_locations, attention_weights, self.device, hw_py=self._hw_cache
        )
        ttnn.deallocate(value)
        ttnn.deallocate(sampling_locations)
        if reference_points.shape[-1] == 2:
            ttnn.deallocate(sampling_locations_add)
        ttnn.deallocate(attention_weights)
        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        return output
