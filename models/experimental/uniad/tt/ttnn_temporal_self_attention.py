# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.experimental.uniad.tt.ttnn_utils import multi_scale_deformable_attn_pytorch
from models.experimental.uniad.tt.ttnn_enc_timing import record as _enc_record, sync_now as _enc_sync_now


class TtTemporalSelfAttention:
    def __init__(
        self,
        device,
        params,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        num_bev_queue=2,
        im2col_step=64,
        batch_first=True,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, got {embed_dims} and {num_heads}")

        self.device = device
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.fp16_enabled = False
        self.params = params
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        # Cached Python ints for spatial shapes — see TtCustomMSDeformableAttention.
        self._spatial_shapes_list_cache = None
        # offset_normalizer_xy and reference_xy are layer-invariant: encoder
        # rebuilds reference_points once per forward then reuses across all
        # 6 layers, and offset_normalizer is a function of spatial_shapes
        # only (encoder constant). Cache to avoid 3 ops per layer.
        self._offset_normalizer_xy_cache = None
        self._reference_xy_cache = None
        self._reference_xy_cache_ref = None

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
        flag="decoder",
        **kwargs,
    ):
        params = self.params
        _enc_stats = kwargs.pop("_enc_stats", None)
        _t = _enc_sync_now(self.device) if _enc_stats is not None else 0.0
        if value is None:
            assert self.batch_first
            bs, len_bev, c = query.shape
            value = ttnn.stack([query, query], dim=1)
            value = ttnn.reshape(value, (bs * 2, len_bev, c))

        if identity is None:
            identity = query
        if query_pos is not None:
            query = ttnn.add(query, query_pos)

        if not self.batch_first:
            query = ttnn.permute(query, (1, 0, 2))
            value = ttnn.permute(value, (1, 0, 2))

        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        assert self.num_bev_queue == 2

        query = ttnn.concat([value[:bs], query], dim=-1)
        if _enc_stats is not None:
            _enc_record(_enc_stats, "tsa_prep", _t, self.device)
            _t = _enc_sync_now(self.device)

        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        value = ttnn.linear(value, params.value_proj.weight, bias=params.value_proj.bias)
        if key_padding_mask is not None:
            mask = key_padding_mask[..., None]
            value = ttnn.where(mask, ttnn.zeros_like(value), value)

        value = ttnn.reshape(value, (bs * self.num_bev_queue, num_value, self.num_heads, -1))
        if _enc_stats is not None:
            _enc_record(_enc_stats, "tsa_value_proj", _t, self.device)
            _t = _enc_sync_now(self.device)

        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        sampling_offsets = ttnn.linear(query, params.sampling_offsets.weight, bias=params.sampling_offsets.bias)
        sampling_offsets = ttnn.reshape(
            sampling_offsets, (bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2)
        )
        if _enc_stats is not None:
            _enc_record(_enc_stats, "tsa_offsets_lin", _t, self.device)
            _t = _enc_sync_now(self.device)

        attention_weights = ttnn.linear(query, params.attention_weights.weight, bias=params.attention_weights.bias)
        ttnn.deallocate(query)
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        )
        attention_weights = ttnn.softmax(attention_weights, dim=-1)
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points)
        )

        attention_weights = ttnn.permute(attention_weights, (0, 3, 1, 2, 4, 5))
        attention_weights = ttnn.reshape(
            attention_weights, (bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points)
        )

        sampling_offsets = ttnn.permute(sampling_offsets, (0, 3, 1, 2, 4, 5, 6))
        sampling_offsets = ttnn.reshape(
            sampling_offsets, (bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        )
        if _enc_stats is not None:
            _enc_record(_enc_stats, "tsa_attn_lin", _t, self.device)
            _t = _enc_sync_now(self.device)

        if reference_points.shape[-1] == 2:
            bs_r, num_query, num_levels, _ = reference_points.shape
            # `ttnn.divide(sampling_offsets, [w, h])` on bf16 loses precision
            # against the host-fp32 reference (per-module BEV PCC 0.99→0.68).
            # Precompute reciprocal `[1/w, 1/h]` in fp32 and multiply.
            if self._offset_normalizer_xy_cache is None:
                _shapes_torch = ttnn.to_torch(spatial_shapes).to(torch.float32)
                _recip = torch.stack([1.0 / _shapes_torch[..., 1], 1.0 / _shapes_torch[..., 0]], dim=-1)
                _recip = _recip.reshape(1, 1, 1, _recip.shape[0], 1, _recip.shape[1])
                self._offset_normalizer_xy_cache = ttnn.from_torch(
                    _recip, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
                )
            offset_normalizer_xy_recip = self._offset_normalizer_xy_cache
            if self._reference_xy_cache_ref is not reference_points:
                self._reference_xy_cache = ttnn.reshape(reference_points, (bs_r, num_query, 1, num_levels, 1, 2))
                self._reference_xy_cache_ref = reference_points
            reference_xy = self._reference_xy_cache

            sampling_offsets = ttnn.to_layout(sampling_offsets, ttnn.TILE_LAYOUT)
            sampling_locations = ttnn.multiply(sampling_offsets, offset_normalizer_xy_recip)
            sampling_locations = reference_xy + sampling_locations

        elif reference_points.shape[-1] == 4:
            reference_points_reshape = ttnn.reshape(
                reference_points,
                [reference_points.shape[0], reference_points.shape[1], 1, reference_points.shape[2], 1, 2],
            )
            sampling_locations = (
                reference_points_reshape + sampling_offsets / self.num_points * reference_points_reshape * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]} instead."
            )

        # Cache spatial_shapes as Python ints once (host read happens during
        # warm-up, not under trace capture).
        if self._spatial_shapes_list_cache is None:
            num_levels = spatial_shapes.shape[0]
            self._spatial_shapes_list_cache = [
                (int(spatial_shapes[lvl][0].item()), int(spatial_shapes[lvl][1].item())) for lvl in range(num_levels)
            ]
        if _enc_stats is not None:
            _enc_record(_enc_stats, "tsa_locations", _t, self.device)
            _t = _enc_sync_now(self.device)

        output = multi_scale_deformable_attn_pytorch(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
            self.device,
            value_spatial_shapes_list=self._spatial_shapes_list_cache,
            _enc_stats=_enc_stats,
        )
        if _enc_stats is not None:
            _enc_record(_enc_stats, "tsa_msda", _t, self.device)
            _t = _enc_sync_now(self.device)
        ttnn.deallocate(attention_weights)
        # MSDA returned shape (bs * num_bev_queue, num_query, embed_dims). The
        # original code did permute (1,2,0) → reshape → to_memory_config(DRAM) →
        # mean(-1) → permute (2,0,1) to mean across num_bev_queue. That dance
        # cost ~210 ms across 6 layers despite the actual linear being only a
        # few ms; the bulk was layout/permute thrashing. Replace with a single
        # reshape that exposes the bev_queue dim, then mean(dim=1).
        output = ttnn.reshape(output, (bs, self.num_bev_queue, num_query, embed_dims))
        output = ttnn.mean(output, dim=1)
        output = ttnn.linear(output, params.output_proj.weight, bias=params.output_proj.bias)

        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        output = ttnn.add(output, identity)
        if _enc_stats is not None:
            _enc_record(_enc_stats, "tsa_out_proj", _t, self.device)
        return output
