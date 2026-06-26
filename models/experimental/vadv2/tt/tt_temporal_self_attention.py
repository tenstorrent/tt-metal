# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import warnings
from models.experimental.vadv2.tt.tt_utils import multi_scale_deformable_attn, build_folded_sampling_offsets
from models.experimental.vadv2.tt.matmul_helpers import linear_flatten_batch


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
        dropout=0.1,
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

        def _is_power_of_2(n):
            if not isinstance(n, int) or n < 0:
                raise ValueError(f"invalid input for _is_power_of_2: {n} (type: {type(n)})")
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "For optimal performance with TTNN, embed_dims should be set "
                "so that dimension of each attention head is a power of 2"
            )

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue

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
        flag="decoder",
        **kwargs,
    ):
        params = self.params
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

        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        value = linear_flatten_batch(value, params.value_proj.weight, bias=params.value_proj.bias)
        if key_padding_mask is not None:
            mask = key_padding_mask[..., None]
            value = ttnn.where(mask, ttnn.zeros_like(value), value)

        value = ttnn.reshape(value, (bs * self.num_bev_queue, num_value, self.num_heads, -1))

        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)

        if self._so_w is None:
            self._so_w, self._so_b = build_folded_sampling_offsets(params.sampling_offsets, spatial_shapes, self.device)
        sampling_offsets = linear_flatten_batch(query, self._so_w, bias=self._so_b)
        sampling_offsets = ttnn.reshape(
            sampling_offsets, (bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2)
        )
        sampling_offsets = ttnn.reallocate(sampling_offsets)

        attention_weights = linear_flatten_batch(
            query, params.attention_weights.weight, bias=params.attention_weights.bias
        )
        ttnn.deallocate(query)
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        )

        attention_weights = ttnn.softmax(attention_weights, -1)
        attention_weights = ttnn.reallocate(attention_weights)
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points)
        )

        attention_weights = ttnn.permute(attention_weights, (0, 3, 1, 2, 4, 5))
        attention_weights = ttnn.reallocate(attention_weights)
        attention_weights = ttnn.reshape(
            attention_weights, (bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points)
        )

        sampling_offsets = ttnn.permute(sampling_offsets, (0, 3, 1, 2, 4, 5, 6))
        sampling_offsets = ttnn.reallocate(sampling_offsets)
        sampling_offsets = ttnn.reshape(
            sampling_offsets, (bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        )

        if reference_points.shape[-1] == 2:
            bs_r, num_query, num_levels, _ = reference_points.shape
            reference_points_shape = reference_points.shape
            reference_points = ttnn.reshape(reference_points, (bs_r, num_query, 1, num_levels, 1, 2))
            sampling_offsets = ttnn.to_layout(sampling_offsets, ttnn.TILE_LAYOUT)
            # sampling_offsets is already divided by offset_normalizer (folded into
            # the sampling_offsets Linear weight), so add reference directly.
            sampling_locations = reference_points + sampling_offsets
            reference_points = ttnn.reshape(reference_points, reference_points_shape)
        elif reference_points.shape[-1] == 4:
            # 4-D (box-refine) reference path. Its formula
            # (reference + sampling_offsets / num_points * reference * 0.5) consumes
            # sampling_offsets WITHOUT the offset_normalizer division — but offsets
            # are now pre-scaled by 1/[W,H] folded into the sampling_offsets Linear
            # weights, so this path would silently miscompute. It is unused in VADv2
            # (callers pass 2-D reference_points); guard loudly for future enablement.
            raise NotImplementedError(
                "4-D reference_points is incompatible with the folded offset_normalizer "
                "(see build_folded_sampling_offsets); this path needs unscaled sampling_offsets."
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]} instead."
            )
        if self._hw_cache is None:
            self._hw_cache = [
                (int(spatial_shapes[lvl, 0].item()), int(spatial_shapes[lvl, 1].item()))
                for lvl in range(self.num_levels)
            ]
        output = multi_scale_deformable_attn(
            value, spatial_shapes, sampling_locations, attention_weights, self.device, hw_py=self._hw_cache
        )
        ttnn.deallocate(attention_weights)
        ttnn.deallocate(sampling_locations)
        ttnn.deallocate(sampling_offsets)
        ttnn.deallocate(value)
        output = ttnn.permute(output, (1, 2, 0))
        output = ttnn.reshape(output, (num_query, embed_dims, bs * self.num_bev_queue))
        output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
        output = ttnn.mean(output, dim=-1, keepdim=True)
        output = ttnn.permute(output, (2, 0, 1))
        output = linear_flatten_batch(output, params.output_proj.weight, bias=params.output_proj.bias)

        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        output = ttnn.add(output, identity)
        ttnn.deallocate(identity)
        return output
