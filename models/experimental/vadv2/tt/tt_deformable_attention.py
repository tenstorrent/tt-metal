# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import warnings
from models.experimental.vadv2.tt.tt_utils import multi_scale_deformable_attn, build_folded_sampling_offsets
from models.experimental.vadv2.tt.matmul_helpers import linear_flatten_batch


class TtCustomMSDeformableAttention:
    def __init__(
        self,
        params,
        device,
        embed_dims=256,
        num_heads=8,
        num_levels=1,
        num_points=4,
        im2col_step=192,
        dropout=0.1,
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
        flag="decoder",
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
        # NOTE: a `ttnn.sum(spatial_shapes[:,0]*spatial_shapes[:,1]) == num_value`
        # sanity assert used to live here, but `== num_value` forces a host-side
        # `__bool__` sync on a device tensor every call — a Metal-Trace capture
        # blocker. spatial_shapes is static, so the invariant is structural.
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
        ttnn.deallocate(query)
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_levels * self.num_points)
        )
        attention_weights = ttnn.softmax(attention_weights, -1)

        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_levels, self.num_points)
        )

        if reference_points.shape[-1] == 2:
            bs_r, num_query, num_levels, _ = reference_points.shape
            reference_xy = ttnn.reshape(reference_points, (bs_r, num_query, 1, num_levels, 1, 2))
            sampling_offsets = ttnn.to_layout(sampling_offsets, ttnn.TILE_LAYOUT)
            sampling_offsets = ttnn.squeeze(sampling_offsets, 0)
            sampling_offsets = ttnn.squeeze(sampling_offsets, 2)
            # sampling_offsets is already divided by offset_normalizer (folded into
            # the sampling_offsets Linear weight).
            sampling_locations = sampling_offsets

            sampling_locations = ttnn.unsqueeze(sampling_locations, 2)
            sampling_locations = ttnn.unsqueeze(sampling_locations, 0)
            sampling_locations = ttnn.add(reference_xy, sampling_locations)

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

        output = linear_flatten_batch(output, params.output_proj.weight, bias=params.output_proj.bias)
        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        output = ttnn.add(output, identity)

        return output
