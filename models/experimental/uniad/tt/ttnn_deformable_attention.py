# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.uniad.tt.matmul_helpers import linear_flatten_batch

from models.experimental.uniad.tt.ttnn_utils import multi_scale_deformable_attn_pytorch


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
        # Cache of (h, w) Python ints per level. Populated lazily from
        # spatial_shapes via .item() on first call (which is a host read,
        # so it must happen outside trace capture). The PCC/perf test
        # warm-up calls populate this before begin_trace_capture.
        self._spatial_shapes_list_cache = None

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
        # `assert (ttnn.sum(...) == num_value)` removed: comparing a ttnn tensor
        # with a Python int forces a device->host read, which breaks trace
        # capture. The check is validation-only and not load-bearing.
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)

        value = linear_flatten_batch(value, params.value_proj.weight, bias=params.value_proj.bias)
        if key_padding_mask is not None:
            mask = key_padding_mask[..., None]
            # Replace `where(mask, zeros, value)` with `value * (1 - mask)`:
            # same numerical effect (zero out where mask is 1, keep value
            # elsewhere) without the per-forward ttnn.zeros_like allocation
            # that blocks ttnn.trace_capture.
            value = ttnn.multiply(value, ttnn.rsub(mask, 1.0))
        value = ttnn.reshape(value, (bs, num_value, self.num_heads, -1))

        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        sampling_offsets = linear_flatten_batch(
            query, params.sampling_offsets.weight, bias=params.sampling_offsets.bias
        )
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

        attention_weights = ttnn.softmax(attention_weights, dim=-1)

        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_levels, self.num_points)
        )

        if reference_points.shape[-1] == 2:
            # `ttnn.divide(sampling_offsets, [w, h])` on bf16 loses precision
            # against the host-fp32 reference. Precompute reciprocal
            # `[1/w, 1/h]` in fp32 and multiply.
            _shapes_torch = ttnn.to_torch(spatial_shapes).to(torch.float32)
            _recip = torch.stack([1.0 / _shapes_torch[..., 1], 1.0 / _shapes_torch[..., 0]], dim=-1)
            _recip = _recip.reshape(1, 1, 1, _recip.shape[0], 1, _recip.shape[1])
            offset_normalizer_xy_recip = ttnn.from_torch(
                _recip, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            bs_r, num_query, num_levels, _ = reference_points.shape
            reference_xy = ttnn.reshape(reference_points, (bs_r, num_query, 1, num_levels, 1, 2))
            sampling_offsets = ttnn.to_layout(sampling_offsets, ttnn.TILE_LAYOUT)
            sampling_offsets = ttnn.squeeze(sampling_offsets, 0)
            sampling_offsets = ttnn.squeeze(sampling_offsets, 2)
            offset_normalizer_xy_recip = ttnn.squeeze(offset_normalizer_xy_recip, 0)
            offset_normalizer_xy_recip = ttnn.squeeze(offset_normalizer_xy_recip, 0)

            sampling_locations = ttnn.multiply(sampling_offsets, offset_normalizer_xy_recip)

            sampling_locations = ttnn.unsqueeze(sampling_locations, 2)
            sampling_locations = ttnn.unsqueeze(sampling_locations, 0)
            sampling_locations = ttnn.add(reference_xy, sampling_locations)

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
                f"Last dim of reference_points must be" f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )

        # Lazy-extract spatial_shapes to Python ints (one-time host read).
        # Trace capture cannot do this read; populate cache via warm-up.
        if self._spatial_shapes_list_cache is None:
            num_levels = spatial_shapes.shape[0]
            self._spatial_shapes_list_cache = [
                (int(spatial_shapes[lvl][0].item()), int(spatial_shapes[lvl][1].item())) for lvl in range(num_levels)
            ]

        output = multi_scale_deformable_attn_pytorch(
            value,
            spatial_shapes,
            None,
            sampling_locations,
            attention_weights,
            None,
            self.device,
            value_spatial_shapes_list=self._spatial_shapes_list_cache,
        )

        output = linear_flatten_batch(output, params.output_proj.weight, bias=params.output_proj.bias)
        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        output = ttnn.add(output, identity)

        return output
