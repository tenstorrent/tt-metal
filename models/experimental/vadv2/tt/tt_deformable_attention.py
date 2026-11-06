# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import warnings
from models.experimental.vadv2.tt.tt_utils import fused_msda, msda_operands
from models.experimental.vadv2.tt.matmul_helpers import linear_flatten_batch


class TtCustomMSDeformableAttention:
    def __init__(
        self,
        params,
        device,
        spatial_shapes_hw,
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

        assert len(spatial_shapes_hw) == num_levels, f"need {num_levels} (H, W) pairs, got {spatial_shapes_hw}"
        self.spatial_shapes_hw = [(int(h), int(w)) for h, w in spatial_shapes_hw]

    def __call__(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
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
        assert sum(h * w for h, w in self.spatial_shapes_hw) == num_value
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)

        value = linear_flatten_batch(value, params.value_proj.weight, bias=params.value_proj.bias)
        if key_padding_mask is not None:
            mask = key_padding_mask[..., None]
            value = ttnn.where(mask, ttnn.zeros_like(value), value)
        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        sampling_offsets, attention_weights = msda_operands(
            query, params, self.num_heads, self.num_levels * self.num_points
        )
        ttnn.deallocate(query)

        if reference_points.shape[-1] != 2:
            raise NotImplementedError(
                f"only 2-D reference points are supported, got {reference_points.shape[-1]} coordinates"
            )
        reference_points = ttnn.reshape(reference_points, (bs, num_query, self.num_levels, 2))
        output = fused_msda(
            value, reference_points, sampling_offsets, attention_weights, self.spatial_shapes_hw, "level"
        )
        output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
        output = linear_flatten_batch(output, params.output_proj.weight, bias=params.output_proj.bias)
        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        output = ttnn.add(output, identity)

        return output
