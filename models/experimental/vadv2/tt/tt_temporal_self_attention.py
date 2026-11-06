# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import warnings
from models.experimental.vadv2.tt.tt_utils import fused_msda, msda_operands
from models.experimental.vadv2.tt.matmul_helpers import linear_flatten_batch


class TtTemporalSelfAttention:
    def __init__(
        self,
        device,
        params,
        spatial_shapes_hw,
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

        assert len(spatial_shapes_hw) == num_levels, f"need {num_levels} (H, W) pairs, got {spatial_shapes_hw}"
        self.spatial_shapes_hw = [(int(h), int(w)) for h, w in spatial_shapes_hw]
        # The fused op samples the previous BEV and the queries as num_bev_queue levels of one
        # value, map-major, and sums over them; the mean's 1 / num_bev_queue goes into the
        # output projection's weight.
        self.map_spatial_shapes_hw = self.spatial_shapes_hw * num_bev_queue
        self.output_proj_mean_weight = ttnn.mul(params.output_proj.weight, 1.0 / num_bev_queue)

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
        """reference_points: (bs, Q, num_bev_queue * num_levels, 2), every map's own, the
        previous BEV's shifted by the ego motion; value: the previous BEV and the queries
        stacked per sample, (bs * num_bev_queue, Q, C), or None without a previous BEV."""
        params = self.params
        assert self.num_bev_queue == 2
        if identity is None:
            identity = query
        if not self.batch_first:
            query = ttnn.permute(query, (1, 0, 2))
            if value is not None:
                value = ttnn.permute(value, (1, 0, 2))

        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        bs, num_query, embed_dims = query.shape
        queue = self.num_bev_queue
        assert sum(h * w for h, w in self.spatial_shapes_hw) == num_query

        if value is None:
            projected = linear_flatten_batch(query, params.value_proj.weight, bias=params.value_proj.bias)
            maps = ttnn.concat([projected] * queue, dim=1)
            first_map = query
        else:
            value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
            # Each sample's maps one after another, previous BEV first, as the encoder stacks them.
            maps = ttnn.reshape(
                linear_flatten_batch(value, params.value_proj.weight, bias=params.value_proj.bias),
                (bs, queue * num_query, embed_dims),
            )
            first_map = value[:bs]

        offsets_query = ttnn.add(query, query_pos) if query_pos is not None else query
        offsets_query = ttnn.concat([first_map, offsets_query], dim=-1)
        offsets_query = ttnn.to_layout(offsets_query, ttnn.TILE_LAYOUT)
        sampling_offsets, attention_weights = msda_operands(
            offsets_query, params, self.num_heads, self.num_levels * self.num_points
        )
        ttnn.deallocate(offsets_query)

        assert tuple(reference_points.shape) == (bs, num_query, queue * self.num_levels, 2), (
            f"reference points must be every map's ({bs}, {num_query}, {queue * self.num_levels}, 2), "
            f"got {tuple(reference_points.shape)}"
        )
        output = fused_msda(
            maps, reference_points, sampling_offsets, attention_weights, self.map_spatial_shapes_hw, "level"
        )
        output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
        output = linear_flatten_batch(output, self.output_proj_mean_weight, bias=params.output_proj.bias)

        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        output = ttnn.add(output, identity)
        return output
