# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import warnings
import torch.nn.functional as F


def multi_scale_deformable_attn(
    value, value_spatial_shapes, sampling_locations, attention_weights, device, reshape=False
):
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = []
    value_list.append(value)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []

    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level]
        value_l_ = ttnn.reshape(value_l_, [value_l_.shape[0], value_l_.shape[1], value_l_.shape[2] * value_l_.shape[3]])
        value_l_ = ttnn.permute(value_l_, (0, 2, 1))
        if reshape:
            value_l_ = ttnn.reshape(value_l_, [bs * num_heads, embed_dims, 12, 20])
        else:
            value_l_ = ttnn.reshape(value_l_, [bs * num_heads, embed_dims, 100, 100])

        sampling_grid_l_ = sampling_grids[:, :, :, level]
        sampling_grid_l_ = ttnn.permute(sampling_grid_l_, (0, 2, 1, 3, 4))

        sampling_grid_l_ = ttnn.reshape(
            sampling_grid_l_,
            [
                sampling_grid_l_.shape[0] * sampling_grid_l_.shape[1],
                sampling_grid_l_.shape[2],
                sampling_grid_l_.shape[3],
                sampling_grid_l_.shape[4],
            ],
        )
        value_l_ = ttnn.to_torch(value_l_).float()
        sampling_grid_l_ = ttnn.to_torch(sampling_grid_l_).float()
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_l_ = ttnn.from_torch(sampling_value_l_, device=device, dtype=ttnn.bfloat16)
        sampling_value_list.append(sampling_value_l_)

        attention_weights = ttnn.permute(attention_weights, (0, 2, 1, 3, 4))

        attention_weights = ttnn.reshape(attention_weights, [bs * num_heads, 1, num_queries, num_levels * num_points])

    output = ttnn.stack(sampling_value_list, -2)
    output = ttnn.reshape(
        output, [output.shape[0], output.shape[1], output.shape[2], output.shape[3] * output.shape[4]]
    )
    output = output * attention_weights
    output = ttnn.sum(output, 3)
    output = ttnn.reshape(output, [bs, num_heads * embed_dims, num_queries])
    output = ttnn.permute(output, (0, 2, 1))
    ttnn.deallocate(attention_weights)
    ttnn.deallocate(sampling_grids)
    ttnn.deallocate(sampling_value_l_)
    ttnn.deallocate(value)
    return output


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
        # ttnn.deallocate(query_pos)

        if not self.batch_first:
            query = ttnn.permute(query, (1, 0, 2))
            value = ttnn.permute(value, (1, 0, 2))

        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        assert self.num_bev_queue == 2

        query = ttnn.concat([value[:bs], query], dim=-1)

        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        value = ttnn.linear(value, params.value_proj.weight, bias=params.value_proj.bias)
        if key_padding_mask is not None:
            mask = key_padding_mask[..., None]
            value = ttnn.where(mask, ttnn.zeros_like(value), value)

        value = ttnn.reshape(value, (bs * self.num_bev_queue, num_value, self.num_heads, -1))

        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        sampling_offsets = ttnn.linear(query, params.sampling_offsets.weight, bias=params.sampling_offsets.bias)
        sampling_offsets = ttnn.reshape(
            sampling_offsets, (bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2)
        )

        attention_weights = ttnn.linear(query, params.attention_weights.weight, bias=params.attention_weights.bias)
        ttnn.deallocate(params.attention_weights.weight)
        ttnn.deallocate(params.attention_weights.bias)
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

        if reference_points.shape[-1] == 2:
            offset_normalizer = ttnn.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1)
            bs_r, num_query, num_levels, _ = reference_points.shape
            reference_xy = ttnn.reshape(reference_points, (bs_r, num_query, 1, num_levels, 1, 2))
            offset_normalizer_xy = ttnn.reshape(
                offset_normalizer, (1, 1, 1, offset_normalizer.shape[0], 1, offset_normalizer.shape[1])
            )
            sampling_offsets = ttnn.to_layout(sampling_offsets, ttnn.TILE_LAYOUT)
            offset_normalizer_xy = ttnn.to_layout(offset_normalizer_xy, ttnn.TILE_LAYOUT)
            sampling_offsets = ttnn.to_torch(sampling_offsets)
            offset_normalizer_xy = ttnn.to_torch(offset_normalizer_xy)
            sampling_locations = sampling_offsets / offset_normalizer_xy
            reference_xy = ttnn.to_torch(reference_xy)
            sampling_locations = reference_xy + sampling_locations
            sampling_locations = ttnn.from_torch(sampling_locations, device=self.device, dtype=ttnn.bfloat16)

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
        # ttnn.deallocate(reference_points)

        output = multi_scale_deformable_attn(
            value, spatial_shapes, sampling_locations, attention_weights, self.device, reshape=False
        )
        ttnn.deallocate(attention_weights)
        # ttnn.deallocate(reference_xy)
        output = ttnn.permute(output, (1, 2, 0))
        output = ttnn.reshape(output, (num_query, embed_dims, bs, self.num_bev_queue))
        output = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.mean(output, dim=-1)
        output = ttnn.permute(output, (2, 0, 1))
        output = ttnn.linear(output, params.output_proj.weight, bias=params.output_proj.bias)
        ttnn.deallocate(params.output_proj.weight)
        ttnn.deallocate(params.output_proj.bias)

        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        output = ttnn.add(output, identity)
        return output
