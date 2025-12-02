# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import warnings
import os
from models.experimental.vadv2.tt.tt_utils import multi_scale_deformable_attn

try:
    from tracy import signpost

    use_signpost = os.getenv("USE_SIGNPOST", "False").lower() in ("true", "1", "yes")
except ModuleNotFoundError:
    use_signpost = False


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
        if use_signpost:
            signpost(header="TtTemporalSelfAttention_call_start")
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

        # Ensure both tensors have the same layout for concat
        value = ttnn.to_layout(value, query.layout)
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
        sampling_offsets = ttnn.reallocate(sampling_offsets)

        attention_weights = ttnn.linear(query, params.attention_weights.weight, bias=params.attention_weights.bias)
        # Don't deallocate model weights - they need to persist across iterations
        # ttnn.deallocate(params.attention_weights.weight)
        # ttnn.deallocate(params.attention_weights.bias)
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
            offset_normalizer = ttnn.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1)
            bs_r, num_query, num_levels, _ = reference_points.shape
            reference_points_shape = reference_points.shape
            reference_points = ttnn.reshape(reference_points, (bs_r, num_query, 1, num_levels, 1, 2))
            offset_normalizer_xy = ttnn.reshape(
                offset_normalizer, (1, 1, 1, offset_normalizer.shape[0], 1, offset_normalizer.shape[1])
            )
            ttnn.deallocate(offset_normalizer)
            sampling_offsets = ttnn.to_layout(sampling_offsets, ttnn.TILE_LAYOUT)
            offset_normalizer_xy = ttnn.to_layout(offset_normalizer_xy, ttnn.TILE_LAYOUT)

            sampling_offsets_shape = sampling_offsets.shape
            sampling_offsets = ttnn.reshape(
                sampling_offsets, (sampling_offsets.shape[0], -1, sampling_offsets.shape[4], sampling_offsets.shape[5])
            )  # [2, 10000*8*1, 4, 2]
            offset_normalizer_xy = ttnn.reshape(
                offset_normalizer_xy,
                (
                    offset_normalizer_xy.shape[0],
                    offset_normalizer_xy.shape[1],
                    offset_normalizer_xy.shape[2],
                    offset_normalizer_xy.shape[-1],
                ),
            )
            sampling_locations = ttnn.div(sampling_offsets, offset_normalizer_xy)
            sampling_locations = ttnn.reshape(sampling_locations, sampling_offsets_shape)
            sampling_locations = reference_points + sampling_locations
            ttnn.deallocate(offset_normalizer_xy)
            reference_points = ttnn.reshape(reference_points, reference_points_shape)
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
        output = multi_scale_deformable_attn(value, spatial_shapes, sampling_locations, attention_weights, self.device)
        ttnn.deallocate(attention_weights)
        ttnn.deallocate(sampling_locations)
        ttnn.deallocate(sampling_offsets)
        ttnn.deallocate(value)

        # Optimization for bs=1: eliminate expensive reshape by averaging directly
        # output shape from multi_scale_deformable_attn: [bs*num_bev_queue, num_query, embed_dims]
        if bs == 1:
            # Fast path for bs=1: mean directly without reshape
            # Shape: [bs*num_bev_queue, num_query, embed_dims] = [2, 10000, 256]
            output = ttnn.permute(output, (1, 2, 0))  # [num_query, embed_dims, num_bev_queue]
            # Shape: [10000, 256, 2]

            output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)

            chunk_size = 1024  # Process 1024 queries at a time
            num_chunks = (num_query + chunk_size - 1) // chunk_size
            mean_output = None

            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, num_query)

                # Slice chunk: [chunk_len, embed_dims, num_bev_queue]
                output_chunk = ttnn.slice(
                    output,
                    [chunk_start, 0, 0],
                    [chunk_end, embed_dims, self.num_bev_queue],
                )

                # Mean over temporal dimension with L1 memory
                output_chunk = ttnn.mean(output_chunk, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
                # Shape: [chunk_len, embed_dims]

                # Move to DRAM for accumulation
                output_chunk = ttnn.to_memory_config(output_chunk, ttnn.DRAM_MEMORY_CONFIG)

                # Accumulate chunks
                if chunk_idx == 0:
                    mean_output = output_chunk
                else:
                    mean_output = ttnn.concat([mean_output, output_chunk], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    ttnn.deallocate(output_chunk)

            ttnn.deallocate(output)
            output = mean_output
            # Shape: [num_query, embed_dims] = [10000, 256]

            # Expand to include bs dimension for compatibility with downstream code
            output = ttnn.unsqueeze(output, 0)  # [bs, num_query, embed_dims]
            # Shape: [1, 10000, 256]
        else:
            # General path for bs > 1: need reshape to separate batch and temporal dimensions
            tmp = output
            output = ttnn.permute(tmp, (1, 2, 0))
            ttnn.deallocate(tmp)
            # Shape: [num_query, embed_dims, bs*num_bev_queue]

            tmp = output
            output = ttnn.reshape(tmp, (num_query, embed_dims, bs, self.num_bev_queue))
            ttnn.deallocate(tmp)
            # Shape: [num_query, embed_dims, bs, num_bev_queue]

            # Swap dims 1 and 2 for more efficient tiled layout
            output = ttnn.permute(output, (0, 2, 1, 3))  # [num_query, bs, embed_dims, num_bev_queue]
            output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)

            # Process in chunks along num_query dimension
            chunk_size = 1024  # Process 1024 queries at a time
            num_chunks = (num_query + chunk_size - 1) // chunk_size
            mean_output = None

            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, num_query)

                # Slice chunk: [chunk_len, bs, embed_dims, num_bev_queue]
                output_chunk = ttnn.slice(
                    output,
                    [chunk_start, 0, 0, 0],
                    [chunk_end, bs, embed_dims, self.num_bev_queue],
                )

                # Mean over temporal dimension with L1 memory
                output_chunk = ttnn.mean(output_chunk, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
                # Shape: [chunk_len, bs, embed_dims]

                # Move to DRAM for accumulation
                output_chunk = ttnn.to_memory_config(output_chunk, ttnn.DRAM_MEMORY_CONFIG)

                # Accumulate chunks
                if chunk_idx == 0:
                    mean_output = output_chunk
                else:
                    mean_output = ttnn.concat([mean_output, output_chunk], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    ttnn.deallocate(output_chunk)

            ttnn.deallocate(output)
            output = mean_output
            # Shape: [num_query, bs, embed_dims]

            output = ttnn.permute(output, (1, 0, 2))  # [bs, num_query, embed_dims]

        output = ttnn.linear(output, params.output_proj.weight, bias=params.output_proj.bias)
        # Don't deallocate model weights - they need to persist across iterations
        # ttnn.deallocate(params.output_proj.weight)
        # ttnn.deallocate(params.output_proj.bias)

        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        output = ttnn.add(output, identity)
        ttnn.deallocate(identity)
        if use_signpost:
            signpost(header="TtTemporalSelfAttention_call_end")
        return output
