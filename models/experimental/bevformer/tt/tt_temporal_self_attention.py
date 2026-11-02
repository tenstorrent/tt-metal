# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Temporal Self Attention (TSA) module for BEVFormer.

This module implements the temporal self-attention mechanism using TTNN operations
to enable BEV features to model temporal dependencies across different timesteps.
It uses deformable attention to aggregate information from current and
historical BEV features.

Based on the reference PyTorch implementation but optimized for TTNN.
"""

import torch
import ttnn
import warnings

# Enable/disable logging output
ENABLE_LOGGING = False

from .tt_common import grid_linear_kwargs, l1_if_fits, signpost
from .tt_ms_deformable_attention import (
    group_sum_compute_config,
    group_sum_matrix,
    grouped_softmax,
    heads_packed_offsets,
    multi_scale_deformable_attn_fused,
)


def shifted_reference_points(reference_points, shift, num_levels=1):
    """The BEV maps' float32 ROW_MAJOR reference points [B, num_queries, 2 * num_levels, 2] for
    ``TTTemporalSelfAttention`` in "level" mode: the previous BEV's ``reference_points``
    [B, num_queries, 1, 2] shifted by ``shift`` [B, 1, 1, 2], once per level, then the queries'.
    They are the same for every layer of a frame, so the encoder builds them once."""
    bs = reference_points.shape[0]
    assert (shift.dtype, shift.layout, tuple(shift.shape)) == (
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
        (bs, 1, 1, 2),
    ), f"shift must be float32 ROW_MAJOR ({bs}, 1, 1, 2), got {shift.dtype} {shift.layout} {tuple(shift.shape)}"
    shifted = ttnn.add(reference_points, shift)
    per_map = ttnn.concat([shifted] * num_levels + [reference_points] * num_levels, dim=2)
    ttnn.deallocate(shifted)
    return per_map


class TTTemporalSelfAttention:
    """
    TTNN Temporal Self Attention module for BEVFormer.

    This attention mechanism models temporal dependencies by allowing BEV queries
    to attend to both current and historical BEV features using deformable attention.
    It's designed to handle temporal relationships for object tracking and motion
    understanding in autonomous driving scenarios.

    Args:
        device: TTNN device for computation
        params: Parameter dict containing weights and biases
        embed_dims (int): The embedding dimension.
        num_heads (int): Number of attention heads.
        num_levels (int): Number of feature levels.
        num_points (int): Number of sampling points in deformable attention.
        num_bev_queue (int): Number of BEV timesteps (typically 2: current + history).
        batch_first (bool): Whether the first dimension of input is batch_size.
        spatial_shapes: BEV grid shape [1, 2] for deformable attention.
        temporal_num_points (int): Sampling points per head and BEV map, overriding ``num_points``,
            which the encoder layer sets to the spatial cross-attention's (BEVFormer-base: 4 and 8).
        residual (bool): Add the identity to the output; the encoder layer adds it inside the
            LayerNorm that follows instead.
        **kwargs: Additional arguments.

    The fused op samples the BEV maps as ``num_bev_queue`` levels of one value, each map's keys
    after the previous one's: the Linears' head-major channels, (head, map, level, point), are
    already the op's (head, level, point), and with an ego shift each map gets its own float32
    reference points, the previous one's shifted. The op sums over the maps, so the output projection's
    weights carry the mean's 1 / num_bev_queue.
    """

    def __init__(
        self,
        device,
        params,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 1,
        num_points: int = 4,
        num_bev_queue: int = 2,
        batch_first: bool = True,
        *,
        spatial_shapes,
        temporal_num_points=None,
        residual: bool = True,
        **kwargs,
    ):
        self.device = device
        self.residual = residual
        self.params = params
        if temporal_num_points is not None:
            num_points = temporal_num_points

        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, " f"but got {embed_dims} and {num_heads}")
        # The first map is the previous BEV, the one the shift moves, as in upstream's stack.
        if num_bev_queue != 2:
            raise ValueError(f"num_bev_queue must be 2 (the previous BEV and the queries), got {num_bev_queue}")

        dim_per_head = embed_dims // num_heads
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.batch_first = batch_first

        # Check if dim_per_head is power of 2 for efficiency
        if not self._is_power_of_2(dim_per_head):
            warnings.warn(
                "For optimal performance with TTNN, embed_dims should be set "
                "so that dimension of each attention head is a power of 2"
            )

        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long)
        if tuple(spatial_shapes.shape) != (num_levels, 2):
            raise ValueError(f"spatial_shapes must have shape ({num_levels}, 2), got {tuple(spatial_shapes.shape)}")
        # The fused op's levels: every BEV map's, map-major.
        self.map_spatial_shapes = spatial_shapes.repeat(num_bev_queue, 1)

        # Softmax over each head's (level, point) samples, per BEV map.
        self.attention_group_sum = group_sum_matrix(num_heads * num_bev_queue, num_levels * num_points, device)
        self.attention_group_sum_config = group_sum_compute_config(device)
        self.linear_kwargs = grid_linear_kwargs(device)

    def forward(
        self,
        query,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        prev_bev=None,
        shift=None,
        **kwargs,
    ):
        """
        Forward pass of TTNN Temporal Self Attention.

        Args:
            query: Current BEV queries [B, num_queries, embed_dims].
            value: The previous BEV and the encoder's input queries stacked per sample,
                [B*num_bev_queue, num_queries, embed_dims], as upstream's encoder builds it.
                If None (no previous BEV), the queries stacked with themselves.
            identity: Identity connection input.
            query_pos: Query positional encoding.
            key_padding_mask: Unused; upstream's self-attention never masks the BEV.
            reference_points: BEV cell centres [B, num_queries, 1, 2] in [0, 1], or with a previous
                BEV in ``value`` every BEV map's [B, num_queries, num_bev_queue * num_levels, 2] from
                :func:`shifted_reference_points`, already shifted, with ``shift`` None.
            prev_bev: Unused; the previous BEV comes in ``value``.
            shift: Ego translation in BEV fractions, float32 ROW_MAJOR [B, 1, 1, 2], added to the
                previous BEV's float32 reference points; None uses ``reference_points`` as given.
            **kwargs: Additional arguments.

        Returns:
            Output features [B, num_queries, embed_dims].

        With ``batch_first`` False, ``query``, ``value``, ``query_pos``, ``identity`` and the output
        are sequence-first instead.
        """
        signpost(header="TTNN TSA Forward Start")

        if not self.batch_first:
            # The features to batch-first; the reference points and the shift are batch-first already.
            query, value, query_pos, identity = (
                None if t is None else ttnn.permute(t, (1, 0, 2)) for t in (query, value, query_pos, identity)
            )

        bs, num_queries, embed_dims = query.shape
        queue = self.num_bev_queue
        levels = queue * self.num_levels
        params = self.params
        # Live at once, in (B, nq, embed_dims) units: the staged value (queue), the concatenated
        # offset query (2) and the query plus its positions (1).
        element_size = query.element_size()
        memory = l1_if_fits(self.device, bs * num_queries * embed_dims * element_size * (queue + 3))
        # Upstream shifts the reference points only toward a previous BEV.
        if value is None:
            shift = None

        assert self.residual or identity is None, "built with residual=False, so the identity would be dropped"
        if identity is None:
            identity = query

        def linear(x, linear_params, memory_config=memory):
            return ttnn.linear(
                x, linear_params.weight, bias=linear_params.bias, memory_config=memory_config, **self.linear_kwargs
            )

        # Every map's values, map-major along the keys, (B, queue * nk, embed_dims), row-major and
        # in L1 when it fits, as the fused op reads them. Staged first: L1 is allocated top-down and the fused op's
        # output and circular buffers must fit below the lowest live buffer, so the value takes the
        # top and the intermediates staged after it, all but the op's own offsets and weights, are
        # freed before the op runs.
        if value is None:
            projected = ttnn.linear(query, params.value_proj.weight, bias=params.value_proj.bias)
            maps = ttnn.concat([projected] * queue, dim=1)
            ttnn.deallocate(projected)
        else:
            # Upstream's (B * queue, nk, embed_dims) stack is each sample's maps one after another.
            maps = ttnn.reshape(
                ttnn.linear(value, params.value_proj.weight, bias=params.value_proj.bias),
                (bs, queue * num_queries, embed_dims),
            )
        staged_value = ttnn.to_layout(maps, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory)
        ttnn.deallocate(maps)

        # The offsets and attention weights read the previous BEV and the queries concatenated,
        # value[:bs] as upstream: each sample's previous BEV only at B=1, BEVFormer's inference
        # batch. Without one, upstream stacks the queries per sample before taking value[:bs].
        if value is not None:
            first_map = value[:bs]
        elif bs == 1:
            first_map = query
        else:
            stacked = ttnn.concat([ttnn.unsqueeze(query, 1)] * queue, dim=1)
            first_map = ttnn.reshape(stacked, (bs * queue, num_queries, embed_dims))[:bs]
        offsets_query = ttnn.add(query, query_pos, memory_config=memory) if query_pos is not None else query
        offsets_query_cat = ttnn.concat([first_map, offsets_query], dim=-1, memory_config=memory)
        if offsets_query is not query:
            ttnn.deallocate(offsets_query)

        sampling_offsets = linear(offsets_query_cat, params.sampling_offsets)
        attention_weights = linear(offsets_query_cat, params.attention_weights)
        ttnn.deallocate(offsets_query_cat)
        attention_weights = grouped_softmax(
            attention_weights, self.attention_group_sum, self.attention_group_sum_config
        )

        # The fused op reads a query's heads of offsets from one row-major page when each head's
        # offsets fill whole units of the buffer's alignment, else from a page per head; a head's
        # attention weights, half as many values, get a page of their own, padded to the alignment.
        sampling_offsets = heads_packed_offsets(
            ttnn.to_layout(sampling_offsets, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory), self.num_heads
        )
        attention_weights = ttnn.reshape(
            ttnn.to_layout(attention_weights, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory),
            (bs, num_queries, self.num_heads, levels * self.num_points),
            memory_config=memory,
        )

        # BEV cell centres, float32 and ROW_MAJOR, where the trailing (R, 2) is not padded to a tile.
        # Each map's own, the previous one's shifted (from shifted_reference_points, as the encoder
        # hands them, or built here from a shift), go to "level" mode. Grid points alone go to
        # "pillar" mode, where one point serves every (level, point).
        ref = ttnn.to_layout(reference_points, ttnn.ROW_MAJOR_LAYOUT)
        if ref.dtype != ttnn.float32:
            ref = ttnn.typecast(ref, ttnn.float32)
        reference_mode = "pillar"
        if reference_points.shape[2] == levels:
            assert shift is None, "reference points per BEV map already carry the shift"
            assert value is not None, "reference points per BEV map shift the previous BEV, so pass it in value"
            reference_mode = "level"
        else:
            ref = ttnn.reshape(ref, (bs, num_queries, 1, 2))
            if shift is not None:
                ref = shifted_reference_points(ref, shift, self.num_levels)
                reference_mode = "level"

        output = multi_scale_deformable_attn_fused(
            value=staged_value,
            reference_points=ref,
            sampling_offsets=sampling_offsets,
            attention_weights=attention_weights,
            spatial_shapes=self.map_spatial_shapes,
            reference_mode=reference_mode,
        )
        for tensor in (staged_value, sampling_offsets, attention_weights):
            ttnn.deallocate(tensor)

        tiled = ttnn.to_layout(output, ttnn.TILE_LAYOUT, memory_config=memory)
        ttnn.deallocate(output)
        output = linear(tiled, params.output_proj_mean)
        ttnn.deallocate(tiled)
        if self.residual:
            output = ttnn.add(output, identity)
        if not self.batch_first:
            # Sequence-first pads the batch to a tile: DRAM, not L1.
            output = ttnn.permute(output, (1, 0, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)

        signpost(header="TTNN TSA Forward End")

        return output

    @staticmethod
    def _is_power_of_2(n: int) -> bool:
        """Check if a number is a power of 2."""
        return (n != 0) and (n & (n - 1) == 0)

    def __call__(self, *args, **kwargs):
        """Make the class callable"""
        return self.forward(*args, **kwargs)

    def extra_repr(self) -> str:
        """String representation for debugging"""
        return (
            f"embed_dims={self.embed_dims}, num_heads={self.num_heads}, "
            f"num_levels={self.num_levels}, num_points={self.num_points}, "
            f"num_bev_queue={self.num_bev_queue}, batch_first={self.batch_first}"
        )
