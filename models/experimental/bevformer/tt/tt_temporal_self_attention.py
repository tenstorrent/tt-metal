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

import ttnn
import warnings

# Enable/disable logging output
ENABLE_LOGGING = False

from .tt_common import signpost
from .tt_ms_deformable_attention import TTMSDeformableAttention, multi_scale_deformable_attn_fused
from ..model_config import DeformableAttentionConfig


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
        **kwargs: Additional arguments.

    The parameters must order the offset and attention Linears' output channels queue-major
    (``model_preprocessing.create_temporal_self_attention_parameters``), so each BEV map's channels
    are one contiguous half of the row.
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
        **kwargs,
    ):
        self.device = device
        self.params = params
        if temporal_num_points is not None:
            num_points = temporal_num_points

        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, " f"but got {embed_dims} and {num_heads}")

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

        # Initialize TTNN MSDeformableAttention using existing implementation
        deform_config = DeformableAttentionConfig(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            batch_first=batch_first,
        )

        # Initialize TTNN MSDeformableAttention using existing implementation
        # Pass params directly since TSA doesn't have its own parameters beyond deformable attention
        self.deformable_attention = TTMSDeformableAttention(
            deform_config, device, self.params, spatial_shapes=spatial_shapes
        )

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
            reference_points: BEV cell centres [B, num_queries, 1, 2] in [0, 1].
            prev_bev: Unused; the previous BEV comes in ``value``.
            shift: Ego translation in BEV fractions, float32 ROW_MAJOR [B, 1, 1, 2], applied to the
                previous BEV's reference points; None leaves them unshifted.
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
        # Upstream shifts the reference points only toward a previous BEV.
        if value is None:
            shift = None
            value = ttnn.reshape(
                ttnn.concat([ttnn.unsqueeze(query, 1)] * queue, dim=1), (bs * queue, num_queries, embed_dims)
            )

        if identity is None:
            identity = query

        if query_pos is not None:
            query = ttnn.add(query, query_pos)

        # Upstream's hybrid reference points, per sample the previous BEV's shifted by the ego
        # motion, then the current queries' as they are: [B*num_bev_queue, num_queries, 1, 2].
        # float32 and ROW_MAJOR, where the trailing (1, 2) is not padded to a tile.
        ref = ttnn.reshape(ttnn.to_layout(reference_points, ttnn.ROW_MAJOR_LAYOUT), (bs, num_queries, 2))
        if ref.dtype != ttnn.float32:
            ref = ttnn.typecast(ref, ttnn.float32)
        if shift is not None:
            assert (shift.dtype, shift.layout, tuple(shift.shape)) == (
                ttnn.float32,
                ttnn.ROW_MAJOR_LAYOUT,
                (bs, 1, 1, 2),
            ), f"shift must be float32 ROW_MAJOR (bs, 1, 1, 2), got {shift.dtype} {shift.layout} {tuple(shift.shape)}"
        shifted = ref if shift is None else ttnn.add(ref, ttnn.reshape(shift, (bs, 1, 2)))
        ref = ttnn.reshape(ttnn.concat([shifted, ref], dim=1), (bs * queue, num_queries, 1, 2))

        # The offsets and attention weights read the previous BEV and the queries concatenated,
        # value[:bs] as upstream: each sample's previous BEV only at B=1, BEVFormer's inference batch.
        query = ttnn.concat([value[:bs], query], dim=-1)
        value = ttnn.linear(value, self.params.value_proj.weight, bias=self.params.value_proj.bias)

        # Queue-major channels to the batch, in ROW_MAJOR, where the queue axis never lands in a
        # tiled dimension: (B, nq, queue * heads * width) -> (B*queue, nq, heads, width).
        def split_queue(tensor):
            width = tensor.shape[-1] // (queue * self.num_heads)
            tensor = ttnn.reshape(ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT), (bs, num_queries, queue, -1))
            return ttnn.reshape(ttnn.permute(tensor, (0, 2, 1, 3)), (bs * queue, num_queries, self.num_heads, width))

        sampling_offsets = ttnn.linear(
            query, self.params.sampling_offsets.weight, bias=self.params.sampling_offsets.bias
        )
        attention_weights = ttnn.linear(
            query, self.params.attention_weights.weight, bias=self.params.attention_weights.bias
        )
        # Softmax over each head's (level, point) samples, per BEV map.
        attention_weights = ttnn.softmax(
            ttnn.reshape(
                attention_weights, (bs, num_queries, queue * self.num_heads, self.num_levels * self.num_points)
            ),
            dim=-1,
        )
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_queries, queue * self.num_heads * self.num_levels * self.num_points)
        )

        output = multi_scale_deformable_attn_fused(
            value=value,
            reference_points=ref,
            sampling_offsets=split_queue(sampling_offsets),
            attention_weights=split_queue(attention_weights),
            spatial_shapes=self.deformable_attention.spatial_shapes,
        )

        # The mean over the two BEV maps' samples, then the output projection and the residual.
        # In DRAM: the fused op leaves its output in L1 when it fits, and the mean's transpose
        # of the base grid's two maps does not.
        output = ttnn.reshape(output, (bs, queue, num_queries, embed_dims))
        output = ttnn.mean(ttnn.to_layout(output, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG), dim=1)
        output = ttnn.linear(output, self.params.output_proj.weight, bias=self.params.output_proj.bias)
        output = ttnn.add(output, identity)
        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

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
