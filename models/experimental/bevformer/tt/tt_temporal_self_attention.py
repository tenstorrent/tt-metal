# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Temporal Self Attention (TSA) module for BEVFormer.

This module implements the temporal self-attention mechanism using TTNN operations
to enable BEV features to model temporal dependencies across different timesteps.
It uses deformable attention to aggregate information from current and
historical BEV features.

Based on the reference PyTorch implementation but optimized for TTNN.

The value stacks two BEV maps per sample, the previous BEV and the current queries, as upstream's
encoder builds it. The offsets and attention weights read both maps concatenated and give one set
of points per map; the two sampled results are averaged. The parameters must come from
``model_preprocessing.create_temporal_self_attention_parameters``, which orders the Linears' output
channels queue-major, so each map's channels are a contiguous half of the row.
"""

import torch
import ttnn
import warnings
from loguru import logger

# Enable/disable logging output
ENABLE_LOGGING = False

from .tt_ms_deformable_attention import multi_scale_deformable_attn_fused

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


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
        **kwargs: Additional arguments.
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
        **kwargs,
    ):
        self.device = device
        self.params = params

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

        if num_levels != 1 or num_bev_queue != 2:
            raise ValueError("the value stacks two maps of one BEV level: the previous BEV and the current query")
        # The Linears are this module's own, not TTMSDeformableAttention's: they read both maps
        # concatenated, and the core attention samples each map with its own half of their channels.
        self.spatial_shapes = torch.as_tensor(spatial_shapes)

    def forward(
        self,
        query,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        prev_bev=None,
        **kwargs,
    ):
        """
        Forward pass of TTNN Temporal Self Attention.

        Args:
            query: Current BEV queries [B, num_queries, embed_dims].
            value: Temporal BEV features [B*num_bev_queue, num_queries, embed_dims], the previous BEV
                and the encoder's input query stacked per sample. If None, the query stacked with itself.
            identity: Identity connection input.
            query_pos: Query positional encoding.
            key_padding_mask: Key padding mask.
            reference_points: [B*num_bev_queue, num_queries, 1, 2] in [0, 1], float32 ROW_MAJOR,
                the previous BEV's ego-shifted, the current query's as they are.
            prev_bev: Unused; the previous BEV comes in ``value``.
            **kwargs: Additional arguments.

        Returns:
            Output features [B, num_queries, embed_dims].
        """
        if use_signpost:
            signpost(header="TTNN TSA Forward Start")

        bs, num_queries, embed_dims = query.shape
        queue = self.num_bev_queue
        if value is None:
            value = ttnn.reshape(
                ttnn.concat([ttnn.unsqueeze(query, 1), ttnn.unsqueeze(query, 1)], dim=1),
                (bs * queue, num_queries, embed_dims),
            )

        # Handle defaults
        if identity is None:
            identity = query

        # Add query positional encoding
        if query_pos is not None:
            query = ttnn.add(query, query_pos)

        # value[:bs], as upstream: the first bs stacked maps, which is each sample's previous BEV
        # only at bs=1 (BEVFormer's inference batch).
        query = ttnn.concat([value[:bs], query], dim=-1)
        value = ttnn.linear(value, self.params.value_proj.weight, bias=self.params.value_proj.bias)

        # Queue-major channels to the batch, in ROW_MAJOR, where the queue axis never lands in a
        # tiled dimension: (bs, nq, queue * heads * width) -> (bs * queue, nq, heads, width).
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
        attention_weights = ttnn.softmax(
            ttnn.reshape(attention_weights, (bs, num_queries, queue * self.num_heads, self.num_points)), dim=-1
        )
        attention_weights = ttnn.reshape(attention_weights, (bs, num_queries, queue * self.num_heads * self.num_points))

        if ENABLE_LOGGING:
            logger.info("TSA Calling Deformable Attention")
        output = multi_scale_deformable_attn_fused(
            value,
            reference_points,
            split_queue(sampling_offsets),
            split_queue(attention_weights),
            self.spatial_shapes,
        )
        output = ttnn.reshape(output, (bs, queue, num_queries, embed_dims))
        output = ttnn.mean(ttnn.to_layout(output, ttnn.TILE_LAYOUT), dim=1)
        output = ttnn.linear(output, self.params.output_proj.weight, bias=self.params.output_proj.bias)

        if ENABLE_LOGGING:
            logger.info("TSA Adding Residual")

        # Residual connection
        output = ttnn.add(output, identity)

        if use_signpost:
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
