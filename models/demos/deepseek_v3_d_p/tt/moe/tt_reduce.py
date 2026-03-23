# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Post-Combine Reduction Module (TTNN Implementation)

This module implements the reduction operation after MoE combine using TTNN.
It performs:
1. Optional weight multiplication (for weighted MoE sum)
2. Local sum over topk dimension
3. Reduce-scatter across chips to get TP-sharded output

After MoE combine, each chip has sparse tensor [seq_len, topk, emb_dim]
where only positions for local experts have valid data.

This module produces TP-sharded output [seq_len, emb_dim / num_chips]
ready for the next layer.
"""

from typing import Optional

from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtReduceModule(LightweightModule):
    """TTNN implementation: sum over topk + reduce_scatter."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        topk_dim: int = 2,
        cluster_axis: int = 1,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
    ):
        """
        Initialize reduce module.

        Args:
            mesh_device: TTNN mesh device
            topk_dim: Dimension of the topk axis in TTNN tensor.
                      NOTE: TTNN adds a batch dimension, so for logical [seq, topk, emb_dim],
                      the actual tensor is [1, seq, topk, emb_dim] and topk is at dim=2.
            cluster_axis: Mesh dimension to reduce across (0=rows, 1=columns)
            num_links: Number of ethernet links to use for collective
            topology: Ring or Linear topology for reduce_scatter
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.topk_dim = topk_dim
        self.cluster_axis = cluster_axis
        self.num_links = num_links
        self.topology = topology

    def forward(
        self,
        combine_output: ttnn.Tensor,
        weights: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Reduce combine output by summing topk and reduce-scattering.

        Args:
            combine_output: Per-chip tensor of shape [1, 1, seq_len, topk, emb_dim]
            weights: Optional gate weights of shape [1, 1, seq_len, topk]
                     If provided, applies weighted sum: weights * combine_output

        Returns:
            output: Per-chip tensor of shape [seq_len, emb_dim / num_chips_in_axis]
        """
        # Apply weights if provided (broadcast [seq_len, topk] -> [seq_len, topk, emb_dim])
        if weights is not None:
            # Prepare weights: ensure shape and layout match combine_output
            # combine_output is 5D: (1, dispatch_group_size, seq_len, topk, emb_dim)
            # weights may be 3D or 4D, need to match first 4 dims of combine_output

            # Add batch dimensions if needed to match combine_output rank - 1
            # (we'll add the final dim via unsqueeze for broadcasting)
            target_rank = len(combine_output.shape) - 1  # 4D for weights
            while len(weights.shape) < target_rank:
                weights = ttnn.unsqueeze(weights, dim=0)

            # Convert to TILE_LAYOUT if not already
            if weights.layout != ttnn.TILE_LAYOUT:
                weights = ttnn.to_layout(weights, ttnn.TILE_LAYOUT)

            # Unsqueeze weights to [1, 1, seq_len, topk, 1] for broadcasting
            weights_expanded = ttnn.unsqueeze(weights, dim=-1)
            combine_output = ttnn.mul(combine_output, weights_expanded)
        else:
            logger.warning("TtReduceModule: weights not provided, using unweighted sum")

        # 1. Sum over topk dimension (local operation on each chip)
        # [seq_len, topk, emb_dim] -> [seq_len, emb_dim]
        summed = ttnn.sum(combine_output, dim=self.topk_dim)

        # 2. Reduce-scatter across chips (only if multiple devices in cluster_axis)
        # Reduces (sums) data across chips and scatters unique portions
        # [seq_len, emb_dim] -> [seq_len, emb_dim / num_chips]
        if self.mesh_device.shape[self.cluster_axis] > 1:
            output = ttnn.reduce_scatter(
                summed,
                dim=-1,
                cluster_axis=self.cluster_axis,
                num_links=self.num_links,
                topology=self.topology,
            )
        else:
            output = summed  # No reduce-scatter needed if only 1 device in axis

        return output
