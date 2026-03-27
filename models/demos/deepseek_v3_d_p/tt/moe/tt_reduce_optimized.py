# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Optimized Post-Combine Reduction Module (TTNN Implementation)

This module implements the OPTIMIZED reduction operation after MoE combine using
a single fused kernel instead of the naive sequence of operations.

BEFORE (naive implementation):
1. ttnn.to_layout() - ROW_MAJOR → TILE_LAYOUT with fillpad (8→32 experts)
2. ttnn.mul() - broadcast weights across embedding dimension
3. ttnn.sum() - reduce over expert dimension
4. ttnn.reduce_scatter() - distributed reduction

AFTER (fused kernel):
1. ttnn.experimental.deepseek_moe_post_combine_reduce() - all-in-one
2. ttnn.reduce_scatter() - distributed reduction

Performance benefits:
- Eliminates 300% padding overhead (8→32 experts)
- Reduces memory bandwidth by 4x
- Eliminates intermediate tensor allocations
- Single kernel launch vs multiple operations
"""

from typing import Optional

from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtReduceModuleOptimized(LightweightModule):
    """Optimized TTNN implementation using fused kernel."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        expert_dim: int = 3,
        cluster_axis: int = 1,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
    ):
        """
        Initialize optimized reduce module.

        Args:
            mesh_device: TTNN mesh device
            expert_dim: Dimension of the expert axis in tensor (default: 3)
            cluster_axis: Mesh dimension to reduce across (0=rows, 1=columns)
            num_links: Number of ethernet links to use for collective
            topology: Ring or Linear topology for reduce_scatter
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.expert_dim = expert_dim
        self.cluster_axis = cluster_axis
        self.num_links = num_links
        self.topology = topology

    def forward(
        self,
        combine_output: ttnn.Tensor,
        weights: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        OPTIMIZED reduce using fused kernel.

        Args:
            combine_output: Per-chip tensor [1, 1, seq_len, num_experts, emb_dim] (ROW_MAJOR)
            weights: Gate weights [1, 1, seq_len, num_experts] (ROW_MAJOR or TILE_LAYOUT)

        Returns:
            output: Per-chip tensor [seq_len, emb_dim / num_chips_in_axis] (TILE_LAYOUT)
        """
        if weights is None:
            raise ValueError("TtReduceModuleOptimized requires weights - cannot do unweighted sum")

        logger.debug(f"[TtReduceOptimized] Input shapes:")
        logger.debug(f"  combine_output: {combine_output.shape} {combine_output.layout}")
        logger.debug(f"  weights: {weights.shape} {weights.layout}")

        # FUSED OPERATION: Replaces ttnn.to_layout + ttnn.mul + ttnn.sum
        # - Reads ROW_MAJOR combine_output directly (no fillpad!)
        # - Performs broadcast multiply + reduce in single kernel
        # - Outputs TILE_LAYOUT result ready for reduce_scatter
        reduced_output = ttnn.experimental.deepseek_moe_post_combine_reduce(
            combine_output,  # ROW_MAJOR [1, 1, seq_len, 8, emb_dim]
            weights,         # Gate weights [1, 1, seq_len, 8]
            expert_dim=self.expert_dim,  # Reduce over expert dimension (3)
            output_memory_config=ttnn.L1_MEMORY_CONFIG
        )
        # Output: TILE_LAYOUT [1, 1, seq_len, emb_dim]

        logger.debug(f"[TtReduceOptimized] After fused kernel: {reduced_output.shape}")

        # Reduce-scatter across chips (only if multiple devices in cluster_axis)
        # [seq_len, emb_dim] → [seq_len, emb_dim / num_chips]
        if self.mesh_device.shape[self.cluster_axis] > 1:
            output = ttnn.reduce_scatter(
                reduced_output,
                dim=-1,  # Scatter along embedding dimension
                cluster_axis=self.cluster_axis,
                num_links=self.num_links,
                topology=self.topology,
            )
        else:
            output = reduced_output  # No reduce-scatter needed if only 1 device in axis

        logger.debug(f"[TtReduceOptimized] Final output: {output.shape}")
        return output