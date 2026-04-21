# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Post-Combine Reduction Module (TTNN Implementation)

This module implements the reduction operation after MoE combine using TTNN.
It performs:
1. Fused weighted sum over topk dimension (multiply + reduce in a single kernel)
2. Reduce-scatter across chips to get TP-sharded output

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
    """TTNN implementation: fused weighted sum over topk + reduce_scatter."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        topk_dim: int = 2,
        cluster_axis: int = 1,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        num_dispatch_subgroups: int = 1,
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
            num_dispatch_subgroups: When > 1, the mesh row axis is partitioned into
                subgroups. The reduce-scatter's cluster_axis must be orthogonal to the
                subgroup axis (=0) so the collective stays inside a subgroup.
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.topk_dim = topk_dim
        self.cluster_axis = cluster_axis
        self.num_links = num_links
        self.topology = topology
        self.num_dispatch_subgroups = num_dispatch_subgroups

        if num_dispatch_subgroups > 1:
            assert cluster_axis != 0, (
                f"TtReduceModule cluster_axis ({cluster_axis}) must differ from the subgroup "
                f"partition axis (0) when num_dispatch_subgroups ({num_dispatch_subgroups}) > 1."
            )

    def forward(
        self,
        combine_output: ttnn.Tensor,
        weights: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Reduce combine output by summing topk and reduce-scattering.

        Args:
            combine_output: Per-chip tensor in ROW_MAJOR layout.
                Shape: [1, dispatch_group_size, seq_len, topk, emb_dim]
            weights: Optional gate weights.
                Shape: [1, dispatch_group_size, seq_len, topk] or [..., topk, 1]
                If provided, applies fused weighted sum: sum(weights * combine_output, dim=topk)

        Returns:
            output: Per-chip tensor of shape [seq_len, emb_dim / num_chips_in_axis]
        """
        if weights is not None:
            # Ensure weights has trailing dim=1 for broadcast: [..., topk] -> [..., topk, 1]
            if weights.shape[-1] != 1:
                weights = ttnn.unsqueeze(weights, dim=-1)

            # Add batch dimensions if needed to match combine_output rank
            while len(weights.shape) < len(combine_output.shape):
                weights = ttnn.unsqueeze(weights, dim=0)

            # Fused weighted sum: multiply by weights and reduce over topk in a single kernel
            # Input: ROW_MAJOR, Output: TILE_LAYOUT
            summed = ttnn.experimental.deepseek_prefill.post_combine_reduce(
                combine_output,
                weights,
                expert_dim=self.topk_dim,
                output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            logger.warning("TtReduceModule: weights not provided, using unweighted sum")
            summed = ttnn.sum(combine_output, dim=self.topk_dim)

        # Reduce-scatter across chips (only if multiple devices in cluster_axis)
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
            output = summed

        return output
