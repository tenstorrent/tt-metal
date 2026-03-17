# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Post-Combine Reduction Module (TTNN Implementation)

This module implements the reduction operation after MoE combine using TTNN.
It performs:
1. Local sum over topk dimension
2. Reduce-scatter across chips to get TP-sharded output

After MoE combine, each chip has sparse tensor [seq_len, topk, hidden_dim]
where only positions for local experts have valid data.

This module produces TP-sharded output [seq_len, hidden_dim / num_chips]
ready for the next layer.
"""

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
                      NOTE: TTNN adds a batch dimension, so for logical [seq, topk, hidden],
                      the actual tensor is [1, seq, topk, hidden] and topk is at dim=2.
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

    def forward(self, combine_output: ttnn.Tensor) -> ttnn.Tensor:
        """
        Reduce combine output by summing topk and reduce-scattering.

        Args:
            combine_output: Per-chip tensor of shape [seq_len, topk, hidden_dim]

        Returns:
            output: Per-chip tensor of shape [seq_len, hidden_dim / num_chips_in_axis]
        """
        # 1. Sum over topk dimension (local operation on each chip)
        # [seq_len, topk, hidden_dim] -> [seq_len, hidden_dim]
        summed = ttnn.sum(combine_output, dim=self.topk_dim)

        # 2. Reduce-scatter across chips
        # Reduces (sums) data across chips and scatters unique portions
        # [seq_len, hidden_dim] -> [seq_len, hidden_dim / num_chips]
        output = ttnn.reduce_scatter(
            summed,
            dim=-1,
            cluster_axis=self.cluster_axis,
            num_links=self.num_links,
            topology=self.topology,
        )

        return output
