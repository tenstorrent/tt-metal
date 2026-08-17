# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 post-combine reduction: fused weighted top-k sum, then the closing TP reduce-scatter.

The reduce-scatter has two jobs at once: it completes the top-4 expert sum across mesh columns (each
column owns a subset of the experts, so its post_combine_reduce output is a partial sum) and scatters
emb across those columns, handing the next stage emb/tp. The collective runs through the
caller-supplied ``reduce_scatter_fn`` (MeshConfig.reduce_scatter) so it uses the same managed CCL path
as every other M3 collective.
"""

from typing import Optional

from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtMiniMaxReduce(LightweightModule):
    """Fused weighted sum over topk + the EP-reduce/TP-scatter reduce-scatter."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        topk_dim: int = 3,
        cluster_axis: int = 1,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        reduce_scatter_fn=None,
    ):
        """
        Args:
            mesh_device: TTNN mesh device.
            topk_dim: Dimension of the topk axis. combine returns [1, 1, seq, topk, emb], so 3.
            cluster_axis: Mesh axis to reduce across (1 = TP columns for M3).
            num_links: Fabric links for the fallback collective.
            topology: Topology for the fallback collective.
            reduce_scatter_fn: `fn(tensor) -> tensor` performing "reduce over cluster_axis, scatter on
                the last dim". This is the intended path (M3 passes MeshConfig.reduce_scatter bound to
                its CclManager). None falls back to the plain `ttnn.reduce_scatter` prim, which keeps
                this module usable in standalone op tests that have no CclManager.
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.topk_dim = topk_dim
        self.cluster_axis = cluster_axis
        self.num_links = num_links
        self.topology = topology
        self.reduce_scatter_fn = reduce_scatter_fn

    def forward(
        self,
        combine_output: ttnn.Tensor,
        weights: Optional[ttnn.Tensor] = None,
        indices: Optional[ttnn.Tensor] = None,
        expert_dispatch_table: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Args:
            combine_output: ROW_MAJOR [1, dispatch_group_size, seq_len, topk, emb_dim]. Slots whose
                expert lives in another dispatch group hold whatever combine left there.
            weights: gate weights [..., topk] or [..., topk, 1]. Fused weighted sum when given.
            indices: global expert ids per (token, slot), UINT16.
            expert_dispatch_table: expert id -> chip id, INT32, sharded per dispatch group. Together
                with `indices` this is what lets the kernel SKIP the ~3 of 4 slots belonging to other
                dispatch groups, so the untouched slots are never read.

        Returns:
            [seq_len, emb_dim / num_chips_in_cluster_axis]
        """
        if weights is not None:
            # [..., topk] -> [..., topk, 1] so the multiply broadcasts along emb.
            if weights.shape[-1] != 1:
                weights = ttnn.unsqueeze(weights, dim=-1)
            while len(weights.shape) < len(combine_output.shape):
                weights = ttnn.unsqueeze(weights, dim=0)

            # Fused multiply + reduce over topk, skipping non-local experts. ROW_MAJOR in, TILE out.
            summed = ttnn.experimental.deepseek_prefill.post_combine_reduce(
                combine_output,
                weights,
                indices,
                expert_dispatch_table,
                expert_dim=self.topk_dim,
                output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            # No dispatch-table skip without weights, so this reads the un-written slots too. Test-only.
            logger.warning("TtMiniMaxReduce: weights not provided, using unweighted sum")
            summed = ttnn.sum(combine_output, dim=self.topk_dim)

        if self.mesh_device.shape[self.cluster_axis] <= 1:
            return summed

        if self.reduce_scatter_fn is not None:
            return self.reduce_scatter_fn(summed)
        return ttnn.reduce_scatter(
            summed,
            dim=-1,
            cluster_axis=self.cluster_axis,
            num_links=self.num_links,
            topology=self.topology,
        )
