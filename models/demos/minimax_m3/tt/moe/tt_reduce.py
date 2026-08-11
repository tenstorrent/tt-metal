# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 post-combine reduction: fused weighted top-k sum, then the closing TP reduce-scatter.

M3's own copy of DeepSeek's TtReduceModule (deepseek_v3_d_p/tt/moe/tt_reduce.py), which we do not
modify. The compute half is identical — the same shared `deepseek_prefill.post_combine_reduce` kernel.
The difference is the collective:

  DeepSeek  ttnn.reduce_scatter                    -- the plain prim, with no barrier semaphore
  here      caller-supplied reduce_scatter_fn      -- M3 passes MeshConfig.reduce_scatter, i.e.
                                                      reduce_scatter_minimal_async with the ping-pong
                                                      + barrier semaphores every other M3 collective
                                                      already uses (tt/config.py)

Why bother, given both reduce over the same axis and measure the same ~145 us of work: consistency.
Every M3 collective now goes through one managed path, which matters for debugging — the shared expert's reduce-scatter and the MoE's were
different ops with different semaphore machinery, which makes a CCL hang or a nondeterministic-PCC
bisect read as an M3-vs-M3 difference when it is really prim-vs-async.

WHAT THIS OP ACTUALLY DOES — worth stating because it is easy to misread as a plain TP shard. It has
TWO jobs at once:

  1. EP all-reduce. A mesh column owns only 32 of the 128 experts, so post_combine_reduce's output is a
     PARTIAL sum over that column's experts. Summing across the 4 columns is what completes top-4.
  2. TP scatter. The same op scatters emb across those 4 columns, handing the next stage emb/tp.

Expert parallelism is folded onto the tensor-parallel axis. That is also why this collective shows the
largest barrier wait in the block (measured 145 us work + 879 us wait): unlike `combine`, which
synchronises 8 chips that share a token set, this synchronises 4 chips whose expert loads are
unrelated, so four independent whale-expert skews meet here.
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
