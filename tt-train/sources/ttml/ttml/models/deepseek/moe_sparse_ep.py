# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V3 Mixture of Experts — Expert-Parallel sparse variant.

Each chip in the EP cluster stores and runs ``E / D_ep`` experts. Per chip:

    1. routing on replicated input → topk_indices [B, 1, S, K]
       (identical on every chip — gate weights are replicated)
    2. broadcast(x), broadcast(scores_for_routing) on EP axis
       — fwd: identity (inputs are already replicated)
       — bwd: all_reduce of d(x) / d(scores) across EP, because each chip's
         moe_group.bw returns a partial covering only its local experts
    3. moe_group with per-chip ``leids`` → grouped/plan/offsets contain only
       the chip's local-expert rows
    4. moe_ffn_swiglu_fw over the E_local local experts (per-chip sharded
       expert weights)
    5. moe_ungroup → partial dense output [B, 1, S, H]
    6. all_reduce across EP axis sums the partials

Compare to:

    SparseMoE     — all experts on every device, no collectives.
    SparseMoEEP   — disjoint expert subsets per device; replicated input/
                    routing/scores; per-chip divergence starts at moe_group
                    via ``leids``. Saves expert-weight memory linearly in
                    D_ep (each chip allocates 1/D_ep of the routed-expert
                    weights vs. the SparseMoE layout which replicates).
"""

from __future__ import annotations

import ttnn
import ttml

from .moe import MoE
from .autograd_ops import to_layout


class SparseMoEEP(MoE):
    """Expert-Parallel sparse MoE.

    Experts partitioned across the EP axis: shard ``r`` holds global experts
    ``[r * E_local, (r+1) * E_local)`` where ``E_local = E / D_ep``. Token
    data is replicated across the EP axis (the input arrives replicated from
    preceding modules), every chip runs the same routing, and the per-chip
    ``leids`` tensor filters moe_group's output to local experts only.
    """

    memory_marker_prefix = "SPARSE_MOE_EP"

    def __init__(self, config, *, axis_name: str | None = None) -> None:
        # The EP axis can be passed explicitly (the transformer dispatcher does
        # this so EP can reuse the existing "tp" axis without introducing a new
        # axis name). Fall back to "tp" — with full-model TP enabled, that same
        # axis carries the EP-sharded experts.
        # (DP + EP on the same axis would need an extra routing CCL, not used here.)
        if axis_name is None:
            axis_name = getattr(config, "moe_axis_name", None) or "tp"
        super().__init__(config, moe_axis_name=axis_name)
        self.axis_name = axis_name
        self.cluster_axis = ttml.mesh().axis_index(axis_name)
        # self._leids is built by MoE.__init__, sharded across this EP axis so
        # shard r holds global IDs [r*E_local, (r+1)*E_local).

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        K = self.n_activated

        # If the EP axis is also the data-parallel axis, the batch is sharded
        # across it — all_gather the full batch so every chip's local experts
        # see all tokens (re-scattered at the end). No-op otherwise: EP normally
        # sits on an axis independent of DP, where the input is already replicated.
        mesh = ttml.mesh()
        dp_gather = mesh.has_axis("dp") and mesh.axis_size("dp") > 1 and mesh.axis_index("dp") == self.cluster_axis
        if dp_gather:
            x = ttml.ops.distributed.all_gather(x, 0, self.cluster_axis, ttml.ops.distributed.GradOutputType.SHARDED)

        B, _, S, _dim = list(x.get_value().shape)
        x = self._memory_snapshot(x, "START")

        # ── routing (replicated; same on every EP chip) ──
        scores, _topk_values, topk_indices = self.compute_routing(x)
        _expert_mask_all, _routing_weights, scores_for_routing, _topk_indices_u32 = self.prepare_routing_weights(
            scores, topk_indices, gather_topk=True
        )
        scores_for_routing = self._memory_snapshot(scores_for_routing, "AFTER_ROUTING")

        # ── broadcast(x), broadcast(scores) for autograd ──
        # Forward is identity (x and scores_for_routing are already replicated
        # across EP). Backward all_reduces d(x) / d(scores) across the EP axis
        # because each chip's moe_group.bw produces a partial gradient covering
        # only the contribution from its local experts; without the broadcast
        # those partials would be dropped before reaching the routing path.
        x_bc = ttml.ops.distributed.broadcast(x, self.cluster_axis)
        scores_for_routing_bc = ttml.ops.distributed.broadcast(scores_for_routing, self.cluster_axis)

        # ── moe_group with per-shard leids ──
        # Each chip filters routing to its own E_local experts. The resulting
        # grouped/plan/offsets/grouped_scores are per-chip distinct: they
        # reference experts [0, E_local) in the chip's local indexing.
        metadata = ttnn.to_layout(ttnn.typecast(topk_indices, ttnn.DataType.UINT16), ttnn.ROW_MAJOR_LAYOUT)
        x_rm = to_layout(x_bc, ttnn.ROW_MAJOR_LAYOUT)
        scores_for_routing_rm = to_layout(scores_for_routing_bc, ttnn.ROW_MAJOR_LAYOUT)
        group_out = ttml.ops.moe.moe_group_op(
            x_rm, metadata, scores_for_routing_rm, self._leids, int(self.e_local), int(K)
        )
        grouped = self._memory_snapshot(group_out.grouped, "AFTER_GROUP")

        # ── moe_ffn_swiglu_fw over local experts ──
        # self.w_gate / w_up / w_down each have E_local Python entries. Each
        # entry is mesh-sharded so the i-th Parameter on shard r holds global
        # expert r*E_local + i — i.e. every chip sees a different set of
        # weight values under the same Python index.
        w_gate = [self.w_gate[i].tensor for i in range(self.e_local)]
        w_up = [self.w_up[i].tensor for i in range(self.e_local)]
        w_down = [self.w_down[i].tensor for i in range(self.e_local)]
        y_grouped = ttml.ops.moe.moe_ffn_swiglu_fw(grouped, group_out.offsets, w_gate, w_up, w_down)
        y_grouped = self._memory_snapshot(y_grouped, "AFTER_FFN")

        # ── moe_ungroup → per-chip partial dense output ──
        output_rm = ttml.ops.moe.moe_ungroup_op(
            y_grouped,
            group_out.grouped_scores,
            metadata,
            self._leids,
            group_out.plan,
            group_out.offsets,
            int(self.e_local),
            int(K),
            int(B),  # D in the wrapper's signature
            1,
            int(S),
        )
        output = to_layout(output_rm, ttnn.TILE_LAYOUT)
        output = self._memory_snapshot(output, "AFTER_UNGROUP")

        # ── all_reduce across EP axis to combine partial sums ──
        # noop_backward=True: each chip's partial contributes equally to the
        # replicated output, so dL/d(partial_chip) = dL/d(output) flows back
        # identically to every chip — no second collective in backward.
        output = ttml.ops.distributed.all_reduce(output, noop_backward=True, cluster_axis=self.cluster_axis)
        output = self._memory_snapshot(output, "AFTER_ALL_REDUCE")

        # Shared experts: not EP-sharded by default — they replicate work and
        # don't need an all_reduce. If a downstream change makes them EP/TP,
        # plug in the appropriate collective.
        if self.shared_experts is not None:
            x_shared = self._memory_snapshot(x, "BEFORE_SHARED")
            shared_out = self.shared_experts(x_shared)
            shared_out = self._memory_snapshot(shared_out, "AFTER_SHARED")
            pre_add_output = output
            output = ttml.ops.binary.add(pre_add_output, shared_out)
            # binary.add's backward is just d_a = d_b = d_out (no read of inputs);
            # the all_reduce above has noop_backward (identity), so its forward
            # output is never read in bwd either; shared_out's parent linear (w2)
            # also doesn't read shared_out itself for its weight grad (it reads w2's
            # input, which is saved separately). All three sources confirm these
            # two tensor values are not needed downstream — release them now.
            pre_add_output.get_value().deallocate(force=True)
            shared_out.get_value().deallocate(force=True)

        # Re-shard the (replicated, full-batch) output back onto the DP axis so
        # downstream layers stay data-parallel. Mirror of the entry all_gather.
        if dp_gather:
            output = ttml.ops.distributed.scatter(output, 0, self.cluster_axis)

        output = self._memory_snapshot(output, "END")
        return output
