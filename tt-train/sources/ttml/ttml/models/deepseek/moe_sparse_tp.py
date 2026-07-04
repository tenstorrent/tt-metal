# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V3 Mixture of Experts — TP-sharded variant of SparseMoE.

Mirrors `SparseMoE`'s route → group → moe_ffn_swiglu_fw → ungroup pipeline,
but every chip in the TP cluster holds all routed experts with each
expert's intermediate dim (I) sharded across the cluster axis. Per chip:

    1. all_gather tokens across cluster_axis     → full [B, 1, S, H]
    2. moe_group                                 → grouped [T_cap, H]
    3. moe_ffn_swiglu_fw with TP-sharded LinearLayer-layout weights:
         w_gate, w_up: [I/D, H]  (column-parallel, output dim sharded)
         w_down:       [H, I/D]  (row-parallel,    input dim sharded)
       → partial output [T_cap, H] per chip
    4. moe_ungroup                               → partial [B, 1, S, H]
    5. all_reduce across cluster_axis            → full [B, 1, S, H]

Routed expert weights are created in `MoE.__init__(..., expert_tp_axis_name=...)`
via `ttml.init.uniform(..., mapper=...)` so there is no dense `Expert` linear
copy and no device→host→device seeding path.

See moe_forward_roofline.md ("Alternative approach: TP-sharded experts")
for the AI/ridge analysis vs the EP path.
"""

from __future__ import annotations

import torch

import ttnn
import ttml

from .moe import MoE
from .moe_sparse import _to_layout


class SparseMoETP(MoE):
    """TP-sharded sparse MoE.

    All experts live on every chip; each expert's intermediate dim is
    sharded across the configured `tp_axis_name` axis. Forward gathers
    the full token set, runs the local 1/D-sharded FFN on every expert,
    and all-reduces the partial outputs across the TP group.
    """

    memory_marker_prefix = "SPARSE_MOE_TP"

    def __init__(self, config) -> None:
        if getattr(config, "use_tp", False):
            axis_name = "tp"
        else:
            axis_name = getattr(config, "moe_axis_name", None) or "tp"
        super().__init__(config, expert_tp_axis_name=axis_name)
        self.axis_name = axis_name
        self.cluster_axis = ttml.mesh().axis_index(self.axis_name)

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        device = ttml.autograd.AutoContext.get_instance().get_device()
        K = self.n_activated
        E = self.num_experts

        # Input is replicated across the TP axis (this is the convention
        # in this codebase — preceding TP modules end with all_reduce).
        # No gather needed.
        B, _, S, _dim = list(x.get_value().shape)
        x = self._memory_snapshot(x, "START")

        # ── 2. routing (same as SparseMoE / dense MoE) ──
        scores, _topk_values, topk_indices = self.compute_routing(x)
        _expert_mask_all, _routing_weights, scores_for_routing, _topk_indices_u32 = self.prepare_routing_weights(
            scores, topk_indices, gather_topk=True
        )
        scores_for_routing = self._memory_snapshot(scores_for_routing, "AFTER_ROUTING")

        # ── 3. moe_group ──
        metadata = ttnn.to_layout(ttnn.typecast(topk_indices, ttnn.DataType.UINT16), ttnn.ROW_MAJOR_LAYOUT)
        leids = ttnn.from_torch(
            torch.arange(E, dtype=torch.int32),
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        x_rm = _to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        scores_for_routing_rm = _to_layout(scores_for_routing, ttnn.ROW_MAJOR_LAYOUT)
        group_out = ttml.ops.moe.moe_group_op(x_rm, metadata, scores_for_routing_rm, leids, int(E), int(K))
        grouped = group_out.grouped
        grouped = self._memory_snapshot(grouped, "AFTER_GROUP")

        # ── 4. moe_ffn_swiglu_fw with TP-sharded weights ──
        # Each chip holds 1/D of every expert's intermediate axis. Local
        # matmul produces a partial [T_cap, H] that is summed across the
        # TP group by the all_reduce below.
        # broadcast(grouped) is identity in forward (grouped is already
        # replicated) but autograd-aware: its backward all_reduce-sums
        # the per-chip partial d(grouped) coming out of moe_ffn into the
        # full gradient. Without this the partials are silently dropped
        # before they reach moe_group's backward (i.e. the routing path).
        grouped = ttml.ops.distributed.broadcast(grouped, self.cluster_axis)
        grouped = self._memory_snapshot(grouped, "AFTER_BROADCAST")
        w_gate = [self.w_gate[e].tensor for e in range(E)]
        w_up = [self.w_up[e].tensor for e in range(E)]
        w_down = [self.w_down[e].tensor for e in range(E)]
        y_grouped = ttml.ops.moe.moe_ffn_swiglu_fw(grouped, group_out.offsets, w_gate, w_up, w_down)
        y_grouped = self._memory_snapshot(y_grouped, "AFTER_FFN")

        # ── 5. moe_ungroup ──
        output_rm = ttml.ops.moe.moe_ungroup_op(
            y_grouped,
            group_out.grouped_scores,
            metadata,
            leids,
            group_out.plan,
            group_out.offsets,
            int(E),
            int(K),
            int(B),  # D in the wrapper's signature
            1,
            int(S),
        )
        output = _to_layout(output_rm, ttnn.TILE_LAYOUT)
        output = self._memory_snapshot(output, "AFTER_UNGROUP")

        # ── 6. all_reduce across TP axis to combine partial sums ──
        # noop_backward=True: each chip's partial contributes equally to
        # the replicated output, so dL/d(partial_chip) = dL/d(output) flows
        # back identically to every chip — no second collective in backward
        # (matches RowParallelLinear's input_is_parallel=True case).
        output = ttml.ops.distributed.all_reduce(output, noop_backward=True, cluster_axis=self.cluster_axis)
        output = self._memory_snapshot(output, "AFTER_ALL_REDUCE")

        # Shared experts: their LinearLayer modules are NOT TP-sharded by
        # default; they replicate work and don't need the all_reduce. If a
        # downstream change makes them TP, swap to ColumnParallel/RowParallel.
        if self.shared_experts is not None:
            x_shared = self._memory_snapshot(x, "BEFORE_SHARED")
            shared_out = self.shared_experts(x_shared)
            shared_out = self._memory_snapshot(shared_out, "AFTER_SHARED")
            pre_add_output = output
            output = ttml.ops.binary.add(pre_add_output, shared_out)
            # binary.add backward is d_a = d_b = d_out (no read of inputs);
            # all_reduce above has noop_backward (identity, no read of its fwd
            # output); shared_out's parent linear (w2) reads w2's input for its
            # weight grad, not shared_out. Safe to release these values now.
            pre_add_output.get_value().deallocate(force=True)
            shared_out.get_value().deallocate(force=True)

        output = self._memory_snapshot(output, "END")
        return output
