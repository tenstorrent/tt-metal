# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V3 Mixture of Experts (MoE) layer — sparse path.

Replaces the dense per-expert mask loop with route-and-dispatch:

    1. compute routing (same as dense MoE)
    2. gather routing weights at top-K positions  → scores [B,1,S,K]
    3. metal::moe_group → packed [1, 1, T_cap, H] grouped + per-row scores
    4. moe_ffn_swiglu_fw runs SwiGLU on each per-expert slice
    5. metal::moe_ungroup → dense [B, 1, S, H] output, fused with the
       per-row routing weight scaling

Numerically equivalent to the dense MoE when E_local == num_experts
(single-device sparse). Used to validate the autograd op wrappers
end-to-end against the dense reference.
"""

from __future__ import annotations

import torch

import ttnn
import ttml

from .moe import MoE
from ttml.common.profiler_utils import profiler_marker_start, profiler_marker_end


# ---------------------------------------------------------------------------
# Sparse-MoE-local autograd helpers. _ToLayout is a general-purpose wrapper we
# don't want to add to the routing module for sparse-only layout conversion.
# ---------------------------------------------------------------------------


class _ToLayout(ttml.autograd.Function):
    """ttnn.to_layout with the inverse layout-convert as backward."""

    @staticmethod
    def forward(ctx, input, target_layout):
        ctx.source_layout = input.get_value().layout
        return ttnn.to_layout(input.get_value(), target_layout)

    @staticmethod
    def backward(ctx, grad_output):
        return ttnn.to_layout(grad_output, ctx.source_layout)


def _to_layout(tensor, target_layout):
    return _ToLayout.apply(tensor, target_layout)


class SparseMoE(MoE):
    """Sparse MoE: same routing as dense MoE, sparse expert dispatch.

    Reuses ``MoE.compute_routing`` and ``MoE.experts`` so the same weights
    can be loaded and the same autograd graph builds for the routing path.
    Only the expert dispatch (the for-loop over experts in dense's forward)
    is replaced with the route → group → moe_ffn_swiglu_fw → ungroup
    pipeline.
    """

    memory_marker_prefix = "SPARSE_MOE"

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        device = ttml.autograd.AutoContext.get_instance().get_device()
        B, _, S, dim = list(x.get_value().shape)
        K = self.n_activated
        E = self.num_experts

        x = profiler_marker_start(x, "MoE")
        x = self._memory_snapshot(x, "START")
        x = profiler_marker_start(x, "MoE.routing")

        scores, _topk_values, topk_indices = self.compute_routing(x)

        _expert_mask_all, _routing_weights, scores_for_routing, _topk_indices_u32 = self.prepare_routing_weights(
            scores, topk_indices, gather_topk=True
        )

        scores_for_routing = profiler_marker_end(scores_for_routing, "MoE.routing")
        scores_for_routing = self._memory_snapshot(scores_for_routing, "AFTER_ROUTING")

        # ── moe_group_op ──
        # x is [B, 1, S, dim] which we treat as [D=B, B'=1, S, H=dim] for the
        # wrapper. Total tokens = B*S as expected.
        # The kernel ingests ROW_MAJOR for x / metadata / scores; convert
        # here (in multi-device this same conversion lives just before the
        # CCL boundary).
        metadata = ttnn.to_layout(ttnn.typecast(topk_indices, ttnn.DataType.UINT16), ttnn.ROW_MAJOR_LAYOUT)
        leids = ttnn.from_torch(
            torch.arange(E, dtype=torch.int32),
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        x_rm = _to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        scores_for_routing_rm = _to_layout(scores_for_routing, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = profiler_marker_start(x_rm, "MoE.group_op")
        group_out = ttml.ops.moe.moe_group_op(x_rm, metadata, scores_for_routing_rm, leids, int(E), int(K))
        # MoEGroupOutputs fields are read-only; thread the marker through a
        # local rebind of `grouped` instead of mutating the struct.
        grouped = profiler_marker_end(group_out.grouped, "MoE.group_op")
        grouped = self._memory_snapshot(grouped, "AFTER_GROUP")

        # ── Per-expert SwiGLU on grouped layout ──
        # moe_ffn_swiglu_fw consumes raw LinearLayer weights in [1, 1, out, in]
        # layout and applies transpose_b inside the variable-M matmuls.
        w_gate = [self.experts[e].w1.weight.tensor for e in range(E)]
        w_up = [self.experts[e].w3.weight.tensor for e in range(E)]
        w_down = [self.experts[e].w2.weight.tensor for e in range(E)]

        # At production shapes the kernel uses the full grid → offsets[-1]
        # equals T_cap, so we feed `grouped` straight into the FFN with no
        # slice/pad dance.
        grouped_marked = profiler_marker_start(grouped, "MoE.ffn")
        y_grouped = ttml.ops.moe.moe_ffn_swiglu_fw(grouped_marked, group_out.offsets, w_gate, w_up, w_down)
        y_grouped = profiler_marker_end(y_grouped, "MoE.ffn")
        y_grouped = self._memory_snapshot(y_grouped, "AFTER_FFN")

        # ── moe_ungroup_op ──
        # The kernel returns ROW_MAJOR; downstream layers (RMSNorm, etc.)
        # expect TILE, so re-tile here.
        y_grouped = profiler_marker_start(y_grouped, "MoE.ungroup_op")
        output_rm = ttml.ops.moe.moe_ungroup_op(
            y_grouped,
            group_out.grouped_scores,
            metadata,
            leids,
            group_out.plan,
            group_out.offsets,
            int(E),
            int(K),
            int(B),  # D
            1,  # B'
            int(S),  # S
        )
        output_rm = profiler_marker_end(output_rm, "MoE.ungroup_op")
        output = _to_layout(output_rm, ttnn.TILE_LAYOUT)
        output = self._memory_snapshot(output, "AFTER_UNGROUP")

        # Shared experts.
        if self.shared_experts is not None:
            x_shared = self._memory_snapshot(x, "BEFORE_SHARED")
            shared_out = self.shared_experts(x_shared)
            shared_out = self._memory_snapshot(shared_out, "AFTER_SHARED")
            output = ttml.ops.binary.add(output, shared_out)

        output = profiler_marker_end(output, "MoE")
        output = self._memory_snapshot(output, "END")
        return output
