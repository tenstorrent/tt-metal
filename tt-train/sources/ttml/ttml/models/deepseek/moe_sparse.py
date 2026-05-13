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
from .autograd_ops import moe_routing_normalize
from ttml.common.profiler_utils import profiler_marker_start, profiler_marker_end


# ---------------------------------------------------------------------------
# Sparse-MoE-local autograd helpers. _ToLayout / _Transpose are general-purpose
# wrappers we don't want to add to the routing module for sparse-only needs.
# The gather below has a custom backward so sparse routing does not need to
# pad every per-expert slice with concat during backprop.
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


class _GatherTopK(ttml.autograd.Function):
    """Gather routing weights at top-k expert indices.

    Forward computes ``out[..., k] = routing_weights[..., topk_indices[..., k]]``.
    Backward scatters ``grad_output`` directly back into the full expert axis,
    avoiding per-expert slice backward concat/permute chains.
    """

    @staticmethod
    def forward(ctx, routing_weights, topk_indices_u32, num_experts):
        routing_value = routing_weights.get_value()
        rw_shape = list(routing_value.shape)
        topk_shape = list(topk_indices_u32.shape)
        B, _, S, E = rw_shape
        K = topk_shape[-1]
        ctx.topk_indices_u32 = topk_indices_u32
        ctx.rw_shape = rw_shape
        ctx.rw_layout = routing_value.layout

        # Build a 4D one-hot matrix over flattened (token, k-slot) rows:
        #   topk_flat      [B,1,S*K,1]
        #   expert_ids     [1,1,1,E]
        #   one_hot        [B,1,S*K,E]
        # Then repeat routing weights along the token axis to match K slots and
        # reduce over E. This is the gather without per-expert slices.
        topk_flat = ttnn.reshape(topk_indices_u32, [B, 1, S * K, 1])
        expert_ids = ttnn.arange(0, num_experts, 1, dtype=ttnn.uint32, device=routing_value.device())
        expert_ids = ttnn.reshape(expert_ids, [1, 1, 1, E])
        one_hot = ttnn.eq(topk_flat, expert_ids)
        one_hot = ttnn.typecast(one_hot, ttnn.float32)
        one_hot = ttnn.typecast(one_hot, ttnn.bfloat16)
        one_hot = ttnn.to_layout(one_hot, routing_value.layout)

        routing_repeated = ttnn.repeat_interleave(routing_value, K, dim=2)
        selected_flat = ttnn.multiply(routing_repeated, one_hot)
        selected_flat = ttnn.sum(selected_flat, dim=-1, keepdim=True)
        return ttnn.reshape(selected_flat, [B, 1, S, K])

    @staticmethod
    def backward(ctx, grad_output):
        # topk indices are unique for a token, so scatter is equivalent to
        # scatter-add here and avoids rebuilding every e:e+1 slice gradient.
        device = grad_output.device()
        topk_indices_rm = ttnn.to_layout(ctx.topk_indices_u32, ttnn.ROW_MAJOR_LAYOUT)
        grad_rm = ttnn.to_layout(grad_output, ttnn.ROW_MAJOR_LAYOUT)
        grad_routing = ttnn.zeros(ctx.rw_shape, grad_output.dtype, ttnn.ROW_MAJOR_LAYOUT, device)
        grad_routing = ttnn.scatter(grad_routing, -1, topk_indices_rm, grad_rm)
        return ttnn.to_layout(grad_routing, ctx.rw_layout)


def _gather_topk(routing_weights, topk_indices_u32, num_experts):
    return _GatherTopK.apply(routing_weights, topk_indices_u32, num_experts)


class SparseMoE(MoE):
    """Sparse MoE: same routing as dense MoE, sparse expert dispatch.

    Reuses ``MoE.compute_routing`` and ``MoE.experts`` so the same weights
    can be loaded and the same autograd graph builds for the routing path.
    Only the expert dispatch (the for-loop over experts in dense's forward)
    is replaced with the route → group → moe_ffn_swiglu_fw → ungroup
    pipeline.
    """

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        device = ttml.autograd.AutoContext.get_instance().get_device()
        B, _, S, dim = list(x.get_value().shape)
        K = self.n_activated
        E = self.num_experts

        x = profiler_marker_start(x, "MoE")
        x = profiler_marker_start(x, "MoE.routing")

        scores, _topk_values, topk_indices = self.compute_routing(x)

        # Per-expert mask for sigmoid-path normalization and token counts.
        # Same scatter construction as dense MoE: one op instead of per-expert
        # eq/sum/gt/typecast chains.
        topk_indices_u32 = ttnn.typecast(topk_indices, ttnn.DataType.UINT32)
        topk_indices_rm = ttnn.to_layout(topk_indices, ttnn.ROW_MAJOR_LAYOUT)
        expert_mask_all = ttnn.zeros([B, 1, S, E], ttnn.DataType.BFLOAT16, ttnn.ROW_MAJOR_LAYOUT, device)
        expert_src = ttnn.ones([B, 1, S, K], ttnn.DataType.BFLOAT16, ttnn.ROW_MAJOR_LAYOUT, device)
        expert_mask_all = ttnn.scatter(expert_mask_all, -1, topk_indices_rm, expert_src)
        expert_mask_all = ttnn.to_layout(expert_mask_all, ttnn.TILE_LAYOUT)

        # Routing weights, autograd.
        if self.score_func == "sigmoid":
            routing_weights = moe_routing_normalize(scores, expert_mask_all, self.route_scale, 1e-20)
        else:
            # Softmax: routing weight at top-K position is just scores * route_scale.
            if self.route_scale != 1.0:
                routing_weights = ttml.ops.binary.mul(scores, self.route_scale)
            else:
                routing_weights = scores

        # Gather to [B, 1, S, K] (autograd-aware).
        scores_for_routing = _gather_topk(routing_weights, topk_indices_u32, E)

        # Token-count accumulator (same as dense MoE).
        mask_bs_flat = ttnn.reshape(expert_mask_all, [1, 1, B * S, E])
        batch_counts = ttnn.sum(mask_bs_flat, dim=-2, keepdim=True)  # [1, 1, 1, num_experts]
        new_counts = ttnn.add(self._token_counts.tensor.get_value(), batch_counts)
        self._token_counts.tensor.set_value(new_counts)

        scores_for_routing = profiler_marker_end(scores_for_routing, "MoE.routing")

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

        # Shared experts.
        if self.shared_experts is not None:
            output = ttml.ops.binary.add(output, self.shared_experts(x))

        output = profiler_marker_end(output, "MoE")
        return output
