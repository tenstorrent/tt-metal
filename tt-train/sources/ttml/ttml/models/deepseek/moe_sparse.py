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
from .autograd_ops import (
    autograd_concat,
    autograd_slice,
    moe_routing_normalize,
)
from ttml.common.profiler_utils import profiler_marker_start, profiler_marker_end


# ---------------------------------------------------------------------------
# Sparse-MoE-local autograd helpers. _ToLayout / _Transpose are general-purpose
# wrappers we don't want to add to the routing module for sparse-only needs.
# The gather is expressed below using only existing autograd primitives
# (autograd_slice, ttml.ops.binary.mul / add, autograd_concat) so the routing
# module's gradient surface stays unchanged.
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


class _Transpose(ttml.autograd.Function):
    """ttnn.transpose with same-axes backward."""

    @staticmethod
    def forward(ctx, input, dim0, dim1):
        ctx.dim0 = dim0
        ctx.dim1 = dim1
        return ttnn.transpose(input.get_value(), dim0, dim1)

    @staticmethod
    def backward(ctx, grad_output):
        return ttnn.transpose(grad_output, ctx.dim0, ctx.dim1)


def _to_layout(tensor, target_layout):
    return _ToLayout.apply(tensor, target_layout)


def _transpose(tensor, dim0, dim1):
    return _Transpose.apply(tensor, dim0, dim1)


def _gather_topk(routing_weights, topk_indices_u32, num_experts):
    """Per-row gather along the last dim, built from existing autograd
    primitives so we don't add a new Function to autograd_ops.py.

    Computes: ``out[..., k] = routing_weights[..., topk_indices[..., k]]``
    as ``out[..., k] = Σ_e 1[e == topk_indices[..., k]] · routing_weights[..., e]``,
    using ``autograd_slice`` to extract per-expert weights and
    ``ttml.ops.binary.mul/add`` to combine them. Gradients flow back to
    ``routing_weights`` through the slice + mul + add chain — same path
    the dense MoE already exercises.
    """
    device = ttml.autograd.AutoContext.get_instance().get_device()
    K = list(topk_indices_u32.shape)[-1]
    rw_shape = list(routing_weights.get_value().shape)
    base_idx_shape = list(topk_indices_u32.shape)

    # Per-expert autograd routing-weight slices, [B, 1, S, 1] each.
    rw_per_expert = []
    for e in range(num_experts):
        end = list(rw_shape)
        end[-1] = e + 1
        rw_e = autograd_slice(routing_weights, [0, 0, 0, e], end)
        rw_per_expert.append(rw_e)

    # arange for building one-hot masks (UINT32 → eq → fp32 → bf16, since
    # ttnn.eq doesn't accept uint16 and direct uint→bf16 typecast is broken).
    arange_E_u32 = ttnn.arange(0, num_experts, 1, dtype=ttnn.uint32, device=device)
    arange_E_u32 = ttnn.reshape(arange_E_u32, [1, 1, 1, num_experts])

    out_per_k = []
    for k in range(K):
        start = [0] * len(base_idx_shape)
        end = list(base_idx_shape)
        start[-1] = k
        end[-1] = k + 1
        idx_k = ttnn.slice(topk_indices_u32, start, end)  # [B, 1, S, 1] uint32

        accumulator = None
        for e in range(num_experts):
            arange_e = ttnn.slice(arange_E_u32, [0, 0, 0, e], [1, 1, 1, e + 1])  # [1,1,1,1]
            mask = ttnn.eq(idx_k, arange_e)  # [B, 1, S, 1] uint32
            mask = ttnn.typecast(mask, ttnn.float32)
            mask = ttnn.typecast(mask, ttnn.bfloat16)
            mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)
            mask_at = ttml.autograd.Tensor(mask, False)
            contribution = ttml.ops.binary.mul(rw_per_expert[e], mask_at)
            accumulator = contribution if accumulator is None else ttml.ops.binary.add(accumulator, contribution)
        out_per_k.append(accumulator)

    return autograd_concat(out_per_k, dim=-1)  # [B, 1, S, K]


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

        # Per-expert masks for sigmoid-path normalization and the on-device
        # token-count accumulator (same workaround the dense MoE uses for
        # ttnn.eq's uint16 quirk).
        topk_indices_u32 = ttnn.typecast(topk_indices, ttnn.DataType.UINT32)
        mask_parts = []
        expert_count_scalars = []
        for expert_idx in range(E):
            match = ttnn.eq(topk_indices_u32, float(expert_idx))
            match_f = ttnn.typecast(match, ttnn.DataType.BFLOAT16)
            match_any = ttnn.sum(match_f, dim=-1, keepdim=True)  # [B,1,S,1]
            mask_narrow_bf = ttnn.typecast(ttnn.gt(match_any, 0.0), ttnn.DataType.BFLOAT16)
            mask_parts.append(mask_narrow_bf)
            expert_count_scalars.append(ttnn.reshape(ttnn.sum(mask_narrow_bf), [1, 1, 1, 1]))

        # Routing weights, autograd.
        if self.score_func == "sigmoid":
            full_mask = ttnn.concat(mask_parts, dim=-1)  # [B, 1, S, num_experts]
            routing_weights = moe_routing_normalize(scores, full_mask, self.route_scale, 1e-20)
        else:
            # Softmax: routing weight at top-K position is just scores * route_scale.
            if self.route_scale != 1.0:
                routing_weights = ttml.ops.binary.mul(scores, self.route_scale)
            else:
                routing_weights = scores

        # Gather to [B, 1, S, K] (autograd-aware).
        scores_for_routing = _gather_topk(routing_weights, topk_indices_u32, E)

        # Token-count accumulator (same as dense MoE).
        batch_counts = ttnn.concat(expert_count_scalars, dim=-1)  # [1, 1, 1, num_experts]
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
        # moe_ffn_swiglu_fw consumes parallel lists of per-expert weights.
        # LinearLayer stores weights as [1, 1, out, in] (matmul does
        # x @ W^T); moe_ffn expects [1, 1, in, out] (matmul does x @ W).
        # Transpose at the boundary, autograd-aware.
        w_gate = [_transpose(self.experts[e].w1.weight.tensor, -2, -1) for e in range(E)]
        w_up = [_transpose(self.experts[e].w3.weight.tensor, -2, -1) for e in range(E)]
        w_down = [_transpose(self.experts[e].w2.weight.tensor, -2, -1) for e in range(E)]

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
