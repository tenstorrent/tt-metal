# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Autograd-aware ops for DeepSeek model.

ttnn.slice / ttnn.concat / ttnn.sigmoid have no autograd backward in ttml.
These wrappers fix that using ttml.autograd.Function.
"""

from __future__ import annotations

import ttnn
import ttml
from ttml.common.autograd_ops import Concat, Split


class Slice(ttml.autograd.Function):
    """Autograd-aware slice (ttnn.slice has no backward).

    Forward: extract a sub-tensor from input.
    Backward: place gradient into zeros at the sliced position.
    """

    @staticmethod
    def forward(ctx, input, start, end):
        ctx.input_shape = list(input.get_value().shape)
        ctx.start = start
        ctx.end = end
        return ttnn.slice(input.get_value(), start, end)

    @staticmethod
    def backward(ctx, grad_output):
        input_shape = ctx.input_shape
        start = ctx.start
        end = ctx.end
        device = grad_output.device()

        result = grad_output
        for dim in range(len(input_shape)):
            before = start[dim]
            after = input_shape[dim] - end[dim]
            if before == 0 and after == 0:
                continue

            cur_shape = list(result.shape)
            parts = []

            if before > 0:
                z_shape = list(cur_shape)
                z_shape[dim] = before
                z_before = ttnn.zeros(z_shape, dtype=result.dtype, layout=result.layout, device=device)
                parts.append(z_before)

            parts.append(result)

            if after > 0:
                z_shape = list(cur_shape)
                z_shape[dim] = after
                z_after = ttnn.zeros(z_shape, dtype=result.dtype, layout=result.layout, device=device)
                parts.append(z_after)

            result = ttnn.concat(parts, dim)

        return result


class Sigmoid(ttml.autograd.Function):
    """Autograd-aware sigmoid.

    Forward: y = sigmoid(x)
    Backward: dL/dx = dL/dy * y * (1 - y)
    """

    @staticmethod
    def forward(ctx, input):
        y = ttnn.sigmoid(input.get_value())
        ctx.y = y
        return y

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.y
        ones = ttnn.ones_like(y)
        grad_input = ttnn.multiply(grad_output, ttnn.multiply(y, ttnn.subtract(ones, y)))
        return grad_input


class Softmax(ttml.autograd.Function):
    """Autograd-aware softmax on last dim.

    Forward: y = softmax(x, dim=-1)
    Backward: uses ttnn.moreh_softmax_backward (fused kernel)
    """

    @staticmethod
    def forward(ctx, input):
        y = ttnn.softmax(input.get_value(), dim=-1)
        ctx.y = y
        return y

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.y
        grad_input = ttnn.moreh_softmax_backward(y, grad_output, 3)
        return grad_input


class SplitHeads(ttml.autograd.Function):
    """Reshape (B, 1, S, H*D) -> (B, H, S, D) with correct data layout.

    A plain reshape would corrupt the data ordering. This does:
      Forward:  reshape to (B, S, H, D) then transpose dims 1,2
      Backward: transpose dims 1,2 then reshape back
    """

    @staticmethod
    def forward(ctx, input, num_heads):
        val = input.get_value()
        B, _, S, HD = list(val.shape)
        head_dim = HD // num_heads
        ctx.num_heads = num_heads
        ctx.B = B
        ctx.S = S
        ctx.head_dim = head_dim

        # (B, 1, S, H*D) -> (B, S, H, D) -> (B, H, S, D)
        reshaped = ttnn.reshape(val, [B, S, num_heads, head_dim])
        transposed = ttnn.transpose(reshaped, 1, 2)
        return transposed

    @staticmethod
    def backward(ctx, grad_output):
        B = ctx.B
        S = ctx.S
        num_heads = ctx.num_heads
        head_dim = ctx.head_dim

        # (B, H, S, D) -> (B, S, H, D) -> (B, 1, S, H*D)
        transposed = ttnn.transpose(grad_output, 1, 2)
        reshaped = ttnn.reshape(transposed, [B, 1, S, num_heads * head_dim])
        return reshaped


class MoERoutingNormalize(ttml.autograd.Function):
    """Autograd-aware routing weight normalization for sigmoid-gated MoE.

    Forward computes, per token:

        weights[i] = mask[i] * route_scale * scores[i] / (sum_j scores[j] * mask[j] + eps)

    This mirrors DeepSeek's ``weights = scores.gather(indices); weights /=
    weights.sum(...); weights *= route_scale`` except we represent the
    selection as a dense 0/1 ``mask`` over all experts (ttml's MoE uses a
    dense-masking execution strategy).

    The key reason this op exists is gradient correctness: letting autograd
    flow through a detached ``ttnn.reciprocal(denom)`` loses the
    cross-expert terms of ∂w_i/∂s_k (where ``i ≠ k`` and both are selected),
    which shows up as cos_sim ≪ 1 on ``gate.weight`` gradients. Here we
    return the closed-form Jacobian directly.

    Backward: for upstream gradient ``G``,
        D = Σ_j s_j * m_j + eps                          (per token)
        T = Σ_i G_i * s_i * m_i / D                      (per token, scalar)
        dL/ds_k = m_k * route_scale * (G_k - T) / D

    Inputs:
      scores: autograd tensor [..., n_experts] (sigmoid scores, bf16)
      mask: raw ttnn tensor [..., n_experts] (bf16 0/1, no grad)
      route_scale: float
      eps: float
    Output: autograd tensor [..., n_experts] (the normalized routing weights)
    """

    @staticmethod
    def forward(ctx, scores, mask, route_scale, eps):
        scores_val = scores.get_value()
        scaled = ttnn.multiply(scores_val, mask)  # [..., n]
        denom = ttnn.sum(scaled, dim=-1, keepdim=True)  # [..., 1]
        denom = ttnn.add(denom, eps)
        inv_denom = ttnn.reciprocal(denom)  # [..., 1]
        scaled_inv = ttnn.multiply(inv_denom, route_scale)
        weights = ttnn.multiply(scaled, scaled_inv)  # [..., n]

        ctx.scores_val = scores_val
        ctx.mask = mask
        ctx.inv_denom = inv_denom
        ctx.route_scale = route_scale
        return weights

    @staticmethod
    def backward(ctx, grad_output):
        scores_val = ctx.scores_val
        mask = ctx.mask
        inv_denom = ctx.inv_denom
        route_scale = ctx.route_scale

        # T = (Σ_i G_i * s_i * m_i) / D
        g_s = ttnn.multiply(grad_output, scores_val)
        g_s_m = ttnn.multiply(g_s, mask)
        t = ttnn.sum(g_s_m, dim=-1, keepdim=True)  # [..., 1]
        t = ttnn.multiply(t, inv_denom)  # [..., 1]

        # dL/ds_k = mask_k * route_scale * (G_k - T) / D
        g_minus_t = ttnn.subtract(grad_output, t)  # broadcast T
        grad_scores = ttnn.multiply(g_minus_t, inv_denom)
        grad_scores = ttnn.multiply(grad_scores, mask)
        grad_scores = ttnn.multiply(grad_scores, route_scale)

        return grad_scores


class ToLayout(ttml.autograd.Function):
    """ttnn.to_layout with the inverse layout-convert as backward.

    ttnn.to_layout has no autograd backward; wrap it so the gradient is
    converted back to the input's original layout.
    """

    @staticmethod
    def forward(ctx, input, target_layout):
        ctx.source_layout = input.get_value().layout
        return ttnn.to_layout(input.get_value(), target_layout)

    @staticmethod
    def backward(ctx, grad_output):
        return ttnn.to_layout(grad_output, ctx.source_layout)


def autograd_slice(tensor, start, end):
    """Slice with autograd backward."""
    return Slice.apply(tensor, start, end)


def autograd_concat(tensors, dim):
    """Concat with autograd backward."""
    return Concat.apply(dim, *tensors)


def autograd_split(tensor, sizes, dim):
    """Split a tensor into contiguous chunks of ``sizes`` along ``dim``.

    Cheaper backward than repeated :func:`autograd_slice` (single concat).
    """
    return Split.apply(tensor, sizes, dim)


def autograd_sigmoid(tensor):
    """Sigmoid with autograd backward."""
    return Sigmoid.apply(tensor)


def autograd_softmax(tensor):
    """Softmax (last dim) with autograd backward."""
    return Softmax.apply(tensor)


def split_heads(tensor, num_heads):
    """(B, 1, S, H*D) -> (B, H, S, D) with proper transpose."""
    return SplitHeads.apply(tensor, num_heads)


def moe_routing_normalize(scores, mask, route_scale, eps=1e-20):
    """Masked + renormalized routing weights with full autograd support.

    See :class:`MoERoutingNormalize` for math and motivation.
    """
    return MoERoutingNormalize.apply(scores, mask, route_scale, eps)


def to_layout(tensor, target_layout):
    """ttnn.to_layout with autograd backward (see :class:`ToLayout`)."""
    return ToLayout.apply(tensor, target_layout)
