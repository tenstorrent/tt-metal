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


class Slice(ttml.autograd.Function):
    """Autograd-aware slice (ttnn.slice has no backward).

    Forward: extract a sub-tensor from input.
    Backward: place gradient into zeros at the sliced position.
    """

    @staticmethod
    def forward(ctx, input, start, end):
        ctx.save_for_backward(input)
        ctx.start = start
        ctx.end = end
        return ttnn.slice(input.get_value(), start, end)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        input_shape = list(input.get_value().shape)
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


class Concat(ttml.autograd.Function):
    """Autograd-aware concat (ttnn.concat has no backward).

    Forward: concatenate tensors along a dimension.
    Backward: split gradient and distribute to each input.
    """

    @staticmethod
    def forward(ctx, dim, *tensors):
        sizes = [list(t.get_value().shape)[dim] for t in tensors]
        ctx.sizes = sizes
        ctx.dim = dim
        ctx.num_tensors = len(tensors)
        ctx.save_for_backward(*tensors)

        raw_tensors = [t.get_value() for t in tensors]
        return ttnn.concat(raw_tensors, dim)

    @staticmethod
    def backward(ctx, grad_output):
        dim = ctx.dim
        sizes = ctx.sizes
        shape = list(grad_output.shape)

        grads = []
        offset = 0
        for size in sizes:
            start = [0] * len(shape)
            end = list(shape)
            start[dim] = offset
            end[dim] = offset + size
            grad_slice = ttnn.slice(grad_output, start, end)
            grads.append(grad_slice)
            offset += size

        return tuple(grads)


class Sigmoid(ttml.autograd.Function):
    """Autograd-aware sigmoid.

    Forward: y = sigmoid(x)
    Backward: dL/dx = dL/dy * y * (1 - y)
    """

    @staticmethod
    def forward(ctx, input):
        y = ttnn.sigmoid(input.get_value())
        ctx.save_for_backward(input)
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
        ctx.save_for_backward(input)
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
        ctx.save_for_backward(input)
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


def autograd_slice(tensor, start, end):
    """Slice with autograd backward."""
    return Slice.apply(tensor, start, end)


def autograd_concat(tensors, dim):
    """Concat with autograd backward."""
    return Concat.apply(dim, *tensors)


def autograd_sigmoid(tensor):
    """Sigmoid with autograd backward."""
    return Sigmoid.apply(tensor)


def autograd_softmax(tensor):
    """Softmax (last dim) with autograd backward."""
    return Softmax.apply(tensor)


def split_heads(tensor, num_heads):
    """(B, 1, S, H*D) -> (B, H, S, D) with proper transpose."""
    return SplitHeads.apply(tensor, num_heads)
