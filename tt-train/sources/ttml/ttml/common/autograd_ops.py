# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Autograd-aware split and concat (ttnn.split / ttnn.concat have no backward in ttml)."""

from __future__ import annotations

import ttnn
import ttml


def _normalize_dim(dim: int, rank: int) -> int:
    if not -rank <= dim < rank:
        raise ValueError(f"dim {dim} out of range for rank-{rank} tensor")
    return dim % rank


def _slices(value: ttnn.Tensor, sizes, dim: int) -> tuple[ttnn.Tensor, ...]:
    """Contiguous chunks along `dim` with ttnn.slice; ttnn.split only runs on a few cores."""
    out, start = [], 0
    for size in sizes:
        begin, end = [0] * len(value.shape), list(value.shape)
        begin[dim], end[dim] = start, start + size
        out.append(ttnn.slice(value, begin, end))
        start += size
    return tuple(out)


class Split(ttml.autograd.Function):
    """Contiguous chunks of `sizes` along `dim`; backward is a single concat.

    Cheaper than several slices of one tensor, whose backward builds a zero-padded full-width gradient per slice
    and sums them."""

    @staticmethod
    def forward(ctx, input, sizes, dim):
        value = input.get_value()
        dim = _normalize_dim(dim, len(value.shape))
        # Chunks must tile the dim exactly, or the backward concat would not rebuild the input gradient.
        if sum(sizes) != value.shape[dim]:
            raise ValueError(f"split sizes {list(sizes)} must tile dim {dim} (size {value.shape[dim]})")
        ctx.dim = dim
        return _slices(value, sizes, dim)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return ttnn.concat(list(grad_outputs), ctx.dim)


class Concat(ttml.autograd.Function):
    """Concatenation along `dim`; backward slices the gradient back into the inputs' pieces."""

    @staticmethod
    def forward(ctx, dim, *inputs):
        values = [t.get_value() for t in inputs]
        ctx.dim = _normalize_dim(dim, len(values[0].shape))
        ctx.sizes = [v.shape[ctx.dim] for v in values]
        return ttnn.concat(values, ctx.dim)

    @staticmethod
    def backward(ctx, grad_output):
        return _slices(grad_output, ctx.sizes, ctx.dim)
