# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Autograd-aware operations shared by TTML model implementations."""

import ttnn
import ttml


class SequencePrefixSlice(ttml.autograd.Function):
    """Slice ``x[:, :, :length, :]`` and zero-pad its gradient back to ``x``."""

    @staticmethod
    def forward(ctx, x, length):
        shape = x.shape()
        ctx.full_sequence_size = shape[2]
        ctx.sequence_size = length
        return ttnn.slice(
            x.get_value(),
            [0, 0, 0, 0],
            [shape[0], shape[1], length, shape[3]],
        )

    @staticmethod
    def backward(ctx, grad_output):
        padding_size = ctx.full_sequence_size - ctx.sequence_size
        padding = [(0, 0), (0, 0), (0, padding_size), (0, 0)]
        return ttnn.pad(grad_output, padding=padding, value=0.0)
