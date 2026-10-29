# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Preserve PyTorch eager BF16 rounding at elementwise operation boundaries.

Direct BF16 arithmetic on the qualified Blackhole runtime rounds halfway sums
away from zero. PyTorch uses ties-to-even. Computing an individual operation in
FP32 and then explicitly casting to BF16 supplies the matching rounding boundary;
keeping an entire expression in FP32 would change the reference's eager semantics.
"""
from __future__ import annotations

import ttnn


def rounded_bf16(operation, *operands, **kwargs):
    """Apply one operation, promoting exclusively BF16 tensor operands to FP32.

    FP32 and mixed-dtype operations retain their existing dtype/semantics. Scalars
    accompanying BF16 tensors are evaluated in FP32, avoiding premature rounding
    of constants such as 1/25 to BF16.
    """
    tensors = [operand for operand in operands if isinstance(operand, ttnn.Tensor)]
    if tensors and all(operand.dtype == ttnn.bfloat16 for operand in tensors):
        promoted = tuple(
            ttnn.typecast(operand, ttnn.float32) if isinstance(operand, ttnn.Tensor) else operand
            for operand in operands
        )
        return ttnn.typecast(operation(*promoted, **kwargs), ttnn.bfloat16)
    return operation(*operands, **kwargs)


def add(*operands, **kwargs):
    return rounded_bf16(ttnn.add, *operands, **kwargs)


def subtract(*operands, **kwargs):
    return rounded_bf16(ttnn.subtract, *operands, **kwargs)


def multiply(*operands, **kwargs):
    return rounded_bf16(ttnn.multiply, *operands, **kwargs)


def silu(value, **kwargs):
    return rounded_bf16(ttnn.silu, value, **kwargs)


def sigmoid(value, **kwargs):
    return rounded_bf16(ttnn.sigmoid, value, **kwargs)


def tanh(value, **kwargs):
    return rounded_bf16(ttnn.tanh, value, **kwargs)


def rotary_split_half(value, cosine, sine):
    """Native split-half RoPE with two BF16 products and one BF16 sum."""
    half = value.shape[-1] // 2
    xf = ttnn.typecast(value, ttnn.float32)
    rotated = ttnn.concat((ttnn.neg(xf[..., half:]), xf[..., :half]), dim=-1)
    a = ttnn.typecast(ttnn.multiply(xf, ttnn.typecast(cosine, ttnn.float32)), ttnn.bfloat16)
    b = ttnn.typecast(ttnn.multiply(rotated, ttnn.typecast(sine, ttnn.float32)), ttnn.bfloat16)
    return add(a, b)
