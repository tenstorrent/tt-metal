# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device FP32 arithmetic at boundaries that affect discrete expert selection."""

import ttnn


def norm_weight(weight, width):
    """Prepare a broadcastable FP32 norm weight during model setup."""
    weight = ttnn.reshape(weight, (1, 1, 1, width))
    return ttnn.to_layout(ttnn.typecast(weight, ttnn.float32), ttnn.TILE_LAYOUT)


def rms_norm(value, epsilon, weight=None):
    """Use SFPU reductions; fused RMSNorm narrows FP32 operands through Src."""
    value = ttnn.typecast(value, ttnn.float32)
    variance = ttnn.mean(ttnn.mul(value, value), dim=-1, keepdim=True)
    scale = ttnn.rsqrt(ttnn.add(variance, epsilon), fast_and_approximate_mode=False)
    result = ttnn.mul(value, scale)
    return result if weight is None else ttnn.mul(result, weight)


def rotary(value, cos, sin, *, decode=False):
    """Preserve FP32 products while using the caller's supplied RoPE values."""
    if decode:
        repeats = (1, 1, value.shape[-2], 1)
        cos = ttnn.repeat(cos[:, :, :1, :], repeats)
        sin = ttnn.repeat(sin[:, :, :1, :], repeats)
    else:
        cos, sin = cos[:, :, : value.shape[-2], :], sin[:, :, : value.shape[-2], :]
    half = value.shape[-1] // 2
    rotated = ttnn.concat((ttnn.neg(value[..., half:]), value[..., :half]), dim=-1)
    return ttnn.add(
        ttnn.mul(value, ttnn.typecast(cos, ttnn.float32)),
        ttnn.mul(rotated, ttnn.typecast(sin, ttnn.float32)),
    )
