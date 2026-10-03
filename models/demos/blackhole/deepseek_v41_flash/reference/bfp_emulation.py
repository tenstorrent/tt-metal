# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bit-exact CPU emulation of Tenstorrent block-float (bfp4_b / bfp8_b) quantisation.

Mirrors ``convert_u32_to_bfp`` in tt_metal/impl/data_format/blockfloat_common.cpp: 16 consecutive elements along the
last dim (one tile-face row) share the largest exponent of the block; each element keeps a sign and ``mant_bits``
magnitude bits (3 for bfp4_b, 7 for bfp8_b) obtained by shifting the 24-bit mantissa right by the exponent difference,
then rounding to nearest-even on the remaining bits and saturating. Subnormals and zeros become 0.

dequantised value = sign * m * 2 ** (shared_exp - 127 - (mant_bits - 1))
"""

import torch


def bfp_roundtrip(x: torch.Tensor, mant_bits: int = 3, block: int = 16) -> torch.Tensor:
    """Quantise and dequantise ``x`` (any float dtype, last dim a multiple of ``block``) -> float32."""
    shape = x.shape
    assert shape[-1] % block == 0
    xf = x.float().contiguous().reshape(-1, block)
    bits = xf.view(torch.int32).to(torch.int64)
    exp = (bits >> 23) & 0xFF
    sign = (bits >> 31) & 1
    frac = bits & 0x7FFFFF
    shared = exp.max(dim=1, keepdim=True).values
    mant = (1 << 23) | frac
    d = (shared - exp).clamp(min=0, max=62)
    mant = mant >> d
    shift = 24 - mant_bits
    rmask = (1 << shift) - 1
    tie = 1 << (shift - 1)
    rv = mant & rmask
    m = mant >> shift
    guard = m & 1
    m = torch.where((rv > tie) | ((rv == tie) & (guard == 1)), m + 1, m)
    m = torch.clamp(m, max=(1 << mant_bits) - 1)
    m = torch.where(exp == 0, torch.zeros_like(m), m)  # zeros / subnormals
    val = m.double() * torch.exp2((shared - 127 - (mant_bits - 1)).double())
    val = torch.where(sign == 1, -val, val)
    return val.float().reshape(shape)
