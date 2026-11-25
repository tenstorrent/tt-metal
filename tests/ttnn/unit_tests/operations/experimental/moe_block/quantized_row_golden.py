# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Torch reference of the quantized y row (ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/
quantized_row.hpp): what the producer (the flat expert's down cores, flat_routed_expert kernels/y_quant.hpp) writes and
what the consumer (the combine's accumulators) reads. Bit exact with the device.

Formats: "fp8" (e4m3 codes x an e8m0 scale) and "int8" (sign-magnitude x a scale, read as Bfp8_b); a power-of-two
scale s a block of `block` values (a multiple of 16) of a row:

  fp8   s = 2^ceil(log2(fp32(max(amax, 1e-4) x fp32(1/448))))    code = e4m3 of RNE_3(|y / s|) (fp32 rounded to 3
        mantissa bits, nearest even), 0 (+0) when that is below 2^-6 (no subnormals), sign | code otherwise
  int8  s = 2^ceil(log2(fp32(max(amax, 2^-100) x fp32(1/127))))  byte = sign | RNE(|y / s|), 0 -> 0x00 (never 0x80)

One exponent byte a 16 values: fp8 log2(s) + 127, int8 log2(s) + 133.
"""

import torch

FORMATS = ("fp8", "int8")
_LIMIT = {"fp8": 448.0, "int8": 127.0}
_FLOOR = {"fp8": 1e-4, "int8": 2.0**-100}
_BIAS = {"fp8": 127, "int8": 133}


def block_scales(y, fmt, block):
    """The power-of-two scale (fp32) of every (row, block) of y [rows, cols]: [rows, cols / block]."""
    rows = y.shape[0]
    amax = y.float().reshape(rows, -1, block).abs().amax(-1)
    v = torch.clamp(amax, min=torch.tensor(_FLOOR[fmt], dtype=torch.float32)) * torch.tensor(
        1 / _LIMIT[fmt], dtype=torch.float32
    )
    return ((v.view(torch.int32) + 0x7FFFFF) & 0x7F800000).view(torch.float32)


def quantize(y, fmt, block):
    """(value bytes uint8 [rows, cols], exponent bytes uint8 [rows, cols / 16], the values as the consumer reads them
    (fp32 [rows, cols]))."""
    rows = y.shape[0]
    s = block_scales(y, fmt, block)
    x = y.float().reshape(rows, -1, block) / s[..., None]  # (exact)
    if fmt == "fp8":
        bits = x.abs().view(torch.int32)
        bits = (bits + 0x7FFFF + ((bits >> 20) & 1)) & 0x7FF00000  # 3 mantissa bits, nearest even
        mag = (bits >> 20) - 960  # e4m3: (fp32 exponent - 120) << 3 | mantissa
        mag = torch.where(mag < 8, torch.zeros_like(mag), mag)  # below 2^-6: 0
        codes = ((((x < 0) & (mag != 0)).to(torch.int32) << 7) | mag).to(torch.uint8)
        vals = codes.view(torch.float8_e4m3fn).float()
    else:
        q = torch.round(x)  # (ties to even; |x| <= 127)
        mag = q.abs().to(torch.int32)
        codes = (((q < 0) & (mag != 0)).to(torch.int32) << 7 | mag).to(torch.uint8)
        vals = q
    vals = (vals * s[..., None]).reshape(rows, -1)
    e = ((s.view(torch.int32) >> 23) + (_BIAS[fmt] - 127)).to(torch.uint8)
    return codes.reshape(rows, -1), e.repeat_interleave(block // 16, dim=1), vals


def cells(e16, producers):
    """The exponent bytes [rows, cols / 16] as the producers' 16-byte cells [rows, producers x 16] (zero padded)."""
    rows, n = e16.shape
    per = n // producers
    out = torch.zeros(rows, producers, 16, dtype=torch.uint8)
    out[:, :, :per] = e16.reshape(rows, producers, per)
    return out.reshape(rows, -1)
