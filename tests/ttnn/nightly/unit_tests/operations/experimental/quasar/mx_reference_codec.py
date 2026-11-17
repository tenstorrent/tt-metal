# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Reference MX codec: the oracle of the Quasar MX tests. Host only.

The truth is tt-regen-spec section 4.2 (format conversions, checked on the Quasar A0 RTL) and 4.3 (the MX block),
then the Tensix Formats spec, then the OCP MX spec. The codec uses torch integer operations on float32 bit patterns.
It imports neither ttnn nor any file under tt_metal/, so it shares no code with the ttnn host codec.

Its host test is test_mx_reference_codec.py in this directory. A codec without that test is not an oracle.

Readings of 4.2 where its text gives no single answer:
- The MXINT code -2^w (0x80, 0x8, 0b10) decodes to -0. The encoder writes it only through the S = 1 wrap.
- An MX float element subnormal takes one rounding of value / X, as the text says. It is not truncated first.
- The float32 decode flush gives a zero with the sign of the datum. 4.2 states no sign there. The bfloat16 flush
  gives +0, as 4.2 states.
- An MXINT Inf saturates at every scale, also at S = 1. The S = 1 wrap applies to a finite magnitude only.
"""

from typing import NamedTuple

import torch


class MxFormat(NamedTuple):
    """The facts of one MX dtype (tt-regen-spec 4.1 and 4.2, OCP MX v1.0)."""

    kind: str  # "int" (MXINT) or "float" (MX float)
    element_bits: int  # bits of one element code
    storage_bits: int  # bits of one element in the tile (a 6-bit code sits in bits 7..2 of a byte)
    exponent_bits: int  # MX float element only; 0 for MXINT
    mantissa_bits: int  # MX float: mantissa bits; MXINT: magnitude bits w
    bias: int  # MX float element only; 0 for MXINT
    emax: int  # the element's largest unbiased exponent; 0 for MXINT


FORMATS = {
    "mxint8": MxFormat("int", 8, 8, 0, 7, 0, 0),
    "mxint4": MxFormat("int", 4, 4, 0, 3, 0, 0),
    "mxint2": MxFormat("int", 2, 2, 0, 1, 0, 0),
    "mxfp8_e4m3": MxFormat("float", 8, 8, 4, 3, 7, 8),
    "mxfp8_e5m2": MxFormat("float", 8, 8, 5, 2, 15, 15),
    "mxfp6_e2m3": MxFormat("float", 6, 8, 2, 3, 1, 2),
    "mxfp6_e3m2": MxFormat("float", 6, 8, 3, 2, 3, 4),
    "mxfp4": MxFormat("float", 4, 4, 2, 1, 1, 2),
}


class _Codes(NamedTuple):
    """The special codes of an MX float element (OCP MX v1.0), without the sign bit. -1 means that none exists."""

    max_finite: int  # the largest finite code: the target of saturation and of Inf (4.2 rule 3)
    nan: int  # the code that a NaN encodes to (4.2 rule 3); -1: a NaN encodes to +0
    inf: int  # the code that decodes to Inf
    first_nan: int  # every code from this one up to the sign bit decodes to NaN


_FLOAT_CODES = {
    "mxfp8_e4m3": _Codes(0x7E, 0x7F, -1, 0x7F),
    "mxfp8_e5m2": _Codes(0x7B, 0x7E, 0x7C, 0x7D),
    "mxfp6_e2m3": _Codes(0x1F, -1, -1, -1),
    "mxfp6_e3m2": _Codes(0x1F, -1, -1, -1),
    "mxfp4": _Codes(0x7, -1, -1, -1),
}

_BLOCK = 32  # datums in one MX block (4.2)
_F32_MAN_BITS = 23  # float32 mantissa bits
_F32_HIDDEN = 1 << _F32_MAN_BITS  # the hidden bit of a float32 normal
_F32_QNAN, _F32_INF = 0x7FC00000, 0x7F800000


def _format(fmt):
    """The MxFormat of fmt. An unknown name raises ValueError."""
    if fmt not in FORMATS:
        raise ValueError(f"mx_reference_codec: unknown MX format {fmt!r}; the formats are {sorted(FORMATS)}")
    return FORMATS[fmt]


def _round_shift(x, k):
    """x / 2^k, rounded to nearest, ties to even. x: int64, 0 <= x < 2^24. k: int64 (tensor or int), k >= 1.

    A shift of 25 or more gives 0, because x is below 2^24. So k is clamped to 25, and a large shift is safe."""
    k = torch.clamp(torch.as_tensor(k, dtype=torch.int64), 1, 25)
    one = torch.ones_like(k)
    q = x >> k
    rest = x & ((one << k) - 1)
    half = one << (k - 1)
    up = (rest > half) | ((rest == half) & ((q & 1) == 1))
    return q + up.long()


def _fields(values):
    """(sign, exponent field, significand) of float32 values, each int64 of the same shape. The significand holds
    the hidden bit; it is valid only for a normal datum."""
    if values.dtype != torch.float32:
        raise TypeError(f"mx_reference_codec: the input must be float32, not {values.dtype}")
    if values.dim() == 0 or values.shape[-1] != _BLOCK:
        raise ValueError(f"mx_reference_codec: the last dimension must be {_BLOCK}, not shape {tuple(values.shape)}")
    bits = values.contiguous().view(torch.int32).long() & 0xFFFFFFFF
    sign = bits >> 31
    exp = (bits >> _F32_MAN_BITS) & 0xFF
    sig = (bits & (_F32_HIDDEN - 1)) | _F32_HIDDEN
    return sign, exp, sig


def _scale(exp, sig, f):
    """4.2 rule 1: the scale byte S of each block. exp, sig: int64 [..., 32]. Returns int64 [...]."""
    nan = (exp == 255) & (sig != _F32_HIDDEN)
    inf = (exp == 255) & (sig == _F32_HIDDEN)
    # A float32 denormal counts as zero, so a finite non-zero datum is a normal one.
    normal = (exp > 0) & (exp < 255)
    m = torch.where(normal, exp, 0).amax(-1)
    s = m if f.kind == "int" else torch.clamp(m - f.emax, min=0)
    # An Inf with no finite non-zero datum gives 254. NaNs only give 255. NaNs and zeros only give 0 (m = 0 above).
    s = torch.where(inf.any(-1) & ~normal.any(-1), 254, s)
    return torch.where(nan.all(-1), 255, s)


def _encode_int(sign, exp, sig, s, f):
    """4.2 rule 2: the MXINT codes. sign, exp, sig: int64 [..., 32]; s: int64 [...]. Returns int64 [..., 32]."""
    w = f.mantissa_bits
    top = (1 << w) - 1
    sb = s.unsqueeze(-1)
    normal = (exp > 0) & (exp < 255)
    inf = (exp == 255) & (sig == _F32_HIDDEN)
    # Align the significand to S with a right shift that truncates. The result has 23 fraction bits.
    aligned = sig >> torch.clamp(sb - exp, 0, 24)
    # Round it to w bits (1 integer bit, w - 1 fraction bits) to nearest, ties to even.
    rounded = torch.where(normal, _round_shift(aligned, _F32_MAN_BITS + 1 - w), 0)
    # At S = 1 an overflowing magnitude wraps in the w-bit field. At every other scale it saturates.
    field = torch.where(sb == 1, rounded & top, torch.clamp(rounded, max=top))
    # Inf saturates with its sign. NaN, zero and a float32 denormal give the magnitude 0.
    field = torch.where(inf, top, field)
    # A zero magnitude is stored as +0. A magnitude that wrapped to 0 keeps its sign: the code -2^w.
    negative = (sign == 1) & ((rounded != 0) | inf)
    return torch.where(negative, ((-field) & top) | (1 << w), field)


def _encode_float(sign, exp, sig, s, fmt, f):
    """4.2 rule 3: the MX float codes. sign, exp, sig: int64 [..., 32]; s: int64 [...]. Returns int64 [..., 32]."""
    m = f.mantissa_bits
    codes = _FLOAT_CODES[fmt]
    sb = s.unsqueeze(-1)
    normal = (exp > 0) & (exp < 255)
    nan = (exp == 255) & (sig != _F32_HIDDEN)
    inf = (exp == 255) & (sig == _F32_HIDDEN)
    # value / X is sig * 2^(e - 23), with e = exp - S its unbiased exponent. The element step at e is 2^(q), with
    # q = max(e, 1 - bias) - m. Below 1 - bias the step stays at the subnormal step.
    e = exp - sb
    q = torch.clamp(e, min=1 - f.bias) - m
    # One rounding of value / X to a multiple of 2^q, to nearest, ties to even.
    n = _round_shift(sig, q - e + _F32_MAN_BITS)
    # A carry to 2^(m + 1) moves the value to the next binade.
    carry = n >> (m + 1)
    n, q = n >> carry, q + carry
    subnormal = n < (1 << m)
    magnitude = torch.where(subnormal, n, ((q + m + f.bias) << m) | (n - (1 << m)))
    # Saturate to the largest finite element. Inf saturates with its sign too.
    magnitude = torch.clamp(magnitude, max=codes.max_finite)
    magnitude = torch.where(inf, codes.max_finite, magnitude)
    # A zero and a float32 denormal keep their sign: the magnitude is 0.
    magnitude = torch.where(normal | inf, magnitude, 0)
    sign_bit = sign << (f.element_bits - 1)
    if codes.nan >= 0:
        return torch.where(nan, sign_bit | codes.nan, sign_bit | magnitude)
    # A format without a NaN code stores a NaN as +0.
    return torch.where(nan, 0, sign_bit | magnitude)


def encode_block(values: torch.Tensor, fmt: str) -> tuple:
    """Encode MX blocks by 4.2 (float32 -> MX).

    values: float32 [..., 32], one MX block per row of 32.
    Returns (scales, codes): int32 [...] scale bytes S, and int32 [..., 32] element codes in the element's own width.
    """
    f = _format(fmt)
    sign, exp, sig = _fields(values)
    scales = _scale(exp, sig, f)
    if f.kind == "int":
        codes = _encode_int(sign, exp, sig, scales, f)
    else:
        codes = _encode_float(sign, exp, sig, scales, fmt, f)
    return scales.to(torch.int32), codes.to(torch.int32)


def _msb(n):
    """floor(log2(n)) for int64 n in [1, 256). Returns 0 for n = 0."""
    return sum(((n >> j) > 0).long() for j in range(1, 8))


def _decode_bits(scales, codes, fmt, flush_keeps_sign):
    """The float32 bit patterns (int64 [..., 32]) of the decoded blocks, by 4.2 (MX -> float32 / bfloat16).

    A non-zero result below 2^-126 in magnitude becomes a zero: with the sign of the datum if flush_keeps_sign,
    else +0."""
    f = _format(fmt)
    scales = torch.as_tensor(scales).long()
    codes = torch.as_tensor(codes).long()
    if codes.dim() == 0 or codes.shape[-1] != _BLOCK or tuple(codes.shape[:-1]) != tuple(scales.shape):
        raise ValueError(
            f"mx_reference_codec: codes {tuple(codes.shape)} must be scales {tuple(scales.shape)} plus a last "
            f"dimension of {_BLOCK}"
        )
    if bool(((scales < 0) | (scales > 255)).any()) or bool(((codes < 0) | (codes >= 1 << f.element_bits)).any()):
        raise ValueError(f"mx_reference_codec: a scale is not a byte or a code is not {f.element_bits} bits")
    sign_bit = 1 << (f.element_bits - 1)
    negative = (codes & sign_bit) != 0
    m = f.mantissa_bits
    # The exponent of X = 2^(S - 127).
    x_exp = scales.unsqueeze(-1) - 127
    nan = torch.zeros_like(negative)
    inf = torch.zeros_like(negative)
    if f.kind == "int":
        # A w + 1-bit two's complement, in units of 2^-(w - 1). The code -2^w decodes to -0 (see the module docstring).
        n = torch.where(negative, (2 * sign_bit - codes) & (sign_bit - 1), codes)
        k = x_exp - (m - 1)
    else:
        magnitude = codes & (sign_bit - 1)
        e_field = magnitude >> m
        mantissa = magnitude & ((1 << m) - 1)
        n = torch.where(e_field == 0, mantissa, mantissa | (1 << m))
        k = torch.clamp(e_field, min=1) - f.bias - m + x_exp
        special = _FLOAT_CODES[fmt]
        if special.inf >= 0:
            inf = magnitude == special.inf
        if special.first_nan >= 0:
            nan = magnitude >= special.first_nan
    # The value is n * 2^k, exact. Build its float32 pattern.
    msb = _msb(n)
    e = msb + k
    bits = ((e + 127) << _F32_MAN_BITS) | ((n << (_F32_MAN_BITS - msb)) & (_F32_HIDDEN - 1))
    # Round to float32's normal range: 2^128 or more is Inf, below 2^-126 is zero. A zero code stays a zero.
    bits = torch.where(e > 127, _F32_INF, bits)
    tiny = (n != 0) & (e < -126)
    bits = torch.where((n == 0) | tiny, 0, bits)
    bits = torch.where(inf, _F32_INF, bits)
    bits = torch.where(nan, _F32_QNAN, bits)
    signed = negative if flush_keeps_sign else negative & ~tiny
    bits = bits | (signed.long() << 31)
    # S = 255 decodes every datum as NaN, with the sign of the datum.
    s255 = (scales == 255).unsqueeze(-1)
    return torch.where(s255, _F32_QNAN | (negative.long() << 31), bits)


def decode_block(scales: torch.Tensor, codes: torch.Tensor, fmt: str) -> torch.Tensor:
    """Decode MX blocks to float32 by 4.2 (MX -> float32).

    scales: int [...]; codes: int [..., 32]. Returns float32 [..., 32].
    """
    bits = _decode_bits(scales, codes, fmt, flush_keeps_sign=True)
    return torch.where(bits >= 1 << 31, bits - (1 << 32), bits).to(torch.int32).view(torch.float32)


def to_bfloat16(scales: torch.Tensor, codes: torch.Tensor, fmt: str) -> torch.Tensor:
    """Decode MX blocks to bfloat16 by 4.2 (MX -> bfloat16).

    scales: int [...]; codes: int [..., 32]. Returns bfloat16 [..., 32].
    """
    bits = _decode_bits(scales, codes, fmt, flush_keeps_sign=False)
    # An MX value has at most 8 significant bits, so the product is exact in bfloat16: the low 16 bits are 0.
    if bool((bits & 0xFFFF).any()):
        raise RuntimeError("mx_reference_codec: a decoded value is not exact in bfloat16")
    high = bits >> 16
    return torch.where(high >= 1 << 15, high - (1 << 16), high).to(torch.int16).view(torch.bfloat16)


def _block_map():
    """(rows, cols) int64 [32, 32]: element i of block b of a 32x32 tile sits at (rows[b, i], cols[b, i]).

    4.3: a block is two consecutive rows of one 16x16 face. The faces are in row-major order. Block b is face b // 8,
    rows 2 (b % 8) and 2 (b % 8) + 1 of that face."""
    b = torch.arange(_BLOCK).unsqueeze(1).expand(_BLOCK, _BLOCK)
    i = torch.arange(_BLOCK).unsqueeze(0).expand(_BLOCK, _BLOCK)
    face = b // 8
    rows = 16 * (face // 2) + 2 * (b % 8) + i // 16
    cols = 16 * (face % 2) + i % 16
    return rows, cols


def _element_slots(f):
    """(byte, shift) int64 [1024]: the data byte (after the 32 scale bytes) and the bit offset of element 32 b + i.

    Elements fill each byte from the low bits up: the low nibble holds the even element of a 4-bit pair."""
    bit = torch.arange(_BLOCK * _BLOCK) * f.storage_bits
    return bit // 8, bit % 8


def encode_tile(tile: torch.Tensor, fmt: str) -> torch.Tensor:
    """Encode one 32x32 tile by 4.2 and 4.3: the 32 scale bytes first, then the elements in their storage width.

    tile: float32 [32, 32]. Returns uint8 [tile bytes] (1,056 B for an 8-bit slot, 544 B for 4 bits, 288 B for 2).
    The conversion is encode_block's: the tile functions add the layout only.
    """
    f = _format(fmt)
    if tuple(tile.shape) != (_BLOCK, _BLOCK):
        raise ValueError(f"mx_reference_codec: a tile is [32, 32], not {tuple(tile.shape)}")
    rows, cols = _block_map()
    scales, codes = encode_block(tile[rows, cols], fmt)
    # A 6-bit code sits in bits 7..2 of its byte.
    units = codes.reshape(-1).long() << (f.storage_bits - f.element_bits)
    byte, shift = _element_slots(f)
    data = torch.zeros(_BLOCK * _BLOCK * f.storage_bits // 8, dtype=torch.int64)
    data.index_add_(0, byte, units << shift)
    return torch.cat([scales.long(), data]).to(torch.uint8)


def decode_tile(raw: torch.Tensor, fmt: str) -> torch.Tensor:
    """Decode one 32x32 MX tile (the bytes of encode_tile) to float32 [32, 32] by 4.2 and 4.3.

    The conversion is decode_block's: the tile functions add the layout only.
    """
    f = _format(fmt)
    size = _BLOCK + _BLOCK * _BLOCK * f.storage_bits // 8
    raw = raw.reshape(-1).long()
    if raw.numel() != size:
        raise ValueError(f"mx_reference_codec: an {fmt} tile is {size} bytes, not {raw.numel()}")
    byte, shift = _element_slots(f)
    units = (raw[_BLOCK + byte] >> shift) & ((1 << f.storage_bits) - 1)
    codes = (units >> (f.storage_bits - f.element_bits)).reshape(_BLOCK, _BLOCK)
    rows, cols = _block_map()
    out = torch.zeros(_BLOCK, _BLOCK, dtype=torch.float32)
    out[rows, cols] = decode_block(raw[:_BLOCK], codes, fmt)
    return out
