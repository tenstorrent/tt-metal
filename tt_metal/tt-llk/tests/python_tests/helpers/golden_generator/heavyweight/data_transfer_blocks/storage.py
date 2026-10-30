# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""What each register can hold, as pure functions over tensors.

The precision rules on their own, with no idea which block applies them. They
sit here rather than inside a unit because several units need the same rule.

See `README.md`, "Register storage", for what each one models.
"""

from typing import Optional, Union

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, format_dict

#: Explicit mantissa bits in an IEEE float32, the width every mask is cut from.
FP32_MANTISSA_BITS = 23

#: What an IEEE float32 adds to an exponent before storing it.
FP32_EXPONENT_BIAS = 127

#: Explicit mantissa bits a src-register datum holds (the datum is 19 bits:
#: 1 sign + 8 exponent + 10 mantissa).
SRC_MANT_BITS = 10

#: Explicit mantissa bits a bf16 datum holds.
BF16_MANT_BITS = 7

#: Explicit mantissa bits of E8M6, the per-datum intermediate the packer
#: narrows to before it shares a block float's exponent.
E8M6_MANT_BITS = 6

#: Largest magnitude an INT8-shaped src datum reaches. Sign-magnitude, so -128
#: has no encoding on this path.
INT8_MAX_MAGNITUDE = 127

#: Formats the unpacker can land in a src register. ``Float16_b`` is an alias
#: for Tf32 here, so it keeps 10 mantissa bits rather than bf16's 7.
SRC_STORAGE_FORMATS = frozenset(
    {DataFormat.Float16, DataFormat.Float16_b, DataFormat.Tf32}
)

#: Formats the unpacker can land in SrcS: the src formats plus Float32.
SRCS_STORAGE_FORMATS = SRC_STORAGE_FORMATS | {DataFormat.Float32}

#: Formats a Dest register can hold. 32-bit with accumulation, 16-bit without.
DEST_STORAGE_FORMATS = frozenset(
    {
        DataFormat.Float32,
        DataFormat.Int32,
        DataFormat.Float16,
        DataFormat.Float16_b,
        DataFormat.Int16,
        DataFormat.Int8,
        DataFormat.UInt8,
    }
)

#: Integer L1 formats that can drive an Int32 Dest. Int16 is left out on
#: purpose: the FPU takes it for MOV only, not in 32-bit Dest mode.
INT32_DEST_COMPATIBLE = frozenset({DataFormat.Int8, DataFormat.UInt8, DataFormat.Int32})

#: 32-bit Dest formats, valid exactly when ``DestAccumulation.Yes``.
DEST_32_BIT_FORMATS = frozenset({DataFormat.Float32, DataFormat.Int32})


def truncate_mantissa(values: torch.Tensor, keep_bits: int) -> torch.Tensor:
    """Keep the top `keep_bits` mantissa bits and zero the rest.

    Truncation, not rounding. Every narrowing on the register paths drops the
    low bits rather than folding them in, so this one primitive is behind all
    of them.
    """
    raw = values.to(torch.float32).contiguous().view(torch.int32)
    return (raw & ~((1 << (FP32_MANTISSA_BITS - keep_bits)) - 1)).view(torch.float32)


def round_mantissa_away(values: torch.Tensor, keep_bits: int) -> torch.Tensor:
    """Round to `keep_bits` mantissa bits, ties away from zero.

    IEEE and torch both round ties to *even*. Some FPUs round ties away from
    zero instead, which only shows up when the dropped bits are exactly half an
    ULP. Random data almost never lands there, but an add or subtract of two
    datums already on the target grid lands there often.

    Adding half an ULP to the bit pattern and truncating gives ties-away,
    because a float is sign-magnitude: the add grows the magnitude whatever the
    sign. Carries into the exponent are correct, so a round up at the top of a
    binade moves to the next one.
    """
    dropped = FP32_MANTISSA_BITS - keep_bits
    wide = values.to(torch.float32)
    raw = wide.contiguous().view(torch.int32)
    rounded = ((raw + (1 << (dropped - 1))) & ~((1 << dropped) - 1)).view(torch.float32)
    # Inf and NaN have no mantissa to round, and the add would corrupt them.
    return torch.where(torch.isfinite(wide), rounded, wide)


def round_mantissa_even(values: torch.Tensor, keep_bits: int) -> torch.Tensor:
    """Round to `keep_bits` mantissa bits, ties to even.

    What IEEE and torch do, written out at an arbitrary width so a conversion
    the hardware makes at a width torch has no dtype for still rounds the same
    way. Adding half an ULP less one, plus the low kept bit, carries exactly
    when the remainder is over half or the kept bit is already odd.
    """
    dropped = FP32_MANTISSA_BITS - keep_bits
    wide = values.to(torch.float32)
    raw = wide.contiguous().view(torch.int32)
    bias = (1 << (dropped - 1)) - 1 + ((raw >> dropped) & 1)
    rounded = ((raw + bias) & ~((1 << dropped) - 1)).view(torch.float32)
    # Inf and NaN have no mantissa to round, and the add would corrupt them.
    return torch.where(torch.isfinite(wide), rounded, wide)


def mantissa_ties(values: torch.Tensor, keep_bits: int) -> torch.Tensor:
    """Where rounding to `keep_bits` lands exactly half an ULP from both sides.

    The only place the rounding mode is observable. Everywhere else ties-to-even
    and ties-away agree, so a model that has not pinned the mode down is still
    right until one of these appears.
    """
    dropped = FP32_MANTISSA_BITS - keep_bits
    wide = values.to(torch.float32)
    low = wide.contiguous().view(torch.int32) & ((1 << dropped) - 1)
    return (low == (1 << (dropped - 1))) & torch.isfinite(wide)


def truncate_src_mantissa(values: torch.Tensor) -> torch.Tensor:
    """Keep the top 10 mantissa bits, truncating as the unpacker does."""
    return truncate_mantissa(values, SRC_MANT_BITS)


def saturate_to_integer(
    values: torch.Tensor, dtype: torch.dtype, dest_format: DataFormat
) -> torch.Tensor:
    """Clamp into `dtype`'s range, then narrow. The hardware saturates.

    A plain cast would wrap instead: 200 narrowed to Int8 lands as -56. Signed
    formats clamp to ``[min + 1, max]`` because the register is sign-magnitude
    and the most negative two's-complement value has no encoding.
    """
    info = torch.iinfo(dtype)
    low = info.min if dest_format.name.startswith("U") else info.min + 1
    if values.is_floating_point():
        # float64, because int32's max is not representable in float32. That
        # would round the bound up to 2**31 and leave the cast undefined, which
        # is the bug being fixed. NaN survives a clamp, so map it first.
        values = torch.nan_to_num(
            values.double(), nan=0.0, posinf=float(info.max), neginf=float(low)
        )
    return values.clamp(low, info.max).to(dtype)


def flush_subnormals(
    values: torch.Tensor, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Zero anything below the smallest normal of `dtype`, or of `values` itself.

    Neither the FPU nor the unpacker writes a denormal. By default the
    threshold comes from the tensor's own dtype, which suits a flush applied
    after the narrowing cast. Pass `dtype` to flush against a narrower format
    while still holding the wide value, needed wherever the cast rounds,
    since rounding can lift the top of the subnormal range past a later flush.
    """
    threshold = torch.finfo(dtype or values.dtype).smallest_normal
    return torch.where(
        values.abs() < threshold,
        torch.zeros_like(values),
        values,
    )


def as_dest_acc(dest_acc: Union[bool, DestAccumulation]) -> bool:
    """Normalise an accumulation setting to a plain bool.

    Takes the enum as well as a bool, and you want it: ``DestAccumulation``
    defines no ``__bool__``, so ``DestAccumulation.No`` is **truthy** and reads
    as "accumulation on" wherever a plain ``if`` tests it.
    """
    if isinstance(dest_acc, DestAccumulation):
        return dest_acc is DestAccumulation.Yes
    return bool(dest_acc)


# Explicit mantissa bits a 16-bit Dest slot keeps.
DEST_MANTISSA_BITS = {
    DataFormat.Float16: SRC_MANT_BITS,
    DataFormat.Float16_b: BF16_MANT_BITS,
    DataFormat.Tf32: SRC_MANT_BITS,
}

#: Explicit mantissa bits an L1 float format holds, for the formats whose
#: lattice is a plain mantissa width. The block-scaled families are absent on
#: purpose: their step is set by an exponent shared across the block, so a
#: per-datum mantissa width does not describe them. See
#: :data:`L1_BLOCK_MAGNITUDE_BITS` for the block-float ones.
L1_MANTISSA_BITS = {
    DataFormat.Float32: FP32_MANTISSA_BITS,
    DataFormat.Tf32: SRC_MANT_BITS,
    DataFormat.Float16: SRC_MANT_BITS,
    DataFormat.Float16_b: BF16_MANT_BITS,
    DataFormat.Fp8_e4m3: 3,
}

#: Lowest exponent each of those formats holds as a normal.
L1_MIN_NORMAL_EXP = {
    DataFormat.Float32: -126,
    DataFormat.Tf32: -126,
    DataFormat.Float16: -14,
    DataFormat.Float16_b: -126,
    DataFormat.Fp8_e4m3: -6,
}

#: Datums a block-float format shares one exponent across.
BFP_BLOCK_SIZE = 16

#: Magnitude bits a block-float datum stores, the sign bit excluded. The block
#: exponent carries the scale, so the leading bit is stored rather than implied
#: and Bfp8_b's 7 bits are worth 6 explicit mantissa bits plus that one.
L1_BLOCK_MAGNITUDE_BITS = {
    DataFormat.Bfp8_b: 7,
    DataFormat.Bfp4_b: 3,
    DataFormat.Bfp2_b: 1,
}

#: Explicit mantissa bits a src register keeps per format.
SRC_MANTISSA_BITS = {
    DataFormat.Float16: SRC_MANT_BITS,
    DataFormat.Float16_b: BF16_MANT_BITS,
    DataFormat.Tf32: SRC_MANT_BITS,
}


def to_src_storage(
    values: torch.Tensor,
    src_format: DataFormat,
    mantissa_bits: Optional[int] = None,
) -> torch.Tensor:
    """Apply the precision a src-register datum can hold.

    Two losses, both the unpacker's: the mantissa truncates, and the Float16
    family's 5-bit exponent saturates or flushes what falls outside it.
    Integers pass through unchanged.

    `mantissa_bits` is how many explicit bits this architecture keeps for
    `src_format`, defaulting to :data:`SRC_MANTISSA_BITS`. It is per
    architecture because Quasar aliases Float16_b onto Tf32.
    """
    if mantissa_bits is None:
        mantissa_bits = SRC_MANTISSA_BITS.get(src_format, SRC_MANT_BITS)
    if src_format is DataFormat.Float16:
        # 1+5+10 is IEEE fp16 exactly. Truncate first so the cast only has to
        # apply the range clamp. Casting straight to fp16 would round.
        truncated = truncate_mantissa(values, mantissa_bits)
        # Flush before the cast, not after: the cast rounds to nearest, and the
        # top of the subnormal band would round up to the smallest normal and
        # sail through a later flush.
        return flush_subnormals(truncated, torch.float16).to(torch.float16)
    if src_format in (DataFormat.Float16_b, DataFormat.Tf32):
        return truncate_mantissa(values, mantissa_bits)
    if src_format is DataFormat.Float32:
        return values.to(torch.float32)
    return values.to(format_dict[src_format])


def truncate_to_block_lattice(
    values: torch.Tensor, magnitude_bits: int, l1_format: DataFormat
) -> torch.Tensor:
    """Quantize toward zero onto a block-float lattice.

    Each block takes the largest biased exponent among its datums, and every
    datum then stores a magnitude weighted
    ``2 ** (shared - FP32_EXPONENT_BIAS - (magnitude_bits - 1))``. Flooring that
    magnitude is the whole conversion, since the leading bit is stored rather
    than implied and there is nothing left to normalise.
    """
    wide = values.to(torch.float32)
    if wide.numel() % BFP_BLOCK_SIZE:
        raise ValueError(
            f"{l1_format} shares one exponent across {BFP_BLOCK_SIZE} datums, "
            f"so {wide.numel()} datums is not a whole number of blocks."
        )
    blocks = wide.reshape(-1, BFP_BLOCK_SIZE)
    raw = blocks.contiguous().view(torch.int32)
    # Every datum arriving here has a real exponent, because Dest flushes
    # subnormals before the packer reads it. An exponent field of zero would
    # mean the leading bit below is fabricated rather than stored.
    shared = ((raw >> FP32_MANTISSA_BITS) & 0xFF).amax(dim=1, keepdim=True)
    # float64 for the step and the division: a block of small exponents puts
    # the step below float32's smallest normal, where it would flush to zero
    # and the division would hand back an infinity.
    weight = shared - FP32_EXPONENT_BIAS - (magnitude_bits - 1)
    step = torch.pow(2.0, weight.double())
    magnitude = torch.clamp(
        torch.floor(blocks.double().abs() / step), max=(1 << magnitude_bits) - 1
    )
    quantized = torch.sign(blocks.double()) * magnitude * step
    # A magnitude that floors to zero drops its sign. The codecs do the same,
    # because the unpacker reads a sign bit over a zero magnitude as -inf.
    quantized = torch.where(magnitude == 0, torch.zeros_like(quantized), quantized)
    return quantized.to(torch.float32).reshape(values.shape)


def truncate_to_l1_lattice(
    values: torch.Tensor, l1_format: DataFormat
) -> Optional[torch.Tensor]:
    """Quantize toward zero onto `l1_format`'s lattice, or None if unmodelled.

    For a packer that truncates where the codec rounds. Dropping whatever the
    format cannot hold lands the value on that format's lattice, so the codec's
    own conversion afterwards has nothing left to round.

    Returns None for a format whose lattice this module does not describe,
    which today means the MX family and its per-block scale.
    """
    block_bits = L1_BLOCK_MAGNITUDE_BITS.get(l1_format)
    if block_bits is not None:
        return truncate_to_block_lattice(values, block_bits, l1_format)
    keep = L1_MANTISSA_BITS.get(l1_format)
    if keep is None:
        return None
    wide = values.to(torch.float32)
    out = truncate_mantissa(wide, keep)
    # Subnormals have a fixed step instead of a relative one, so truncating a
    # mantissa width keeps more precision than the format can hold there.
    min_exp = L1_MIN_NORMAL_EXP.get(l1_format)
    if min_exp is None:
        return out
    step = 2.0 ** (min_exp - keep)
    floored = torch.sign(wide) * torch.floor(wide.abs() / step) * step
    return torch.where(wide.abs() < 2.0**min_exp, floored, out)


def to_dest_storage(
    values: torch.Tensor, dest_format: DataFormat, ties_away: bool = False
) -> torch.Tensor:
    """Round `values` to what a Dest slot can hold.

    The FPU writes no denormal, so anything below the format's smallest normal
    lands as zero. In practice only a Float16 Dest reaches that threshold.

    `ties_away` picks the rounding mode, and is per architecture. The default
    is the IEEE one torch gives, ties to even. Blackhole rounds ties away from
    zero on a 16-bit Dest write.
    """
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values)
    target = format_dict[dest_format]
    if not target.is_floating_point:
        return saturate_to_integer(values, target, dest_format)
    keep = DEST_MANTISSA_BITS.get(dest_format)
    if ties_away and keep is not None:
        # Already on the target grid, so the cast below cannot round again.
        values = round_mantissa_away(values, keep)
    return flush_subnormals(values.to(target))


def dest_to_src_storage(
    values: torch.Tensor, dest_format: DataFormat, src_format: DataFormat
) -> torch.Tensor:
    """Re-quantize a Dest datum into a 19-bit src datum.

    Driven by the **Dest** format. `src_format` only decides whether a wide
    Dest needs its exponent rebiased into the 5-bit range. Every case is
    bit-slicing, so nothing rounds.

    See `README.md`, "The feedback path", for what each Dest format costs.
    """
    if dest_format is DataFormat.Int16:
        # Carried unchanged. The hardware borrows the bf16 label to move the
        # datum, but that is transport, not conversion. Masking in the float
        # domain here would quantize a genuine integer.
        return values
    if dest_format is DataFormat.Float16_b:
        # 7 explicit mantissa bits survive, and the low 3 arrive as zeros.
        return truncate_mantissa(values, BF16_MANT_BITS)
    if dest_format is DataFormat.Float16:
        return values.to(torch.float16).to(torch.float32)
    if dest_format is DataFormat.Int32:
        # Only 7 magnitude bits plus a sign reach the src register.
        return (
            values.to(torch.float32)
            .clamp(-INT8_MAX_MAGNITUDE, INT8_MAX_MAGNITUDE)
            .trunc()
        )
    if dest_format in (DataFormat.Float32, DataFormat.Tf32):
        if src_format is DataFormat.Float16:
            return to_src_storage(values, DataFormat.Float16)
        return truncate_src_mantissa(values)
    return to_src_storage(values, src_format)
