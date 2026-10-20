# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""What each register can hold, as pure functions over tensors.

These are the precision rules themselves, with no knowledge of which block
applies them. They live apart from the blocks because more than one block needs
the same rule -- :func:`flush_subnormals` is taken by both the unpacker and the
FPU, and the Dest narrowing is applied on the unpack-to-Dest path, on the FPU's
write, and again by the packer as a safety net -- so none of them belongs to a
single block.

Keeping them as module-level functions also keeps them directly testable: a
rule here can be checked against a tensor without constructing an architecture.
"""

from typing import Optional, Union

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, format_dict

#: Explicit mantissa bits an IEEE float32 holds, the width every mask below is
#: cut from.
FP32_MANTISSA_BITS = 23

#: Explicit mantissa bits a src-register datum holds. The datum is 19 bits:
#: 1 sign + 8 exponent + 10 mantissa, whatever format is stored in it.
SRC_MANT_BITS = 10

#: Explicit mantissa bits a bf16 datum holds, and so what survives a Float16_b
#: Dest on the way back to a src register.
BF16_MANT_BITS = 7

#: Symmetric magnitude an INT8-shaped src datum reaches: 7 magnitude bits plus
#: a sign, so -128 has no representation on this path.
INT8_MAX_MAGNITUDE = 127

#: Formats the unpacker can land in a src register.
#:
#: Only two families exist in the datapath. ``Float16_b`` is an **alias for
#: Tf32** -- the hardware converts it to Tf32 and the distinction survives only
#: in the row/column mask path, which is not on the datum path.
#: Nothing truncates the mantissa to bf16's 7 bits, so precision in a src
#: register is ``min(input mantissa bits, 10)`` -- the src format sets the
#: exponent range, not the mantissa width.
SRC_STORAGE_FORMATS = frozenset(
    {DataFormat.Float16, DataFormat.Float16_b, DataFormat.Tf32}
)

#: Formats the unpacker can land in SrcS: the src-register formats plus full
#: Float32, which SrcA/SrcB cannot hold.
SRCS_STORAGE_FORMATS = SRC_STORAGE_FORMATS | {DataFormat.Float32}

#: Formats a Dest register can hold.
#:
#: Dest is 32-bit when accumulation is enabled and 16-bit otherwise -- that is
#: the whole of what ``DestAccumulation`` controls. Tf32 has no separate Dest
#: encoding; it lives in a Float32 container.
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

#: Integer L1 formats that can drive a 32-bit Int32 Dest. Int16 is absent on
#: purpose: the FPU supports it for MOV only and not in 32-bit Dest mode.
INT32_DEST_COMPATIBLE = frozenset({DataFormat.Int8, DataFormat.UInt8, DataFormat.Int32})

#: 32-bit Dest formats -- valid exactly when ``DestAccumulation.Yes``.
DEST_32_BIT_FORMATS = frozenset({DataFormat.Float32, DataFormat.Int32})


def truncate_mantissa(values: torch.Tensor, keep_bits: int) -> torch.Tensor:
    """Keep the top `keep_bits` explicit mantissa bits, zeroing the rest.

    Truncation, not rounding: every narrowing on the register paths drops the
    low bits rather than folding them in, so this is the one primitive behind
    all of them -- the unpacker's 10-bit src datum, a Float16_b Dest's 7 bits
    on the way back, and the high half of a fidelity phase.
    """
    raw = values.to(torch.float32).contiguous().view(torch.int32)
    return (raw & ~((1 << (FP32_MANTISSA_BITS - keep_bits)) - 1)).view(torch.float32)


def truncate_src_mantissa(values: torch.Tensor) -> torch.Tensor:
    """Keep the top 10 mantissa bits, truncating as the unpacker does."""
    return truncate_mantissa(values, SRC_MANT_BITS)


def saturate_to_integer(
    values: torch.Tensor, dtype: torch.dtype, dest_format: DataFormat
) -> torch.Tensor:
    """Clamp into `dtype`'s range, then narrow -- the hardware saturates.

    A plain cast wraps instead: a Dest result of 200 narrowed to Int8 lands as
    -56, so a later clamp in the packer sees -56 where it should have seen 127.
    A float beyond int32's range is worse, because the cast is undefined.

    Signed formats clamp to ``[min + 1, max]``: the register is sign-magnitude,
    so the most negative two's-complement value has no encoding. Unsigned
    formats use the full range.

    The same rule as ``golden_generators.saturate_integer``, duplicated for the
    reason the rest of this package duplicates -- heavyweight owns its model of
    the hardware so the old golden can be retired without stranding it.
    """
    info = torch.iinfo(dtype)
    low = info.min if dest_format.name.startswith("U") else info.min + 1
    if values.is_floating_point():
        # float64 for the clamp: int32's max is not representable in float32,
        # so clamping there rounds the bound *up* to 2^31 and the cast is
        # undefined again -- the very thing being fixed. float64 holds every
        # bound these formats use exactly. NaN survives a clamp and makes the
        # cast undefined too, so map it first.
        values = torch.nan_to_num(
            values.double(), nan=0.0, posinf=float(info.max), neginf=float(low)
        )
    return values.clamp(low, info.max).to(dtype)


def flush_subnormals(
    values: torch.Tensor, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Zero anything below the smallest normal of `dtype`, or of `values` itself.

    Neither the FPU nor the unpacker writes a denormal: a slot whose exponent
    field is zero has a zero mantissa too. Keeping the subnormal instead lets a
    value the hardware zeroed survive to the packer, which rounds it *up* onto
    the output lattice -- so a datum the device reports as 0 comes back as the
    output format's smallest representable value.

    By default the threshold comes from the tensor's own dtype, which suits a
    flush applied *after* the narrowing cast. Pass `dtype` to flush against a
    narrower format while still holding the wide value -- needed wherever the
    cast rounds, because rounding can carry the top of the subnormal range up
    to the smallest normal and so past a flush applied afterwards.
    """
    threshold = torch.finfo(dtype or values.dtype).smallest_normal
    return torch.where(
        values.abs() < threshold,
        torch.zeros_like(values),
        values,
    )


def as_dest_acc(dest_acc: Union[bool, DestAccumulation]) -> bool:
    """Normalise an accumulation setting to a plain bool.

    Accepts ``DestAccumulation`` as well as a bool, because the rest of the
    harness passes the enum and a bare ``bool`` parameter cannot be trusted to
    notice: ``DestAccumulation`` defines no ``__bool__``, so ``DestAccumulation.No``
    is **truthy**. Passed to a bool parameter it reads as "accumulation on" and
    silently selects a 32-bit Dest -- the model then runs at fp32 where the
    device had 16 bits, which looks like an arithmetic discrepancy rather than
    a wiring mistake.

    Taking the enum here and reducing to a bool once keeps the codec boundary
    (``helpers.pack`` / ``helpers.unpack``, which size tiles from a bool) plain
    while the ops speak the harness's own vocabulary.
    """
    if isinstance(dest_acc, DestAccumulation):
        return dest_acc is DestAccumulation.Yes
    return bool(dest_acc)


def to_src_storage(values: torch.Tensor, src_format: DataFormat) -> torch.Tensor:
    """Apply the precision a src-register datum can hold.

    Two losses, both from the unpacker:

    * **Mantissa** -- a src datum keeps 10 explicit bits, truncated (not
      rounded). Only Float32/Tf32 data carries more.
    * **Exponent range** -- the Float16 family has a 5-bit range, so values
      above it saturate to infinity and values below the smallest normal
      flush to zero. The Tf32 family has fp32's range and clips nothing.

    **Float32 into a src register loses its low 13 mantissa bits.** A 19-bit
    datum cannot hold fp32, and the unpacker's format table allows a Float32
    source to land only in Tf32, Float16 or Float16_b when the destination is a
    src register -- all of which keep 10. Full fp32 survives only on the
    unpack-to-Dest path and in SrcS; the ``Float32`` case below is reachable
    only from the SrcS block. Integer formats pass through unchanged.
    """
    if src_format is DataFormat.Float16:
        # 1+5+10 is IEEE fp16 exactly. Truncate first so the cast only has to
        # apply the range clamp -- casting straight to fp16 would round.
        truncated = truncate_src_mantissa(values)
        # The unpacker flushes to zero once the rebiased exponent hits 0, so
        # fp16 subnormals never reach the register. Flush *before* the cast:
        # the cast rounds to nearest, and the top subnormal band (within half
        # an ulp of 2**-14) would round up to the smallest normal and sail
        # through a flush applied afterwards.
        return flush_subnormals(truncated, torch.float16).to(torch.float16)
    if src_format in (DataFormat.Float16_b, DataFormat.Tf32):
        # Float16_b is an alias for Tf32 here -- see SRC_STORAGE_FORMATS.
        return truncate_src_mantissa(values)
    if src_format is DataFormat.Float32:
        return values.to(torch.float32)
    return values.to(format_dict[src_format])


def to_dest_storage(values: torch.Tensor, dest_format: DataFormat) -> torch.Tensor:
    """Round `values` to what a Dest slot can hold.

    The FPU produces no denormal result: a float Dest slot with a zero exponent
    has a zero mantissa too, so anything below the format's smallest normal
    lands as zero. Keeping the subnormal instead lets a value the hardware
    zeroed survive to the packer, which rounds it *up* onto the output lattice
    -- a datum the device reports as 0 comes back as the output format's
    smallest representable value. Only Float16 Dest meets the threshold in
    practice; the wider formats bottom out near 2**-126.
    """
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values)
    target = format_dict[dest_format]
    if not target.is_floating_point:
        return saturate_to_integer(values, target, dest_format)
    return flush_subnormals(values.to(target))


def dest_to_src_storage(
    values: torch.Tensor, dest_format: DataFormat, src_format: DataFormat
) -> torch.Tensor:
    """Re-quantize a Dest datum into a 19-bit src datum.

    The conversion is driven by the **Dest** format, not the src format -- the
    src format is consulted only to decide whether a wide Dest needs its
    exponent rebiasing into the 5-bit range. Every case is bit-slicing, so
    nothing rounds: bits below the target width are dropped, not folded in.

    * **Float16_b** keeps 7 mantissa bits and zero-fills the low 3, so a bf16
      Dest does **not** come back with a src register's full 10.
    * **Int16 is carried unchanged.** The hardware routes it through the same
      path by labelling it bf16, but that is transport, not conversion: the
      routing redistributes all 16 bits of the datum -- sign, then bits 14:8,
      three padding zeros, then bits 7:0 -- so every bit survives and the move
      back reassembles them. Treating the label as a conversion and masking in
      the float domain would quantize a genuine integer instead, turning 257
      into 256 and 1001 into 1000.
    * **Float16** passes as fp16 -- 10 mantissa bits, 5-bit exponent.
    * **Int32 saturates to INT8**, clamping to +/-127. A wide integer Dest
      cannot survive the trip, and the clamp is silent.
    * **Float32 / Tf32** keep 10 mantissa bits and the 8-bit exponent, unless
      the src register is Float16, in which case the exponent is rebiased and
      values below the fp16 normal range flush to zero.
    """
    if dest_format is DataFormat.Int16:
        # Bit-preserving: see the Int16 note above. The hardware borrows the
        # bf16 label only to get the datum across intact.
        return values
    if dest_format is DataFormat.Float16_b:
        # 7 explicit mantissa bits survive; the low 3 arrive as zeros.
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
