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

from typing import Optional

import torch
from helpers.format_config import DataFormat

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
