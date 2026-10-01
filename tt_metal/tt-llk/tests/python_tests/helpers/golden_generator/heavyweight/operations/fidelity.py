# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""FPU fidelity phases, shared by the operations that model them.

The multiplier is narrower than a src-register datum, so a multiply is run
several times over different slices of the operands' mantissas and the partial
products are accumulated. ``MathFidelity`` chooses how many of those passes
run.

These primitives live here rather than on an operation because both the
element-wise multiply and the matmul decompose the same way -- the only
difference is whether the partial product is an element-wise product or a
matrix one.
"""

import math
from typing import Optional, Tuple

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import MathFidelity, format_dict

#: Phases each MathFidelity runs. The FPU decomposes a multiply into partial
#: products (AH_BH, AL_BH, AH_BL, AL_BL) accumulated across passes, and
#: fidelity chooses how many of them to run.
FIDELITY_PHASES = {
    MathFidelity.LoFi: 1,
    MathFidelity.HiFi2: 2,
    MathFidelity.HiFi3: 3,
    MathFidelity.HiFi4: 4,
}

#: Which half of each operand a phase uses, in FPU phase order.
PHASE_OPERAND_HALVES = (
    ("hi", "hi"),  # AH_BH — the most significant partial product
    ("lo", "hi"),  # AL_BH
    ("hi", "lo"),  # AH_BL
    ("lo", "lo"),  # AL_BL — the least significant
)

#: Explicit mantissa bits a float32 datum carries.
FP32_MANTISSA_BITS = 23


def split_mantissa(
    values: torch.Tensor, keep_bits: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split into (high, low) at `keep_bits` explicit mantissa bits.

    The low half is taken as ``value - high`` rather than by masking bits in
    place: the implicit leading 1 belongs to the high half, so a masked
    mantissa re-read as a float is not the remainder. Subtracting is exact and
    needs no implicit-bit bookkeeping.

    Equivalent to the mask-and-reassemble the lightweight golden performs. Its
    masks keep the top ``1 + keep_bits`` bits of an 11-bit ``{implicit,
    mantissa}`` field for the high half and the remaining bits with the
    implicit position cleared for the low half, which reassembles to exactly
    this remainder.
    """
    raw = values.to(torch.float32).contiguous().view(torch.int32)
    high = (raw & ~((1 << (FP32_MANTISSA_BITS - keep_bits)) - 1)).view(torch.float32)
    return high, values.to(torch.float32) - high


def operand_halves(
    srcA: torch.Tensor,
    srcB: torch.Tensor,
    split: Tuple[int, int],
    phase: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The slice of each operand that fidelity `phase` multiplies."""
    a_bits, b_bits = split
    a_half, b_half = PHASE_OPERAND_HALVES[phase]
    a_hi, a_lo = split_mantissa(srcA, a_bits)
    b_hi, b_lo = split_mantissa(srcB, b_bits)
    return (
        a_hi if a_half == "hi" else a_lo,
        b_hi if b_half == "hi" else b_lo,
    )


def min_normal_exponent(dest_format: DataFormat) -> Optional[int]:
    """Lowest exponent the Dest format holds as a normal, or None if integer."""
    dtype = format_dict[dest_format]
    if not dtype.is_floating_point:
        return None
    return int(math.log2(torch.finfo(dtype).smallest_normal))


def ieee_exponent(values: torch.Tensor) -> torch.Tensor:
    """The exponent an IEEE float stores for each value.

    ``torch.frexp`` returns a significand in [0.5, 1), so its exponent is one
    above the one the FP lane's exponent field carries.
    """
    _, exponent = torch.frexp(values.to(torch.float32))
    return exponent - 1


def flush_pre_carry_denormals(
    product: torch.Tensor,
    srcA: torch.Tensor,
    srcB: torch.Tensor,
    min_exponent: Optional[int],
) -> torch.Tensor:
    """Zero a product whose exponent lands below Dest's lowest normal.

    The FP lane forms the exponent by adding the two *stored src* exponents and
    rebiasing into Dest's range, then flushes the whole term -- mantissa
    included -- when that lands below the lowest normal. The decision is taken
    **before** the mantissa product's carry into the next binade, so the rule
    is one binade coarser than testing the finished magnitude: when the two
    mantissas multiply to 2.0 or more a product anywhere in
    ``[tiny, 2 * tiny)`` is a legitimate Dest normal that hardware still
    returns as zero.

    The exponent is a property of the src **datum** -- one field shared by
    every phase, since a phase selects a mantissa window and not an exponent --
    so the sum is identical on all four phases. Using the exponent of a split
    half instead flushes almost everything.

    Applies to an element-wise multiply, where each product is a lane result
    written to Dest. It does **not** apply inside a matmul: there the products
    enter the sum-of-products network at full width
    (``SOP_IN_MAN_PREC = (MAN_PREC_A+1) + (MAN_PREC_B+1)``) and never become
    lane results, so only the accumulated sum is subject to the rule.
    """
    if min_exponent is None:
        return product
    pre_carry = ieee_exponent(srcA) + ieee_exponent(srcB)
    return torch.where(pre_carry < min_exponent, torch.zeros_like(product), product)
