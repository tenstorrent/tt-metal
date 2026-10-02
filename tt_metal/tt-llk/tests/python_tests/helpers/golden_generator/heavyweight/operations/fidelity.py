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
import warnings
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


def warn_unmodelled_split(op_name: str, golden_name: str) -> None:
    """Say so when a multiply falls back to the exact product.

    A multiply whose ``MANTISSA_SPLIT`` is ``None`` is computed exactly and
    ``math_fidelity`` is ignored entirely. That is not a near-miss: hardware
    truncates both operands' mantissas on every phase, so the golden comes out
    *more* accurate than the device at every fidelity, and a test comparing
    against it sees real mismatches with nothing obviously wrong in the golden.

    Note "every fidelity", not just the reduced ones. On Wormhole and Blackhole
    the per-phase masks never cover SrcA's least significant bit -- the four
    phases reach mantissa bits 9..1 and bit 0 participates in none of them -- so
    even HiFi4 is not the exact product there. HiFi4 is merely the closest.
    """
    warnings.warn(
        f"{golden_name} computes {op_name} as an exact product: this "
        f"architecture's per-phase mantissa split is not modelled, so "
        f"math_fidelity is ignored and the result is more accurate than the "
        f"device at every fidelity, HiFi4 included. Treat it as a reference, "
        f"not as an exact golden.",
        stacklevel=3,
    )


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


def resolve_non_finite(
    a_half: torch.Tensor,
    b_half: torch.Tensor,
    srcA: torch.Tensor,
    srcB: torch.Tensor,
    phase: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fix up operand positions holding an infinity or a NaN.

    A non-finite datum has no mantissa to split, and ``split_mantissa``'s
    ``value - high`` makes ``Inf - Inf = NaN``. Zeroing only the low half is not
    enough either: a later phase would then multiply the surviving infinity by a
    zero low half, and ``Inf * 0`` is a NaN by IEEE.

    So carry the raw value at those positions on the first phase, where it is
    the whole of that datum's contribution, and zero on every later phase. The
    finite positions keep their truncated halves, so LoFi stays truncated, and
    an infinity reaches the result once rather than being split, flushed or
    turned into a NaN. ``Inf + 0`` stays ``Inf``; a NaN propagates as a NaN.

    Applies to the element-wise and matrix products alike: in a matmul the
    phase-0 term carries the infinity into every output position that sums over
    it, and the later phases add exact zeros there.
    """
    bad_a = ~torch.isfinite(srcA.float())
    bad_b = ~torch.isfinite(srcB.float())
    if not (bool(bad_a.any()) or bool(bad_b.any())):
        return a_half, b_half
    if phase == 0:
        return (
            torch.where(bad_a, srcA.float(), a_half),
            torch.where(bad_b, srcB.float(), b_half),
        )
    return (
        torch.where(bad_a, torch.zeros_like(a_half), a_half),
        torch.where(bad_b, torch.zeros_like(b_half), b_half),
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


def non_finite_lanes(srcA: torch.Tensor, srcB: torch.Tensor) -> torch.Tensor:
    """Lanes where either operand is an infinity or a NaN.

    These need excluding from the mantissa-split and flush rules, both of which
    assume a finite operand with a meaningful exponent and mantissa. ``frexp``
    reports exponent 0 for a non-finite value, which reads as a magnitude near
    2^-1 and would make the flush rule treat an infinity as a tiny number.
    """
    return ~(torch.isfinite(srcA.float()) & torch.isfinite(srcB.float()))


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
    # A non-finite operand stores an all-ones exponent field, so the lane's
    # exponent adder can never land below the normal floor and no flush occurs.
    # Without this guard frexp's exponent-0 for Inf/NaN reads as ~2^-1 and
    # Inf * b flushes to zero for any |b| below the floor.
    pre_carry = ieee_exponent(srcA) + ieee_exponent(srcB)
    dead = (pre_carry < min_exponent) & ~non_finite_lanes(srcA, srcB)
    return torch.where(dead, torch.zeros_like(product), product)
