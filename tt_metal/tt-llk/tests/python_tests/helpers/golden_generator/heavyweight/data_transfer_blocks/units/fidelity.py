# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The FPU multiplier's fidelity decomposition.

The multiplier is narrower than a src datum, so a multiply runs several times
over different slices of the operands' mantissas and the partial products are
added up. ``MathFidelity`` chooses how many of those phases run.

Everything here describes the FPU rather than any operation, which is why it
sits beside :mod:`.fpu`. See `README.md`, "units/fidelity.py in full".
"""

import warnings
from typing import Optional, Tuple

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import MathFidelity, format_dict

from ..storage import truncate_mantissa
from .pack import UnmodelledHardwareWarning

#: How many phases a multiply runs at each fidelity.
FIDELITY_PHASES = {
    MathFidelity.LoFi: 1,
    MathFidelity.HiFi2: 2,
    MathFidelity.HiFi3: 3,
    MathFidelity.HiFi4: 4,
}

#: Which half of each operand a phase multiplies, in the order the FPU runs
#: them. Biggest term first, so stopping early drops the terms contributing
#: least rather than something important.
PHASE_OPERAND_HALVES = (
    ("hi", "hi"),  # A-high x B-high, the biggest term, and all LoFi runs
    ("lo", "hi"),  # A-low  x B-high
    ("hi", "lo"),  # A-high x B-low
    ("lo", "lo"),  # A-low  x B-low, the smallest, only HiFi4 pays for it
)


def fidelity_phases(math_fidelity: MathFidelity) -> range:
    """The phase numbers to loop over for a multiply at `math_fidelity`.

    **Every phase multiplies the original srcA and srcB**, not what the phase
    before it produced. Chaining them zeroes every phase after the first, and
    the answer still looks plausible.
    """
    return range(FIDELITY_PHASES[math_fidelity])


def warn_unmodelled_split(op_name: str, arch_name: str, stacklevel: int = 2) -> None:
    """Warn that this architecture's multiply is the exact product.

    "HiFi4 included" is not a slip: Wormhole's and Blackhole's per-phase masks
    never reach SrcA's least significant bit, so no fidelity there is exact.
    """
    warnings.warn(
        f"{arch_name} computes {op_name} as an exact product: this "
        f"architecture's per-phase mantissa split is not modelled, so "
        f"math_fidelity is ignored and the result is more accurate than the "
        f"device at every fidelity, HiFi4 included. Treat it as a reference, "
        f"not as an exact golden.",
        UnmodelledHardwareWarning,
        stacklevel=stacklevel,
    )


def split_mantissa(
    values: torch.Tensor, keep_bits: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split into (high, low) at `keep_bits` explicit mantissa bits.

    The low half is ``value - high`` rather than masked bits read back as a
    float: the implicit leading 1 belongs to the high half, so a masked
    mantissa is not the remainder.
    """
    high = truncate_mantissa(values, keep_bits)
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

    Carry the raw value on phase 0 and zero it on every later phase, so the
    infinity reaches the result exactly once. See `README.md`, "Infinities and
    NaNs", for why zeroing the low half alone is not enough.
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


def ieee_exponent(values: torch.Tensor) -> torch.Tensor:
    """The exponent an IEEE float stores for each value.

    ``torch.frexp`` returns a significand in [0.5, 1), so its exponent is one
    above the field the FP lane carries.
    """
    _, exponent = torch.frexp(values.to(torch.float32))
    return exponent - 1


def min_normal_exponent(dest_format: DataFormat) -> Optional[int]:
    """Lowest exponent the Dest format holds as a normal, or None if integer.

    Read with the same :func:`ieee_exponent` the flush rule compares against,
    so the two cannot drift apart.
    """
    dtype = format_dict[dest_format]
    if not dtype.is_floating_point:
        return None
    return int(ieee_exponent(torch.tensor(torch.finfo(dtype).smallest_normal)))


def non_finite_lanes(srcA: torch.Tensor, srcB: torch.Tensor) -> torch.Tensor:
    """Lanes where either operand is an infinity or a NaN.

    The split and flush rules both assume a finite operand, and ``frexp``
    reports exponent 0 for a non-finite value, which reads as a magnitude
    near 2^-1.
    """
    return ~(torch.isfinite(srcA.float()) & torch.isfinite(srcB.float()))


def flush_pre_carry_denormals(
    product: torch.Tensor,
    srcA: torch.Tensor,
    srcB: torch.Tensor,
    min_exponent: Optional[int],
) -> torch.Tensor:
    """Zero a product whose exponent lands below Dest's lowest normal.

    The lane adds the two stored src exponents and flushes the whole term
    before the mantissa product's carry, so the rule is one binade coarser than
    testing the finished magnitude. Use the **datum's** exponent, not a split
    half's. See `README.md`, "flush_pre_carry_denormals, the pre-carry rule".
    """
    if min_exponent is None:
        return product
    # A non-finite operand stores an all-ones exponent field, so its lane can
    # never land below the floor. Without this guard frexp's exponent-0 reads
    # as ~2^-1 and Inf * b flushes to zero for any small |b|.
    pre_carry = ieee_exponent(srcA) + ieee_exponent(srcB)
    dead = (pre_carry < min_exponent) & ~non_finite_lanes(srcA, srcB)
    return torch.where(dead, torch.zeros_like(product), product)
