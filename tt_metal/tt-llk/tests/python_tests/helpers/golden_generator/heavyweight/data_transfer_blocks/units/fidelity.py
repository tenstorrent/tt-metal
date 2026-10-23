# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The FPU multiplier's fidelity decomposition.

The multiplier is narrower than a src-register datum, so a multiply is run
several times over different slices of the operands' mantissas and the partial
products are accumulated. ``MathFidelity`` chooses how many of those passes
run.

Every rule here describes the **FPU**: how many phases a fidelity runs, which
mantissa window each phase takes, how wide the multiplier is, and when a
product is flushed before its carry. That is why it sits beside
:mod:`.fpu` rather than with the operations -- an operation selects *which*
FPU instruction runs, and the FPU decides how it behaves. Both the element-wise
multiply and the matmul decompose the same way; the only difference is whether
the partial product is element-wise or a matrix one, which is the operation's
business.
"""

import warnings
from typing import Optional, Tuple

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import MathFidelity, format_dict

from ..storage import truncate_mantissa
from .pack import UnmodelledHardwareWarning

#: How many phases a multiply runs at each fidelity.
#:
#: The multiplier is narrower than a src datum, so it cannot take two whole
#: mantissas at once. It cuts each operand into a high half and a low half,
#: multiplies the halves against each other one pair at a time, and adds the
#: four partial products up. Each of those is a phase, and fidelity is simply
#: how many of them you pay for -- LoFi runs one and drops the rest, HiFi4
#: runs all four.
FIDELITY_PHASES = {
    MathFidelity.LoFi: 1,
    MathFidelity.HiFi2: 2,
    MathFidelity.HiFi3: 3,
    MathFidelity.HiFi4: 4,
}

#: Which half of each operand a phase multiplies, in the order the FPU runs
#: them. Biggest term first, which is what makes stopping early cheap rather
#: than catastrophic: a reduced fidelity drops the terms that were
#: contributing least.
PHASE_OPERAND_HALVES = (
    ("hi", "hi"),  # A-high x B-high -- the biggest term, and all LoFi runs
    ("lo", "hi"),  # A-low  x B-high
    ("hi", "lo"),  # A-high x B-low
    ("lo", "lo"),  # A-low  x B-low  -- the smallest, only HiFi4 pays for it
)


def fidelity_phases(math_fidelity: MathFidelity) -> range:
    """The phase numbers to loop over for a multiply at `math_fidelity`.

    **Every phase multiplies the original srcA and srcB.** The tempting mistake
    is to hand each phase what the one before it produced, as though the halves
    were a remainder being whittled down. They are not: every phase cuts its
    halves out of the full operand again. Chain them and each phase after the
    first comes out zero, and that does not read as a bug, because the answer
    still looks reasonable.
    """
    return range(FIDELITY_PHASES[math_fidelity])


def warn_unmodelled_split(op_name: str, arch_name: str, stacklevel: int = 2) -> None:
    """
    "HiFi4 included" in the message below is not a slip. Wormhole's and
    Blackhole's per-phase masks never reach SrcA's least significant bit, so no
    fidelity there is the exact product.

    `stacklevel` counts from here, and the default 2 blames the immediate
    caller. It cannot be made to blame the calling test instead: ``build_chain``
    is reached from ``run`` and from ``_run_blocked``, which sit at different
    depths.
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

    The low half is taken as ``value - high`` rather than by masking bits in
    place: the implicit leading 1 belongs to the high half, so a masked
    mantissa re-read as a float is not the remainder. Subtracting is exact and
    needs no implicit-bit bookkeeping.
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


def ieee_exponent(values: torch.Tensor) -> torch.Tensor:
    """The exponent an IEEE float stores for each value.

    ``torch.frexp`` returns a significand in [0.5, 1), so its exponent is one
    above the one the FP lane's exponent field carries.
    """
    _, exponent = torch.frexp(values.to(torch.float32))
    return exponent - 1


def min_normal_exponent(dest_format: DataFormat) -> Optional[int]:
    """Lowest exponent the Dest format holds as a normal, or None if integer.

    Read with the same :func:`ieee_exponent` the flush rule compares against,
    so the two cannot drift apart. ``smallest_normal`` is an exact power of two
    in every float format here, and ``frexp`` is exact, so this is the
    format's true exponent floor and not a rounded log.
    """
    dtype = format_dict[dest_format]
    if not dtype.is_floating_point:
        return None
    return int(ieee_exponent(torch.tensor(torch.finfo(dtype).smallest_normal)))


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
    enter the sum-of-products network at the combined width of both
    significands and never become lane results, so only the accumulated sum is
    subject to the rule.
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
