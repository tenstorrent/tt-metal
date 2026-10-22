# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The FPU: it computes, and it writes Dest.

Unlike the unpacker and the packer, which only move and convert, this unit also
does arithmetic -- so it owns the arithmetic. Everything about *how* a multiply
behaves on this device lives here or in :mod:`.fidelity`: the multiplier's
width, the phase decomposition, the pre-carry denormal flush, the rounding that
happens on every Dest write, and what the Dest-to-src trip costs.

What stays with the operation is *which* instruction runs and in what order --
an Elwadd rather than an Elwmul, how many tiles fold into one Dest, which
operand rides which register. The operation sequences; the FPU performs.
"""

from typing import Optional, Tuple

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import MathFidelity, MathOperation

from ..l1_support import L1Support
from ..storage import dest_to_src_storage, to_dest_storage
from .fidelity import (
    fidelity_phases,
    flush_pre_carry_denormals,
    min_normal_exponent,
    operand_halves,
    resolve_non_finite,
    warn_unmodelled_split,
)

#: The element-wise operations the FPU performs on two src registers.
_ELEMENTWISE = {
    MathOperation.Elwadd: lambda a, b: a + b,
    MathOperation.Elwsub: lambda a, b: a - b,
    MathOperation.Elwmul: lambda a, b: a * b,
}


class Fpu:
    """What the FPU computes, and what its writes cost."""

    def __init__(
        self, l1: L1Support, *, mantissa_split: Optional[Tuple[int, int]] = None
    ) -> None:
        #: Explicit mantissa bits the multiplier takes from each operand per
        #: fidelity phase, as (srcA, srcB). ``None`` means this architecture's
        #: multiplier is not modelled, so a multiply is computed exactly and
        #: ``math_fidelity`` is ignored -- which warns rather than passing
        #: quietly.
        #:
        #: Quasar multiplies 7x7 bits per phase, so a src datum's 10 explicit
        #: bits split 7 high / 3 low on **both** operands. That symmetry is a
        #: Quasar property: Wormhole and Blackhole split 4/6, and their SrcA
        #: additionally drops its least significant bit at every fidelity, so
        #: even HiFi4 is not the exact product there.
        self.mantissa_split = mantissa_split

        self.l1 = l1

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def elementwise(
        self, operation: MathOperation, srcA: torch.Tensor, srcB: torch.Tensor
    ) -> torch.Tensor:
        """One element-wise FPU operation on two src registers.

        Computed in a type exact for the operands it was given. The unpacker
        already landed them in the register's own dtype, so an integer input
        really is an integer tensor here, and the FPU's integer path really
        does integer arithmetic. Doing it in float32 would be exact for Int8 but
        not for Int16 -- ``32767 * 32767`` loses its low bit past float32's
        2**24 -- and the whole point of the model is that the type is the
        hardware's, not the convenient one.

        ``int64`` rather than the operands' own width because torch integers
        **wrap**: ``int8`` 100 * 2 is -56. It is exact for every product
        reachable through SrcA/SrcB, since Int32 never lands in one.
        """
        try:
            compute = _ELEMENTWISE[operation]
        except KeyError:
            raise ValueError(
                f"{operation} is not an element-wise FPU operation; this unit "
                f"performs {sorted(o.name for o in _ELEMENTWISE)}."
            ) from None
        wide = torch.float32 if srcA.dtype.is_floating_point else torch.int64
        return compute(srcA.to(wide), srcB.to(wide))

    def sum_of_products(
        self, srcB_rows: torch.Tensor, srcA_cols: torch.Tensor
    ) -> torch.Tensor:
        """One matrix multiply, accumulated wider than its operands.

        The FPU's sum-of-products network takes products in at the combined
        width of both significands and sums them in wide fixed point with guard
        bits, so within a single instruction there is no per-product rounding
        to model. ``float64`` rather than ``float32`` for exactly that reason:
        a float32 accumulation would introduce an error the hardware does not
        have.

        Exact for the integer path too, where the operands reach 2**15 and a
        32-term sum of their products stays far inside float64's 2**53. The
        rounding the hardware *does* take, between one instruction and the
        next, belongs to the Dest write and is modelled by whoever calls this
        once per pass.

        Operands in register order, which is not the caller's argument order:
        ``Dest = SrcB * SrcA``. The matrix each register contributes -- which
        rows, which columns, how a face-ordered register becomes a matrix at
        all -- is the operation's business, because it is the operation that
        knows which slice of the inner dimension this instruction covers.
        """
        return srcB_rows.double() @ srcA_cols.double()

    # ------------------------------------------------------------------
    # The multiplier's fidelity decomposition
    # ------------------------------------------------------------------

    def models_fidelity(self, in_format: DataFormat) -> bool:
        """Whether a multiply on `in_format` datums decomposes into phases here.

        Two conditions. The architecture's split has to be modelled at all, and
        the operands have to be **floats**: the decomposition slices a mantissa,
        and an integer datum has none to slice. Applying it anyway truncates the
        operand's low bits -- an Int16 ``3 * 10922`` came back 32640 instead of
        32766 -- and the FPU's integer path does not decompose into phases.

        Takes the format rather than a tensor because an operation asks this
        while it is *building* its pipeline, before any register holds a value.
        """
        return self.mantissa_split is not None and not in_format.is_integer()

    def phases(self, math_fidelity: MathFidelity) -> range:
        """The phase indices a multiply at `math_fidelity` runs."""
        return fidelity_phases(math_fidelity)

    def warn_split_unmodelled(self, op_name: str, stacklevel: int = 3) -> None:
        """Say so when a multiply here falls back to the exact product.

        `stacklevel` is counted from the warning itself, so the default 3
        blames whoever called *this* method -- the chain builder that selected
        the fallback -- rather than this method or the helper behind it. A
        caller that adds a frame of its own raises it.
        """
        warn_unmodelled_split(op_name, self.l1.name, stacklevel=stacklevel)

    def operand_halves(
        self, srcA: torch.Tensor, srcB: torch.Tensor, phase: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The slice of each operand this architecture's multiplier takes.

        Non-finite operands are carried on the first phase and zeroed on the
        rest, since a mantissa window of an Inf or a NaN is meaningless.
        """
        a, b = operand_halves(srcA, srcB, self.mantissa_split, phase)
        return resolve_non_finite(a, b, srcA, srcB, phase)

    def flush_pre_carry(
        self,
        product: torch.Tensor,
        srcA: torch.Tensor,
        srcB: torch.Tensor,
        dest_format: Optional[DataFormat],
    ) -> torch.Tensor:
        """Zero a lane product whose exponent lands below Dest's lowest normal.

        Applies where each product *is* a lane result written to Dest, i.e. an
        element-wise multiply. A matmul is exempt: there the products enter the
        sum-of-products network at full width and never become lane results, so
        only the accumulated sum is subject to the rule. Pass ``None`` to skip
        it.
        """
        if dest_format is None:
            return product
        return flush_pre_carry_denormals(
            product, srcA, srcB, min_normal_exponent(dest_format)
        )

    # ------------------------------------------------------------------
    # Dest
    # ------------------------------------------------------------------

    def src_to_dest(
        self,
        values: torch.Tensor,
        dest_format: DataFormat = DataFormat.Float32,
        current: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Write a math result into Dest.

        A Dest slot holds only what `dest_format` can represent, and the value
        is rounded on the way *in*, not on the way out to L1.

        `current` is Dest's existing contents, which a multi-pass op passes to
        accumulate rather than replace -- the FPU has an accumulate-enable bit,
        not a second instruction. Because the rounding happens on this write, it
        happens on every pass; accumulating at full precision and rounding once
        at the end would model an accumulator the hardware does not have.
        """
        self.l1.check_dest_format(dest_format)
        if current is not None:
            values = self._accumulate(values, current, dest_format)
        return to_dest_storage(values, dest_format)

    @staticmethod
    def _accumulate(
        values: torch.Tensor, current: torch.Tensor, dest_format: DataFormat
    ) -> torch.Tensor:
        """Add a new result to what Dest holds, in a type exact for both.

        Three cases, because the operands are not always the same kind:

        * **Both integer** -- an integer op, whose operands the unpacker landed
          as integers and whose result stayed integer. Add as integers, which
          is what the device does. ``int64`` because the narrow types *wrap*:
          ``int8`` 100 + 100 is -56, and the saturation below would then clamp
          the wrong number.
        * **A float result into an integer Dest** -- reachable from a matmul on
          integer inputs, and from a caller naming an integer ``dest_format``
          explicitly. ``float64``, because float32 holds integers exactly only
          to 2**24 and an Int32 Dest reaches 2**31; a value past that would
          lose its low bits here, before the narrow, so ``16777217 + 1`` would
          come back 16777216.
        * **A float Dest** -- ``float32``, the width the rest of the float path
          uses.

        The narrowing and the saturation happen once, afterwards, in
        :func:`..storage.to_dest_storage`.
        """
        both_integer = (
            not values.dtype.is_floating_point and not current.dtype.is_floating_point
        )
        if both_integer:
            wide = torch.int64
        elif dest_format.is_integer():
            wide = torch.float64
        else:
            wide = torch.float32
        return current.to(wide) + values.to(wide)

    def dest_to_src(
        self,
        dest_values: torch.Tensor,
        dest_format: DataFormat = DataFormat.Float32,
        src_format: DataFormat = DataFormat.Float16_b,
    ) -> torch.Tensor:
        """Move Dest back into a src register, for an op that reuses its result.

        Dest is wider than a src register, so the round trip is lossy --
        modelling that loss where the hardware takes it is the point of having
        this block. SrcA and SrcB share the datum layout, so one method serves
        both. See :func:`..storage.dest_to_src_storage` for what each Dest
        format costs.
        """
        return dest_to_src_storage(dest_values, dest_format, src_format)
