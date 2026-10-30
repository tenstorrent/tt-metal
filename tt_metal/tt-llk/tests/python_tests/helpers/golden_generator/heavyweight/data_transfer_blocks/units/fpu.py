# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The FPU: it computes, and it writes Dest.

The unpacker and the packer only move and convert. This unit also does
arithmetic, so it owns the arithmetic: the multiplier's width, the phase
decomposition, the pre-carry flush, the rounding on every Dest write, and what
the trip back to a src register costs.

See `README.md`, "The FPU computes".
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
        self,
        l1: L1Support,
        *,
        mantissa_split: Optional[Tuple[int, int]] = None,
        dest_ties_away: bool = False,
    ) -> None:
        # Mantissa bits the multiplier takes from each operand per phase, as
        # (srcA, srcB). None means this architecture's multiplier is not
        # modelled: multiplies are exact and math_fidelity is ignored.
        self.mantissa_split = mantissa_split

        # Rounding mode for a 16-bit Dest write. False is the IEEE default,
        # ties to even. Blackhole rounds ties away from zero, which only shows
        # up where the dropped bits are exactly half an ULP.
        self.dest_ties_away = dest_ties_away

        self.l1 = l1

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def elementwise(
        self, operation: MathOperation, srcA: torch.Tensor, srcB: torch.Tensor
    ) -> torch.Tensor:
        """One element-wise FPU operation on two src registers.

        Computed in a type exact for the operands it was given, so an integer input
        really does integer arithmetic. ``int64`` rather than the operand's own
        width because torch integers wrap.
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

        The hardware's network takes products at the combined width of both
        significands and sums them with guard bits, so there is no per-product
        rounding to model inside one instruction. Rounding between instructions
        belongs to the Dest write.
        """
        wide = torch.float64 if srcB_rows.dtype.is_floating_point else torch.int64
        return srcB_rows.to(wide) @ srcA_cols.to(wide)

    # ------------------------------------------------------------------
    # The multiplier's fidelity decomposition
    # ------------------------------------------------------------------

    def models_fidelity(self, in_format: DataFormat) -> bool:
        """Whether a multiply on `in_format` decomposes into phases here.

        Needs the architecture's split to be modelled *and* the operands to be
        floats, since the decomposition slices a mantissa, which an integer has none
        of. Takes the format rather than a tensor because an operation asks while
        it is still building its pipeline.
        """
        return self.mantissa_split is not None and not in_format.is_integer()

    def phases(self, math_fidelity: MathFidelity) -> range:
        """The phase indices a multiply at `math_fidelity` runs."""
        return fidelity_phases(math_fidelity)

    def warn_split_unmodelled(self, op_name: str, stacklevel: int = 3) -> None:
        """Say so when a multiply here falls back to the exact product.

        Silent unless this architecture's multiplier is genuinely unmodelled. An
        integer multiply also skips the phase loop, but it is exact on the device
        too, so warning there would be wrong.

        `stacklevel` counts from ``warnings.warn``, and 3 blames whoever called this.
        """
        if self.mantissa_split is not None:
            return
        warn_unmodelled_split(op_name, self.l1.name, stacklevel=stacklevel)

    def operand_halves(
        self, srcA: torch.Tensor, srcB: torch.Tensor, phase: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The slice of each operand this architecture's multiplier takes.

        Non-finite operands are carried on the first phase and zeroed on the rest.
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

        For a product that *is* a lane result written to Dest. A matmul is exempt
        and passes ``None``: its products enter the sum-of-products network at
        full width and never become lane results.
        """
        if dest_format is None:
            return product
        return flush_pre_carry_denormals(
            product, srcA, srcB, min_normal_exponent(dest_format)
        )

    def partial_product(
        self,
        srcA: torch.Tensor,
        srcB: torch.Tensor,
        phase: int,
        dest_format: Optional[DataFormat] = None,
    ) -> torch.Tensor:
        """One fidelity phase of a multiply: window, multiply, flush.

        The whole phase, so the multiply stays in the unit that defines how this
        device multiplies. A matmul does not use this, because it needs the halves
        unmultiplied, to arrange into matrices and slice a K face first.
        """
        a, b = self.operand_halves(srcA, srcB, phase)
        return self.flush_pre_carry(
            self.elementwise(MathOperation.Elwmul, a, b), srcA, srcB, dest_format
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

        A Dest slot holds only what `dest_format` can represent, and the value is
        rounded on the way *in*, not on the way out to L1.

        `current` is Dest's existing contents, which a multi-pass op passes to
        accumulate rather than replace. Because the rounding happens on this
        write, it happens on every pass.
        """
        self.l1.check_dest_format(dest_format)
        if current is not None:
            values = self._accumulate(values, current)
        return to_dest_storage(values, dest_format, self.dest_ties_away)

    @staticmethod
    def _accumulate(values: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        """Add a new result to what Dest holds, in a type exact for both.

        ``int64`` for integers because the narrow types wrap, and ``float32`` for
        floats. The width follows the values, not `dest_format`. Narrowing and
        saturation happen afterwards in :func:`..storage.to_dest_storage`.
        """
        wide = torch.float32 if values.dtype.is_floating_point else torch.int64
        return current.to(wide) + values.to(wide)

    def dest_to_src(
        self,
        dest_values: torch.Tensor,
        dest_format: DataFormat = DataFormat.Float32,
        src_format: DataFormat = DataFormat.Float16_b,
    ) -> torch.Tensor:
        """Move Dest back into a src register, for an op that reuses its result.

        Dest is wider than a src register, so the round trip is lossy. SrcA and
        SrcB share the datum layout, so one method serves both. See
        :func:`..storage.dest_to_src_storage` for what each Dest format costs.
        """
        return dest_to_src_storage(dest_values, dest_format, src_format)
