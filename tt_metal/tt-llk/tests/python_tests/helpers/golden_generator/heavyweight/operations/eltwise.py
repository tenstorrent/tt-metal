# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Element-wise binary operations."""

from functools import partial
from typing import Optional, Tuple

import torch
from helpers.llk_params import MathFidelity, MathOperation

from .chain import Chain, Registers
from .golden import Golden, OpConfig

#: Phases each MathFidelity runs. The FPU decomposes a multiply into partial
#: products (AH_BH, AL_BH, AH_BL, AL_BL) accumulated across passes, and fidelity
#: chooses how many of them to run.
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


class EltwiseBinaryGolden(Golden):
    """L1 -> SrcA, L1 -> SrcB -> op -> Dest -> L1.

    Add and subtract are exact: the FPU splits only *multiplies* into fidelity
    phases. A multiply runs one accumulate step per phase, so the chain for
    HiFi4 has four of them.
    """

    op_name = "eltwise"

    #: Explicit mantissa bits the multiplier takes from each operand per phase,
    #: as (srcA, srcB). ``None`` means this architecture's split is not modelled
    #: and a multiply is computed exactly, at Dest precision.
    MANTISSA_SPLIT: Optional[Tuple[int, int]] = None

    def __init__(
        self,
        operation: MathOperation = MathOperation.Elwadd,
        math_fidelity: MathFidelity = MathFidelity.HiFi4,
        blocks=None,
    ):
        super().__init__(blocks)
        self.operation = operation
        self.math_fidelity = math_fidelity
        self.op_name = f"eltwise:{operation.name.lower()}"

    # ------------------------------------------------------------------

    @property
    def models_fidelity(self) -> bool:
        """Whether this op runs the multiply as accumulated partial products."""
        return (
            self.operation is MathOperation.Elwmul and self.MANTISSA_SPLIT is not None
        )

    def build_chain(self, cfg: OpConfig) -> Chain:
        chain = Chain(
            [
                self.l1_to_srcA(cfg, source="in0"),
                self.l1_to_srcB(cfg, source="in1"),
            ]
        )
        if self.models_fidelity:
            # One accumulate per phase, each reading the *original* srcA/srcB.
            # Feeding a phase the previous phase's masked operands zeroes every
            # phase after the first, which silently turns fidelity into a no-op.
            for phase in range(FIDELITY_PHASES[self.math_fidelity]):
                chain.then(
                    self.accumulate_into_dest(
                        partial(self.partial_product, phase=phase),
                        reads=("srcA", "srcB"),
                    )
                )
        else:
            chain.then(self.src_to_dest(self.apply, reads=("srcA", "srcB")))
        return chain.then(self.dest_to_l1(cfg, into="out"))

    # ------------------------------------------------------------------

    @staticmethod
    def split_mantissa(
        values: torch.Tensor, keep_bits: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split into (high, low) at `keep_bits` explicit mantissa bits.

        The low half is taken as ``value - high`` rather than by masking bits in
        place: the implicit leading 1 belongs to the high half, so a masked
        mantissa re-read as a float is not the remainder. Subtracting is exact
        and needs no implicit-bit bookkeeping.
        """
        raw = values.to(torch.float32).contiguous().view(torch.int32)
        high = (raw & ~((1 << (23 - keep_bits)) - 1)).view(torch.float32)
        return high, values.to(torch.float32) - high

    def partial_product(self, regs: Registers, *, phase: int) -> torch.Tensor:
        """One fidelity phase: the partial product of the chosen operand halves."""
        a_bits, b_bits = self.MANTISSA_SPLIT
        a_half, b_half = PHASE_OPERAND_HALVES[phase]
        a_hi, a_lo = self.split_mantissa(regs["srcA"], a_bits)
        b_hi, b_lo = self.split_mantissa(regs["srcB"], b_bits)
        return (a_hi if a_half == "hi" else a_lo) * (b_hi if b_half == "hi" else b_lo)

    def apply(self, regs: Registers) -> torch.Tensor:
        a, b = regs["srcA"].float(), regs["srcB"].float()
        if self.operation is MathOperation.Elwadd:
            return a + b
        if self.operation is MathOperation.Elwsub:
            return a - b
        if self.operation is MathOperation.Elwmul:
            return (
                a * b
            )  # Only done for LoFi. Checkout the high fidelity implementation above.
        raise ValueError(f"{self.operation} is not an element-wise binary op")
