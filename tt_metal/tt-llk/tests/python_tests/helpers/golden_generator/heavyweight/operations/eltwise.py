# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Element-wise binary operations."""

import math
from functools import partial
from typing import Optional, Tuple

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import MathFidelity, MathOperation, format_dict

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
        chain = Chain()
        for tile in range(cfg.tiles_per_output):
            chain.then(
                self.l1_to_srcA(cfg, source=self.source(0, tile)),
                self.l1_to_srcB(cfg, source=self.source(1, tile)),
            )
            self._math_steps(chain, cfg, accumulate=tile > 0)
        return chain.then(self.dest_to_l1(cfg, into="out"))

    def _math_steps(self, chain: Chain, cfg: OpConfig, *, accumulate: bool) -> Chain:
        """The maths for one input tile, appended to `chain`.

        Accumulates into Dest when this tile is not the first of its block, or
        when a fidelity phase has already written Dest.
        """
        if self.models_fidelity:
            # One accumulate per phase, each reading the *original* srcA/srcB.
            # Feeding a phase the previous phase's masked operands zeroes every
            # phase after the first, which silently turns fidelity into a no-op.
            for phase in range(FIDELITY_PHASES[self.math_fidelity]):
                chain.then(
                    self.src_to_dest(
                        cfg,
                        partial(
                            self.partial_product,
                            phase=phase,
                            dest_format=cfg.dest_format,
                        ),
                        reads=("srcA", "srcB"),
                        accumulate=accumulate or phase > 0,
                    )
                )
        else:
            chain.then(
                self.src_to_dest(
                    cfg, self.apply, reads=("srcA", "srcB"), accumulate=accumulate
                )
            )
        return chain

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

    @staticmethod
    def _min_normal_exponent(dest_format: DataFormat) -> Optional[int]:
        """Lowest exponent the Dest format holds as a normal, or None if integer."""
        dtype = format_dict[dest_format]
        if not dtype.is_floating_point:
            return None
        return int(math.log2(torch.finfo(dtype).smallest_normal))

    def partial_product(
        self, regs: Registers, *, phase: int, dest_format: Optional[DataFormat] = None
    ) -> torch.Tensor:
        """One fidelity phase: the partial product of the chosen operand halves.

        The lane forms the result's exponent by adding the two *stored src*
        exponents and rebiasing into Dest's range, then flushes the whole term
        — mantissa included — when that lands below Dest's lowest normal. That
        decision is taken before the mantissa product's carry into the next
        binade, so the two rules differ by exactly one binade: when the
        mantissas multiply to 2.0 or more the finished value is a normal of
        Dest's format while the pre-carry exponent is not, and the hardware
        still returns zero. Flushing on the finished magnitude instead keeps
        those, and they are only visible through an MX output, where one zeroed
        element shifts the block scale far enough to fail the comparison.

        The exponent is a property of the src *datum*, one field shared by every
        phase — a phase selects a mantissa window, not an exponent — so the sum
        is the same on all four, and taking the exponent of a split half instead
        flushes almost everything.
        """
        a_bits, b_bits = self.MANTISSA_SPLIT
        a_half, b_half = PHASE_OPERAND_HALVES[phase]
        a_hi, a_lo = self.split_mantissa(regs["srcA"], a_bits)
        b_hi, b_lo = self.split_mantissa(regs["srcB"], b_bits)
        product = (a_hi if a_half == "hi" else a_lo) * (
            b_hi if b_half == "hi" else b_lo
        )
        if dest_format is None:
            return product
        min_exponent = self._min_normal_exponent(dest_format)
        if min_exponent is None:
            return product
        _, exp_a = torch.frexp(regs["srcA"].float())
        _, exp_b = torch.frexp(regs["srcB"].float())
        # frexp returns a mantissa in [0.5, 1), so its exponent is one above the
        # IEEE one that the lane's exponent field carries.
        pre_carry = (exp_a - 1) + (exp_b - 1)
        return torch.where(pre_carry < min_exponent, torch.zeros_like(product), product)

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
