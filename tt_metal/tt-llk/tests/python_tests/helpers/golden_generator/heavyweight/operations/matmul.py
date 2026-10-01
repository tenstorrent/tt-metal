# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Matmul — the MVMUL family."""

from functools import partial
from typing import Optional, Tuple

import torch
from helpers.llk_params import MathFidelity

from .chain import Chain, Registers
from .fidelity import FIDELITY_PHASES, operand_halves
from .golden import Golden, OpConfig

#: Datums along one edge of a tile.
TILE_DIM = 32


class MatmulGolden(Golden):
    """L1 -> SrcA, L1 -> SrcB -> product -> Dest -> L1.

    A matmul decomposes into fidelity phases the same way an element-wise
    multiply does -- the multiplier is narrower than a src datum, so each pass
    multiplies a different slice of the operands' mantissas and the partial
    products accumulate in Dest. The only difference from the element-wise case
    is that the partial product is a matrix product.

    **Accumulation is exact within a pass, rounded once at the Dest write.**
    Products enter the FPU's sum-of-products network at full significand width
    (``SOP_IN_MAN_PREC = (MAN_PREC_A+1) + (MAN_PREC_B+1)``) and are summed in
    wide fixed point with guard bits (``SOP_PREC``, ``MAN_PREC_ACC = 23``), so
    there is no per-MAC rounding to model and no per-product denormal flush:
    an individual product never becomes an FP lane result, so RES_A2 does not
    reach it. Only the accumulated sum is written to Dest, where
    ``src_to_dest`` applies Dest precision and the no-denormal rule.
    """

    op_name = "matmul"

    #: Explicit mantissa bits the multiplier takes from each operand per phase,
    #: as (srcA, srcB). ``None`` means this architecture's split is not modelled
    #: and the product is computed exactly, at Dest precision.
    MANTISSA_SPLIT: Optional[Tuple[int, int]] = None

    #: Which src register each stimulus lands in, in argument order. Set so that
    #: ``run([arg0, arg1])`` mirrors ``_llk_unpack_matmul_init_(arg0, arg1)`` and
    #: produces ``arg0 @ arg1`` on every architecture -- the per-architecture
    #: difference is which register each operand travels through, not what the
    #: caller has to pass.
    OPERAND_REGISTERS: Tuple[str, str] = ("srcA", "srcB")

    def __init__(
        self,
        math_fidelity: MathFidelity = MathFidelity.HiFi4,
        blocks=None,
    ):
        super().__init__(blocks)
        self.math_fidelity = math_fidelity

    @property
    def models_fidelity(self) -> bool:
        """Whether this runs the product as accumulated partial products."""
        return self.MANTISSA_SPLIT is not None

    def build_chain(self, cfg: OpConfig) -> Chain:
        unpack = {"srcA": self.l1_to_srcA, "srcB": self.l1_to_srcB}
        chain = Chain(
            [
                unpack[register](
                    cfg, source=self.source(operand), into=register, index=operand
                )
                for operand, register in enumerate(self.OPERAND_REGISTERS)
            ]
        )
        if self.models_fidelity:
            # One accumulate per phase, each reading the *original* srcA/srcB.
            # Feeding a phase the previous phase's sliced operands zeroes every
            # phase after the first and turns fidelity into a silent no-op.
            for phase in range(FIDELITY_PHASES[self.math_fidelity]):
                chain.then(
                    self.src_to_dest(
                        cfg,
                        partial(self.partial_product, phase=phase),
                        reads=("srcA", "srcB"),
                        accumulate=phase > 0,
                    )
                )
        else:
            chain.then(self.src_to_dest(cfg, self.apply, reads=("srcA", "srcB")))
        return chain.then(self.dest_to_l1(cfg, into="out"))

    # ------------------------------------------------------------------

    @staticmethod
    def _as_tile(values: torch.Tensor) -> torch.Tensor:
        return values.float().reshape(TILE_DIM, TILE_DIM)

    def _product(self, srcA: torch.Tensor, srcB: torch.Tensor) -> torch.Tensor:
        """The matrix product, in this architecture's operand order.

        The one place operand order lives, so a subclass that inverts it does
        not have to override both the exact and the per-phase paths. Summed in
        float64 and returned flat: the hardware's accumulator is wider than its
        operands, so the model must not introduce a float32 accumulation error
        the hardware does not have.
        """
        return (
            (self._as_tile(srcA).double() @ self._as_tile(srcB).double())
            .reshape(-1)
            .float()
        )

    def apply(self, regs: Registers) -> torch.Tensor:
        """The exact product, for architectures whose split is not modelled."""
        return self._product(regs["srcA"], regs["srcB"])

    def partial_product(self, regs: Registers, *, phase: int) -> torch.Tensor:
        """One fidelity phase: the matrix product of the chosen operand halves."""
        a, b = operand_halves(regs["srcA"], regs["srcB"], self.MANTISSA_SPLIT, phase)
        return self._product(a, b)
