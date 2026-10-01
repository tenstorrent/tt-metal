# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Element-wise binary operations."""

from functools import partial
from typing import Optional, Tuple

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import MathFidelity, MathOperation

from .chain import Chain, Registers
from .fidelity import (
    FIDELITY_PHASES,
    flush_pre_carry_denormals,
    min_normal_exponent,
    operand_halves,
    split_mantissa,
)
from .golden import Golden, OpConfig


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

    #: Kept as a class member because the fidelity split is documented per
    #: operation; the implementation is shared with matmul.
    split_mantissa = staticmethod(split_mantissa)

    def partial_product(
        self, regs: Registers, *, phase: int, dest_format: Optional[DataFormat] = None
    ) -> torch.Tensor:
        """One fidelity phase: the partial product of the chosen operand halves.

        Each product is a lane result written to Dest, so the FPU's pre-carry
        denormal flush applies to it -- see
        :func:`.fidelity.flush_pre_carry_denormals` for the rule and why a
        matmul is exempt. Pass `dest_format` to model it; ``None`` skips it.
        """
        a, b = operand_halves(regs["srcA"], regs["srcB"], self.MANTISSA_SPLIT, phase)
        product = a * b
        if dest_format is None:
            return product
        return flush_pre_carry_denormals(
            product, regs["srcA"], regs["srcB"], min_normal_exponent(dest_format)
        )

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
