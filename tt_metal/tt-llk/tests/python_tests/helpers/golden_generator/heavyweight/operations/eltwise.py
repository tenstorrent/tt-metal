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
    fidelity_phases,
    flush_pre_carry_denormals,
    min_normal_exponent,
    operand_halves,
    resolve_non_finite,
    warn_unmodelled_split,
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
            for phase in fidelity_phases(self.math_fidelity):
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
            if self.operation is MathOperation.Elwmul:
                warn_unmodelled_split(self.op_name, type(self).__name__)
            chain.then(
                self.src_to_dest(
                    cfg, self.apply, reads=("srcA", "srcB"), accumulate=accumulate
                )
            )
        return chain

    # ------------------------------------------------------------------

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
        a, b = resolve_non_finite(a, b, regs["srcA"], regs["srcB"], phase)
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
            # The exact product, reached only where MANTISSA_SPLIT is None --
            # i.e. on an architecture whose split is not modelled, at every
            # fidelity. Where the split *is* modelled, as on Quasar, the
            # multiply always goes through partial_product and never arrives
            # here, LoFi included: LoFi is one phase, not zero.
            return a * b
        raise ValueError(f"{self.operation} is not an element-wise binary op")
