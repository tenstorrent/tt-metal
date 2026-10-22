# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Element-wise binary operations."""

from functools import partial
from typing import Optional

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import MathFidelity, MathOperation

from ..chain import Chain, Registers
from ..golden import Golden, OpConfig


class EltwiseBinaryGolden(Golden):
    """L1 -> SrcA, L1 -> SrcB -> op -> Dest -> L1.

    Add and subtract are exact: the FPU splits only *multiplies* into fidelity
    phases. A multiply runs one accumulate step per phase, so the chain for
    HiFi4 has four of them.
    """

    op_name = "eltwise"

    def __init__(
        self,
        operation: MathOperation = MathOperation.Elwadd,
        math_fidelity: MathFidelity = MathFidelity.HiFi4,
        arch=None,
        blocks=None,
    ):
        super().__init__(arch, blocks)
        self.operation = operation
        self.math_fidelity = math_fidelity
        self.op_name = f"eltwise:{operation.name.lower()}"

    # ------------------------------------------------------------------

    def models_fidelity(self, cfg: OpConfig) -> bool:
        """Whether this op runs the multiply as accumulated partial products.

        Two questions, each answered where it belongs. Whether the operation is
        a multiply at all is this op's: add and subtract are exact, because the
        FPU splits only multiplies into phases. Whether a multiply on *these*
        datums decomposes is the FPU's, and it needs the input format to say
        so -- the decomposition slices a mantissa, which an integer datum does
        not have.
        """
        return self.operation is MathOperation.Elwmul and self.fpu.models_fidelity(
            cfg.in_formats[0]
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
        if self.models_fidelity(cfg):
            for phase in self.fpu.phases(self.math_fidelity):
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
                self.fpu.warn_split_unmodelled(self.op_name)
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

        The window each operand contributes and the flush that follows are the
        FPU's; what this adds is that each product here *is* a lane result
        written to Dest, which is why the flush applies at all -- a matmul,
        whose products never become lane results, passes ``None`` instead.
        """
        a, b = self.fpu.operand_halves(regs["srcA"], regs["srcB"], phase)
        return self.fpu.flush_pre_carry(a * b, regs["srcA"], regs["srcB"], dest_format)

    def apply(self, regs: Registers) -> torch.Tensor:
        """The whole operation in one FPU step, where it does not split.

        Add and subtract always come here; a multiply only where the
        architecture's split is not modelled, in which case the FPU returns the
        exact product at every fidelity -- LoFi included, since LoFi is one
        phase and not zero. Where the split *is* modelled, as on Quasar, a
        multiply goes through :meth:`partial_product` and never arrives here.
        """
        return self.fpu.elementwise(self.operation, regs["srcA"], regs["srcB"])
