# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Matmul — the MVMUL family."""

from functools import partial
from typing import Dict, Optional, Tuple

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import MathFidelity
from helpers.tilize_untilize import tilize_block, untilize_block

from ..data_transfer_blocks.l1_codec import datums_per_tile
from .chain import Chain, Registers
from .fidelity import (
    FIDELITY_PHASES,
    operand_halves,
    resolve_non_finite,
    warn_unmodelled_split,
)
from .golden import Golden, OpConfig

#: Datums along one edge of a tile.
TILE_DIM = 32


def self_shape(geometry: dict) -> dict:
    """Tile geometry in the form the tilize/untilize helpers take."""
    return dict(
        dimensions=[TILE_DIM, TILE_DIM],
        tile_dimensions=[TILE_DIM, TILE_DIM],
        **geometry,
    )


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
    #: produces ``arg0 @ arg1``, which holds on every architecture -- what the
    #: caller passes does not change, only which register each operand rides.
    #:
    #: Every architecture routes the first operand to SrcB and the second to
    #: SrcA, so this is the default rather than a Quasar specialisation:
    #: ``llk_math_matmul.h`` states "D = in0 * in1, where in0 is loaded to SrcB
    #: and in1 to SrcA" on Wormhole and Blackhole alike, matching Quasar's
    #: ``_llk_unpack_matmul_init_``.
    OPERAND_REGISTERS: Tuple[str, str] = ("srcB", "srcA")

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
                        partial(
                            self.partial_product,
                            phase=phase,
                            geometry=cfg.geometry,
                        ),
                        reads=("srcA", "srcB"),
                        accumulate=phase > 0,
                    )
                )
        else:
            warn_unmodelled_split(self.op_name, type(self).__name__)
            chain.then(
                self.src_to_dest(
                    cfg,
                    partial(self.apply, geometry=cfg.geometry),
                    reads=("srcA", "srcB"),
                )
            )
        return chain.then(self.dest_to_l1(cfg, into="out"))

    # ------------------------------------------------------------------

    @staticmethod
    def _as_tile(values: torch.Tensor, geometry: Dict) -> torch.Tensor:
        """A src register's contents as the logical matrix the FPU multiplies.

        A src register holds its tile **face-ordered**, the way the unpacker
        wrote it, so a plain ``reshape(32, 32)`` is not the logical matrix: its
        row 0 is logical row 0 columns 0-15 followed by logical row 1 columns
        0-15. The leading entries of each row line up, which is exactly why the
        wrong answer reads as a plausible one.
        """
        expected = datums_per_tile(**geometry)
        if values.numel() != expected:
            raise ValueError(
                f"matmul handles one {TILE_DIM}x{TILE_DIM} tile at a time; got "
                f"{values.numel()} datums where {expected} were expected. A "
                f"multi-tile matmul is a block matmul over the inner dimension, "
                f"which this golden does not model yet."
            )
        return untilize_block(
            values.float(), stimuli_format=DataFormat.Float32, **self_shape(geometry)
        ).reshape(TILE_DIM, TILE_DIM)

    @staticmethod
    def _from_tile(matrix: torch.Tensor, geometry: Dict) -> torch.Tensor:
        """The logical result back in face order, which is how Dest holds it."""
        return (
            tilize_block(
                matrix.reshape(-1).float(),
                stimuli_format=DataFormat.Float32,
                **self_shape(geometry),
            )
            .flatten()
            .float()
        )

    def _product(
        self, srcA: torch.Tensor, srcB: torch.Tensor, geometry: Dict
    ) -> torch.Tensor:
        """The matrix product, in this architecture's operand order.

        ``Dest = SrcB @ SrcA``, which is not a transpose of the caller's
        arguments: :attr:`OPERAND_REGISTERS` puts the first operand in SrcB, so
        the two together give ``arg0 @ arg1``. Both halves are load-bearing and
        have to agree -- the routing without this order, or this order without
        the routing, transposes the result into something wrong everywhere but
        still plausible-looking, which is how it was last caught.

        The one place operand order lives, so a subclass that diverges does not
        have to override both the exact and the per-phase paths. Summed in
        float64: the hardware's accumulator is wider than its operands, so the
        model must not introduce a float32 accumulation error the hardware does
        not have.
        """
        product = (
            self._as_tile(srcB, geometry).double()
            @ self._as_tile(srcA, geometry).double()
        )
        return self._from_tile(product, geometry)

    def apply(self, regs: Registers, *, geometry: Dict) -> torch.Tensor:
        """The exact product, for architectures whose split is not modelled."""
        return self._product(regs["srcA"], regs["srcB"], geometry)

    def partial_product(
        self, regs: Registers, *, phase: int, geometry: Dict
    ) -> torch.Tensor:
        """One fidelity phase: the matrix product of the chosen operand halves.

        Non-finite operands go through :func:`.fidelity.resolve_non_finite`,
        which carries them on the first phase and zeroes them on the rest.
        """
        a, b = operand_halves(regs["srcA"], regs["srcB"], self.MANTISSA_SPLIT, phase)
        a, b = resolve_non_finite(a, b, regs["srcA"], regs["srcB"], phase)
        return self._product(a, b, geometry)
