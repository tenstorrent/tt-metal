# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Matmul — the MVMUL family."""

from functools import partial
from typing import Dict, Tuple

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import MathFidelity
from helpers.tile_constants import (
    DEFAULT_TILE_C_DIM,
    DEFAULT_TILE_R_DIM,
    MAX_FACE_R_DIM,
)
from helpers.tilize_untilize import tilize_block, untilize_block

from ...data_transfer_blocks.l1_codec import datums_per_tile
from ..chain import Chain, Registers
from ..golden import Golden, OpConfig

#: Datums along one edge of a tile. Aliased from the shared tile constants so
#: there is one definition of 32 rather than another literal here.
TILE_DIM = DEFAULT_TILE_R_DIM
#: Datums in the one full tile this golden multiplies.
TILE_DATUMS = DEFAULT_TILE_R_DIM * DEFAULT_TILE_C_DIM

#: Datums of the inner dimension one MVMUL consumes. SrcA's face is 16x16 and
#: the instruction is ``D[8,16] += B[8,16] * A[16,16]``, so a 32-wide K is
#: covered in two passes.
K_FACE_DIM = MAX_FACE_R_DIM
#: K faces in one tile, hence Dest writes per fidelity phase.
K_FACES = DEFAULT_TILE_R_DIM // K_FACE_DIM


def tilize_kwargs() -> dict:
    """Shape arguments for the tilize/untilize helpers, for one full tile.

    Takes no geometry: a matmul here is always one 32x32 tile, and the helpers
    recompute ``num_faces``/``face_r_dim`` from ``tile_dimensions`` anyway, so
    forwarding a caller's geometry only looked as though it did something.
    :meth:`MatmulGolden._as_tile` rejects any other shape before this is used.
    """
    return dict(
        dimensions=[TILE_DIM, TILE_DIM],
        tile_dimensions=[TILE_DIM, TILE_DIM],
    )


class MatmulGolden(Golden):
    """L1 -> SrcA, L1 -> SrcB -> product -> Dest -> L1.

    A matmul decomposes into fidelity phases the same way an element-wise
    multiply does -- the multiplier is narrower than a src datum, so each pass
    multiplies a different slice of the operands' mantissas and the partial
    products accumulate in Dest. The only difference from the element-wise case
    is that the partial product is a matrix product.

    **Accumulation is exact within one MVMUL, rounded at every Dest write.**
    Products enter the FPU's sum-of-products network at the combined width of
    both significands and are summed in wide fixed point with guard bits, so
    there is no per-MAC rounding to model and no per-product denormal flush:
    an individual product never becomes a lane result, so the flush rule does
    not reach it. :meth:`.Fpu.sum_of_products` is where that width lives.

    But one MVMUL is only ``D[8,16] += B[8,16] * A[16,16]``, so a 32-wide inner
    dimension takes **two** of them, and the second accumulates onto a Dest the
    first already rounded. ``ADDR_MOD_3`` advances K and rewinds Dest to the
    start of the tile for exactly that, and the MOP nests the K-face replay
    inside the fidelity-phase loop
    (``ckernel_template(1, FIDELITY_PHASES, TT_OP_REPLAY(...))``), so the real
    order is phase-outer, K-face-inner. This chain emits one ``src_to_dest``
    per (phase, K face) pair for that reason: summing all 32 K terms before a
    single Dest write would drop one rounding per phase, worth about an ULP at
    a 16-bit Dest.
    """

    op_name = "matmul"

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
        arch=None,
        blocks=None,
    ):
        super().__init__(arch, blocks)
        self.math_fidelity = math_fidelity

    def models_fidelity(self, cfg: OpConfig) -> bool:
        """Whether this runs the product as accumulated partial products.

        The FPU's answer, which needs the input format: the decomposition
        slices a mantissa, and an integer datum has none to slice.
        """
        return self.fpu.models_fidelity(cfg.in_formats[0])

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
        if self.models_fidelity(cfg):
            # One Dest write per (phase, K face) -- see the class docstring.
            for phase in self.fpu.phases(self.math_fidelity):
                for k_face in range(K_FACES):
                    chain.then(
                        self.src_to_dest(
                            cfg,
                            partial(
                                self.partial_product,
                                phase=phase,
                                k_face=k_face,
                                geometry=cfg.geometry,
                            ),
                            reads=("srcA", "srcB"),
                            accumulate=(phase, k_face) != (0, 0),
                            name=f"{self.op_name}[p{phase}k{k_face}]",
                        )
                    )
        else:
            self.fpu.warn_split_unmodelled(self.op_name, stacklevel=2)
            # Still one step per K face: the missing mantissa split is a
            # separate gap from the missing Dest rounding, and modelling only
            # the exact product here would hide the second one too.
            for k_face in range(K_FACES):
                chain.then(
                    self.src_to_dest(
                        cfg,
                        partial(self.apply, k_face=k_face, geometry=cfg.geometry),
                        reads=("srcA", "srcB"),
                        accumulate=k_face > 0,
                        name=f"{self.op_name}[k{k_face}]",
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
        # Against the full tile, not datums_per_tile(**geometry): the product
        # below is a 32x32 one whatever geometry says, so a partial-face tile
        # has to fail here with this message rather than further down inside
        # untilize_block, which only knows it was handed the wrong count.
        if values.numel() != TILE_DATUMS:
            raise ValueError(
                f"matmul handles one {TILE_DIM}x{TILE_DIM} tile at a time; got "
                f"{values.numel()} datums where {TILE_DATUMS} were expected "
                f"(geometry {geometry} gives {datums_per_tile(**geometry)}). "
                f"A partial-face or multi-tile matmul is a block matmul over "
                f"the inner dimension, which this golden does not model yet."
            )
        return untilize_block(
            values.float(), stimuli_format=DataFormat.Float32, **tilize_kwargs()
        ).reshape(TILE_DIM, TILE_DIM)

    @staticmethod
    def _from_tile(matrix: torch.Tensor, geometry: Dict) -> torch.Tensor:
        """The logical result back in face order, which is how Dest holds it."""
        return (
            tilize_block(
                matrix.reshape(-1).float(),
                stimuli_format=DataFormat.Float32,
                **tilize_kwargs(),
            )
            .flatten()
            .float()
        )

    def _product(
        self, srcA: torch.Tensor, srcB: torch.Tensor, geometry: Dict, k_face: int
    ) -> torch.Tensor:
        """The matrix product, in this architecture's operand order.

        ``Dest = SrcB @ SrcA``, which is not a transpose of the caller's
        arguments: :attr:`OPERAND_REGISTERS` puts the first operand in SrcB, so
        the two together give ``arg0 @ arg1``. Both halves are load-bearing and
        have to agree -- the routing without this order, or this order without
        the routing, transposes the result into something wrong everywhere but
        still plausible-looking, which is how it was last caught.

        The one place operand order lives, so a subclass that diverges does not
        have to override both the exact and the per-phase paths. The multiply
        itself is the FPU's -- including the accumulator being wider than the
        operands -- so this arranges the two registers into the matrices one
        instruction covers and hands them over. The rounding the hardware takes
        *between* K faces is the caller's, which writes Dest once per face
        rather than once per phase.

        `k_face` selects the 16-wide slice of the inner dimension this MVMUL
        covers -- columns of SrcB, rows of SrcA.
        """
        lo = k_face * K_FACE_DIM
        hi = lo + K_FACE_DIM
        product = self.fpu.sum_of_products(
            self._as_tile(srcB, geometry)[:, lo:hi],
            self._as_tile(srcA, geometry)[lo:hi, :],
        )
        return self._from_tile(product, geometry)

    def apply(self, regs: Registers, *, geometry: Dict, k_face: int) -> torch.Tensor:
        """One K face's exact product, where the split is not modelled."""
        return self._product(regs["srcA"], regs["srcB"], geometry, k_face)

    def partial_product(
        self, regs: Registers, *, phase: int, k_face: int, geometry: Dict
    ) -> torch.Tensor:
        """One fidelity phase: the matrix product of the chosen operand halves.

        The FPU picks the windows, non-finite operands included -- it carries
        those on the first phase and zeroes them on the rest, so an infinity
        reaches every output position that sums over it exactly once.
        """
        a, b = self.fpu.operand_halves(regs["srcA"], regs["srcB"], phase)
        return self._product(a, b, geometry, k_face)
