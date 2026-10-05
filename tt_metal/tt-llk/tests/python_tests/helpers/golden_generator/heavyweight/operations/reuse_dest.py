# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Element-wise binary with Dest fed back in as an operand."""

from typing import List, Optional, Sequence, Union

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import (
    DestAccumulation,
    EltwiseBinaryReuseDestType,
    MathFidelity,
    MathOperation,
)
from helpers.tile_constants import MAX_FACE_R_DIM, MAX_NUM_FACES

from ..data_transfer_blocks.l1_codec import datums_per_tile
from .chain import Chain, Registers, StageRecord
from .eltwise import EltwiseBinaryGolden
from .golden import OpConfig, check_source_layout


class EltwiseBinaryReuseDestGolden(EltwiseBinaryGolden):
    """Fold several input tiles into one output tile through Dest.

    Unlike an accumulating op, nothing sums: each pass *replaces* Dest with
    ``srcA op srcB``, and the feedback happens through the operand — one src
    register is loaded from Dest rather than from L1.

        l1_to_srcA(seed), datacopy(A2D)             seeds Dest through SrcA
        repeat: dest_to_srcA, l1_to_srcB, math      (or the srcB mirror)
        dest_to_l1

    The round trip is lossy on purpose: Dest is wider than a src register, so
    feeding it back re-quantizes, and the chain models that where the hardware
    does it rather than carrying full precision through the loop.
    """

    op_name = "eltwise_reuse_dest"

    def __init__(
        self,
        operation: MathOperation = MathOperation.Elwadd,
        math_fidelity: MathFidelity = MathFidelity.LoFi,
        reuse_dest_type: EltwiseBinaryReuseDestType = (
            EltwiseBinaryReuseDestType.DEST_TO_SRCA
        ),
        blocks=None,
    ):
        super().__init__(operation, math_fidelity, blocks)
        self.reuse_dest_type = reuse_dest_type
        self.op_name = f"reuse_dest:{operation.name.lower()}"

    #: Register the seed tile is placed in before the chain runs.
    SEED = "seed"

    def build_chain(self, cfg: OpConfig) -> Chain:
        # The seed reaches Dest through SrcA, not straight from L1. The kernel's
        # phase 1 unpacks into SrcA with IS_32B_DEST_EN=false and moves it over
        # with an A2D datacopy -- the same two steps DataCopyGolden runs -- so
        # the seed sees src-register storage on the way. l1_to_dest would model
        # the unpack_to_dest=true path, which this test does not configure.
        #
        # It only changes the numbers for an input wider than a src register's
        # SRC_MANT_BITS mantissa: a Float32 seed is truncated to 10 bits at SrcA
        # and then rounded into Dest, where l1_to_dest rounds once from the full
        # 23. Every format the sweep currently runs (Float16, Float16_b, MxFp4)
        # fits in a src datum exactly, so this is a no-op for them today.
        chain = Chain(
            [
                self.l1_to_srcA(cfg, source=self.SEED),
                self.src_to_dest(
                    cfg,
                    lambda regs: regs["srcA"],
                    reads=("srcA",),
                    name="datacopy(A2D)",
                ),
            ]
        )
        for tile in range(cfg.tiles_per_output):
            if self.reuse_dest_type is EltwiseBinaryReuseDestType.DEST_TO_SRCA:
                chain.then(
                    self.dest_to_srcA(cfg),
                    self.l1_to_srcB(cfg, source=self.source(1, tile)),
                )
            elif self.reuse_dest_type is EltwiseBinaryReuseDestType.DEST_TO_SRCB:
                chain.then(
                    self.l1_to_srcA(cfg, source=self.source(0, tile)),
                    self.dest_to_srcB(cfg),
                )
            else:
                # Named rather than caught by an else: the enum also has NONE,
                # which an else would quietly build as the SrcB chain -- a
                # different operation that still returns a plausible tile, with
                # one stimulus never read.
                raise ValueError(
                    f"{self.reuse_dest_type} does not feed Dest back into an "
                    f"operand, so there is no reuse-dest chain to build. This "
                    f"golden models DEST_TO_SRCA and DEST_TO_SRCB; for NONE use "
                    f"the plain element-wise binary golden."
                )
            # Replaces Dest rather than accumulating: the feedback is the
            # operand, not an accumulator.
            self._math_steps(chain, cfg, accumulate=False)
        return chain.then(self.dest_to_l1(cfg, into="out"))

    def run(
        self,
        stimuli: Sequence[torch.Tensor],
        in_formats: Union[DataFormat, Sequence[DataFormat]],
        out_format: DataFormat,
        *,
        inner_dim: int = 1,
        output_tiles_in_block: int = 1,
        dest_acc: Union[bool, DestAccumulation] = False,
        dest_format: Optional[DataFormat] = None,
        num_faces: int = MAX_NUM_FACES,
        face_r_dim: int = MAX_FACE_R_DIM,
        trace: Optional[List[StageRecord]] = None,
        dest_out: Optional[List[torch.Tensor]] = None,
        **pack_effects,
    ) -> torch.Tensor:
        """Run the fold over pre-tilized stimuli, returning one tile per output tile.

        `inner_dim` input tiles collapse into each output tile. The inputs a
        given output tile consumes are strided by `output_tiles_in_block`, not
        contiguous, because the kernel walks a block at a time.

        Pass a list as `dest_out` to also collect each output tile's Dest
        contents as they stood *before* the pack. That separates two ways a
        datum can disagree with silicon — the value the math left in Dest, and
        what the packer then made of it — which the packed output alone
        cannot.
        """
        src_a, src_b = stimuli
        geometry = dict(num_faces=num_faces, face_r_dim=face_r_dim)
        per_tile = datums_per_tile(**geometry)
        in_formats, cfg = self._make_config(
            in_formats,
            out_format,
            operands=2,
            geometry=geometry,
            tiles_per_output=inner_dim,
            dest_format=dest_format,
            dest_acc=dest_acc,
            pack_effects=pack_effects,
        )
        chain = self.build_chain(cfg)
        self.last_chain = chain

        flat_a, flat_b = src_a.reshape(-1), src_b.reshape(-1)
        check_source_layout(flat_a.numel() // per_tile, geometry)
        tile_count_out = flat_a.numel() // (per_tile * inner_dim)
        input_tiles_in_block = inner_dim * output_tiles_in_block

        packed: List[int] = []
        for out_tile in range(tile_count_out):
            block = out_tile // output_tiles_in_block
            in_block = out_tile % output_tiles_in_block
            regs = Registers(
                **{
                    self.SEED: self._tile_to_l1(
                        flat_a, out_tile, in_formats[0], geometry
                    )
                }
            )
            for tile in range(inner_dim):
                index = (
                    block * input_tiles_in_block
                    + tile * output_tiles_in_block
                    + in_block
                )
                regs[self.source(0, tile)] = self._tile_to_l1(
                    flat_a, index, in_formats[0], geometry
                )
                regs[self.source(1, tile)] = self._tile_to_l1(
                    flat_b, index, in_formats[1], geometry
                )
            finished = chain.run(regs, trace=trace)
            packed.extend(finished["out"])
            if dest_out is not None:
                dest_out.append(finished["dest"].flatten().float())
        return self.blocks.unpack_from_l1(packed, out_format, **geometry)
