# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Element-wise binary with Dest fed back in as an operand."""

from typing import List, Optional, Sequence, Union

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import (
    EltwiseBinaryReuseDestType,
    MathFidelity,
    MathOperation,
)

from ..data_transfer_blocks.l1_codec import datums_per_tile
from .chain import Chain, Registers, StageRecord
from .eltwise import EltwiseBinaryGolden
from .golden import OpConfig


class EltwiseBinaryReuseDestGolden(EltwiseBinaryGolden):
    """Fold several input tiles into one output tile through Dest.

    Unlike an accumulating op, nothing sums: each pass *replaces* Dest with
    ``srcA op srcB``, and the feedback happens through the operand — one src
    register is loaded from Dest rather than from L1.

        l1_to_dest(seed)
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
        chain = Chain([self.l1_to_dest(cfg, source=self.SEED)])
        for tile in range(cfg.tiles_per_output):
            if self.reuse_dest_type is EltwiseBinaryReuseDestType.DEST_TO_SRCA:
                chain.then(
                    self.dest_to_srcA(cfg),
                    self.l1_to_srcB(cfg, source=self.source(1, tile)),
                )
            else:
                chain.then(
                    self.l1_to_srcA(cfg, source=self.source(0, tile)),
                    self.dest_to_srcB(cfg),
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
        dest_acc: bool = False,
        dest_format: Optional[DataFormat] = None,
        num_faces: int = 4,
        face_r_dim: int = 16,
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
        if isinstance(in_formats, DataFormat):
            in_formats = [in_formats] * 2
        geometry = dict(num_faces=num_faces, face_r_dim=face_r_dim)
        per_tile = datums_per_tile(**geometry)

        cfg = OpConfig(
            in_formats=list(in_formats),
            out_format=out_format,
            dest_format=dest_format
            or self.blocks.dest_format_for(in_formats[0], dest_acc),
            geometry=geometry,
            tiles_per_output=inner_dim,
            **pack_effects,
        )
        chain = self.build_chain(cfg)
        self.last_chain = chain

        flat_a, flat_b = src_a.reshape(-1), src_b.reshape(-1)
        tile_count_out = flat_a.numel() // (per_tile * inner_dim)
        input_tiles_in_block = inner_dim * output_tiles_in_block

        def tile_bytes(values: torch.Tensor, index: int, fmt: DataFormat):
            chunk = values[index * per_tile : (index + 1) * per_tile]
            return self.blocks.pack_to_l1(chunk, fmt, **geometry)

        packed: List[int] = []
        for out_tile in range(tile_count_out):
            block = out_tile // output_tiles_in_block
            in_block = out_tile % output_tiles_in_block
            regs = Registers(**{self.SEED: tile_bytes(flat_a, out_tile, in_formats[0])})
            for tile in range(inner_dim):
                index = (
                    block * input_tiles_in_block
                    + tile * output_tiles_in_block
                    + in_block
                )
                regs[self.source(0, tile)] = tile_bytes(flat_a, index, in_formats[0])
                regs[self.source(1, tile)] = tile_bytes(flat_b, index, in_formats[1])
            finished = chain.run(regs, trace=trace)
            packed.extend(finished["out"])
            if dest_out is not None:
                dest_out.append(finished["dest"].flatten().float())
        return self.blocks.unpack_from_l1(packed, out_format, **geometry)
