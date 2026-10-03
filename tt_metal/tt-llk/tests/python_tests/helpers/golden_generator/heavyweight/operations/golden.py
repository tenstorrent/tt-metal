# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Per-operation goldens.

An operation declares its own pipeline in :meth:`Golden.build_chain` — the
sequence of data-transfer blocks and maths the hardware runs — and
:meth:`Golden.run` executes it.

The block methods here return a :class:`~.chain.Step` rather than doing the
work, so an op composes them into a chain and the chain runs it later. They live
here rather than in :mod:`.chain` because they need the architecture's blocks,
and a chain deliberately knows nothing about hardware.

Writing a test never involves the data-transfer blocks: pick the golden for the
architecture and call it with tensors.

    golden = QuasarDataCopyGolden()
    result = golden.run(stimuli, in_format, out_format)
"""

from dataclasses import dataclass, field, fields
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import PackerReluType, StochasticRounding
from helpers.tile_constants import MAX_FACE_R_DIM, MAX_NUM_FACES, MAX_TILE_ELEMENTS

from ..data_transfer_blocks.data_transfer_blocks import DataTransferBlocks
from ..data_transfer_blocks.l1_codec import datums_per_tile
from ..data_transfer_blocks.pack_effects import PackEdgeMask
from .chain import Chain, Registers, StageRecord, Step


@dataclass
class OpConfig:
    """How one invocation of an operation is configured.

    Not a register file — this is the set of knobs the chain is *built* from,
    fixed before it runs. Data flows through :class:`~.chain.Registers`.
    """

    in_formats: Sequence[DataFormat]
    out_format: DataFormat
    dest_format: DataFormat
    geometry: Dict = field(default_factory=dict)
    #: Input tiles folded into one output tile. 1 is the ordinary case: one
    #: tile in, one tile out. How they fold is the operation's business — an
    #: accumulating op sums them into Dest, a reuse-dest op feeds Dest back in
    #: as an operand.
    tiles_per_output: int = 1
    relu_type: PackerReluType = PackerReluType.NoRelu
    relu_threshold: float = 0.0
    edge_mask: Optional[PackEdgeMask] = None
    stoch_rnd: StochasticRounding = StochasticRounding.No


def check_source_layout(tile_count: int, geometry: Dict) -> None:
    """Refuse a stimuli layout this model cannot infer.

    ``pack_to_l1`` lays tiles out back to back, ``datums_per_tile`` apart. The
    harness has two writers and only one of them matches that:

    * ``write_matrix_w_tile_dimensions`` (``use_dense_tile_dimensions=True``)
      strides the source by the tile's own datum count -- same as here.
    * ``write_matrix``, the default, always strides the source by
      ``MAX_TILE_ELEMENTS`` regardless of tile size, writing only
      ``num_faces * face_r_dim * 16`` of each stride.

    They coincide only when a tile *is* ``MAX_TILE_ELEMENTS`` datums, or when
    there is a single tile and the stride never applies. For a smaller tile over
    several tiles the two read different source elements from tile 1 on, so the
    golden would quietly compute on data the device never saw. Which writer a
    test used is not visible from here, so raise rather than pick one.
    """
    per_tile = datums_per_tile(**geometry)
    if tile_count > 1 and per_tile != MAX_TILE_ELEMENTS:
        raise ValueError(
            f"{tile_count} tiles of {per_tile} datums is ambiguous: this packs "
            f"tiles {per_tile} apart, but StimuliConfig.write_matrix strides the "
            f"source by {MAX_TILE_ELEMENTS} for any tile size, so the two agree "
            f"only at {MAX_TILE_ELEMENTS} datums per tile or on a single tile. "
            f"Use use_dense_tile_dimensions=True in the StimuliConfig, which "
            f"strides by the tile's own size, or hand over real L1 buffers with "
            f"run_l1."
        )


def check_pack_effects(pack_effects: Dict) -> Dict:
    """Reject a keyword that would otherwise fail later, inside OpConfig.

    ``run`` funnels its surplus keywords into :class:`OpConfig`, so a misspelled
    argument surfaces as an OpConfig error naming a class the caller never
    mentioned. Checking here names the caller's own options instead.
    """
    unknown = sorted(set(pack_effects) - {f.name for f in fields(OpConfig)})
    if unknown:
        raise TypeError(
            f"run() got unexpected keyword argument(s) {unknown}. It takes "
            f"dest_acc, dest_format, num_faces, face_r_dim, "
            f"num_tiles_per_output, trace, and the pack effects relu_type, "
            f"relu_threshold, edge_mask and stoch_rnd."
        )
    return pack_effects


class Golden:
    """An operation, expressed as a chain of data-transfer blocks."""

    #: The architecture's blocks. Set by each architecture's subclass, so
    #: constructing a golden needs no arguments.
    blocks_class: Optional[type] = None

    op_name: str = "golden"

    def __init__(self, blocks: Optional[DataTransferBlocks] = None):
        if blocks is None:
            if self.blocks_class is None:
                raise TypeError(
                    f"{type(self).__name__} has no architecture. Use an "
                    f"architecture's golden (e.g. QuasarDataCopyGolden) or pass "
                    f"blocks explicitly."
                )
            blocks = self.blocks_class()
        self.blocks = blocks

    # ------------------------------------------------------------------
    # What an operation declares
    # ------------------------------------------------------------------

    def build_chain(self, cfg: OpConfig) -> Chain:
        """The pipeline this operation runs. Every op declares its own."""
        raise NotImplementedError

    @staticmethod
    def source(operand: int, tile: int = 0) -> str:
        """Register holding one operand's L1 buffer for one tile of a block.

        Tile 0 keeps the plain name, so an op that does not accumulate reads
        ``in0``/``in1`` and its trace stays uncluttered.
        """
        return f"in{operand}" if tile == 0 else f"in{operand}_t{tile}"

    # ------------------------------------------------------------------
    # The data-transfer blocks, as chainable steps
    # ------------------------------------------------------------------

    #: Src registers an unpack step can target, and the block method for each.
    #: The register name is the whole difference between the three public
    #: methods below, so they share one builder -- the format-slot argument
    #: order had to be corrected in two separate copies of the Dest feedback
    #: pair for exactly this reason.
    def _l1_to_register(
        self,
        cfg: OpConfig,
        register: str,
        source: str,
        into: str,
        index: int,
        src_format: Optional[DataFormat],
    ) -> Step:
        """One unpack step, L1 -> `register`."""
        l1_format = cfg.in_formats[index]
        unpack = getattr(self.blocks, f"l1_to_{register}")

        def run(regs: Registers) -> None:
            regs[into] = unpack(regs[source], l1_format, src_format, **cfg.geometry)

        return Step(f"l1_to_{register}({source})", run, reads=(source,), writes=(into,))

    def _dest_to_register(
        self,
        cfg: OpConfig,
        register: str,
        source: str,
        into: str,
        src_format: Optional[DataFormat],
    ) -> Step:
        """One Dest-feedback step, Dest -> `register`."""
        fmt = src_format or self.blocks.src_format(cfg.in_formats[0])
        convert = getattr(self.blocks, f"dest_to_{register}")

        def run(regs: Registers) -> None:
            # The conversion branches on the **Dest** format; the src format
            # only decides whether a wide Dest gets its exponent rebiased. Both
            # are needed, and they differ whenever dest_acc widens Dest or the
            # input is Float32/Tf32.
            regs[into] = convert(regs[source], cfg.dest_format, fmt)

        return Step(
            f"dest_to_{register}({source})", run, reads=(source,), writes=(into,)
        )

    def l1_to_srcA(
        self,
        cfg: OpConfig,
        source: str = "in0",
        into: str = "srcA",
        index: int = 0,
        src_format: Optional[DataFormat] = None,
    ) -> Step:
        """Unpack the L1 buffer in `source` into srcA.

        `src_format` overrides the storage format the unpacker lands it in;
        ``None`` lets the architecture choose.
        """
        return self._l1_to_register(cfg, "srcA", source, into, index, src_format)

    def l1_to_srcB(
        self,
        cfg: OpConfig,
        source: str = "in1",
        into: str = "srcB",
        index: int = 1,
        src_format: Optional[DataFormat] = None,
    ) -> Step:
        """Unpack the L1 buffer in `source` into srcB.

        `src_format` overrides the storage format the unpacker lands it in;
        ``None`` lets the architecture choose.
        """
        return self._l1_to_register(cfg, "srcB", source, into, index, src_format)

    def l1_to_srcS(
        self,
        cfg: OpConfig,
        source: str = "in0",
        into: str = "srcS",
        index: int = 0,
        src_format: Optional[DataFormat] = None,
    ) -> Step:
        """Unpack the L1 buffer in `source` into srcS.

        `src_format` overrides the storage format the unpacker lands it in;
        ``None`` lets the architecture choose.
        """
        return self._l1_to_register(cfg, "srcS", source, into, index, src_format)

    def l1_to_dest(
        self, cfg: OpConfig, source: str = "in0", into: str = "dest", index: int = 0
    ) -> Step:
        """Seed Dest straight from an L1 buffer, bypassing the src registers.

        Not shared with :meth:`_l1_to_register`: the third argument is the Dest
        format, not a src format, which is the slot the two kept getting mixed
        up in.
        """
        l1_format = cfg.in_formats[index]

        def run(regs: Registers) -> None:
            regs[into] = self.blocks.l1_to_dest(
                regs[source], l1_format, cfg.dest_format, **cfg.geometry
            )

        return Step(f"l1_to_dest({source})", run, reads=(source,), writes=(into,))

    def dest_to_srcA(
        self,
        cfg: OpConfig,
        source: str = "dest",
        into: str = "srcA",
        src_format: Optional[DataFormat] = None,
    ) -> Step:
        """Feed Dest back into SrcA, re-quantized to src-register precision."""
        return self._dest_to_register(cfg, "srcA", source, into, src_format)

    def dest_to_srcB(
        self,
        cfg: OpConfig,
        source: str = "dest",
        into: str = "srcB",
        src_format: Optional[DataFormat] = None,
    ) -> Step:
        """Feed Dest back into SrcB, re-quantized to src-register precision."""
        return self._dest_to_register(cfg, "srcB", source, into, src_format)

    def dest_to_l1(
        self, cfg: OpConfig, into: str = "out", source: str = "dest"
    ) -> Step:
        """Pack Dest out to the L1 buffer `into`, with the packer's effects."""

        def run(regs: Registers) -> None:
            regs[into] = self.blocks.dest_to_l1(
                regs[source],
                cfg.out_format,
                cfg.dest_format,
                relu_type=cfg.relu_type,
                relu_threshold=cfg.relu_threshold,
                edge_mask=cfg.edge_mask,
                stoch_rnd=cfg.stoch_rnd,
                **cfg.geometry,
            )

        return Step(f"dest_to_l1({into})", run, reads=(source,), writes=(into,))

    def src_to_dest(
        self,
        cfg: OpConfig,
        fn: Callable[[Registers], torch.Tensor],
        *,
        reads: tuple = ("srcA",),
        into: str = "dest",
        accumulate: bool = False,
        name: Optional[str] = None,
    ) -> Step:
        """The maths between the src registers and Dest.

        `fn` supplies the arithmetic, which is the operation's business; the
        block decides how the result lands in a Dest slot.

        `name` labels the step in a trace. It defaults to the operation's name,
        which is right for the step that *is* the operation; pass it when a
        chain has a second src-to-Dest step that is something else, such as the
        datacopy that seeds Dest in a reuse-dest chain.

        With `accumulate`, the result is added to what Dest already holds rather
        than replacing it, and each pass rounds to Dest precision — which is what
        lets a multi-pass op be written as a loop.
        """

        # Accumulating reads Dest as well as the src registers, so declare it:
        # otherwise dry_run calls an accumulate-before-any-write chain sound.
        declared_reads = tuple(reads)
        if accumulate and into not in declared_reads:
            declared_reads += (into,)

        def run(regs: Registers) -> None:
            # Subscript, not .get -- a missing Dest here means the chain put an
            # accumulating step before anything wrote the slot, and silently
            # accumulating onto nothing turns that into a plain replace with a
            # plausible-looking result.
            current = regs[into] if accumulate else None
            regs[into] = self.blocks.src_to_dest(fn(regs), cfg.dest_format, current)

        if name is None:
            name = f"{self.op_name}+=" if accumulate else self.op_name
        elif accumulate:
            name = f"{name}+="
        return Step(name, run, reads=declared_reads, writes=(into,))

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def run(
        self,
        stimuli: Union[torch.Tensor, Sequence[torch.Tensor]],
        in_formats: Union[DataFormat, Sequence[DataFormat]],
        out_format: DataFormat,
        *,
        dest_acc: bool = False,
        dest_format: Optional[DataFormat] = None,
        num_faces: int = MAX_NUM_FACES,
        face_r_dim: int = MAX_FACE_R_DIM,
        num_tiles_per_output: int = 1,
        trace: Optional[List[StageRecord]] = None,
        **pack_effects,
    ) -> torch.Tensor:
        """Run the operation on stimuli tensors and return the result values.

        Takes and returns what a test has: tensors. The blocks only ever handle
        L1 bytes, so this is the adapter on either side of the chain — the
        stimuli are packed into L1 buffers before it runs and the output buffer
        is read back after. Use :meth:`run_l1` to hand over real buffers and
        skip both conversions.
        """
        if isinstance(stimuli, torch.Tensor):
            stimuli = [stimuli]
        if isinstance(in_formats, DataFormat):
            in_formats = [in_formats] * len(stimuli)
        geometry = dict(num_faces=num_faces, face_r_dim=face_r_dim)

        cfg = OpConfig(
            in_formats=list(in_formats),
            out_format=out_format,
            dest_format=self.blocks.resolve_dest_format(
                dest_format, in_formats[0], dest_acc
            ),
            geometry=geometry,
            tiles_per_output=num_tiles_per_output,
            **check_pack_effects(pack_effects),
        )
        if num_tiles_per_output > 1:
            return self._run_blocked(stimuli, in_formats, cfg, trace)
        # Lay the stimuli out in L1 the way the harness does, so the chain
        # reads the bytes the hardware read. Tiles go back to back here, which
        # matches the dense writer but not the default one -- see
        # check_source_layout for the cases that cannot agree.
        check_source_layout(stimuli[0].numel() // datums_per_tile(**geometry), geometry)
        regs = Registers(
            **{
                f"in{i}": self.blocks.pack_to_l1(t, f, **geometry)
                for i, (t, f) in enumerate(zip(stimuli, in_formats))
            }
        )
        self.last_chain = self.build_chain(cfg)
        l1_out = self.last_chain.run(regs, result="out", trace=trace)
        return self.blocks.unpack_from_l1(l1_out, out_format, **geometry)

    def _run_blocked(
        self,
        stimuli: Sequence[torch.Tensor],
        in_formats: Sequence[DataFormat],
        cfg: OpConfig,
        trace: Optional[List[StageRecord]],
    ) -> torch.Tensor:
        """Run one chain per block of `cfg.tiles_per_output` input tiles.

        Each block folds its tiles into a single Dest and packs once, so the
        output holds one tile per block rather than one per input tile.
        """
        per_tile = datums_per_tile(**cfg.geometry)
        depth = cfg.tiles_per_output
        total_tiles = stimuli[0].numel() // per_tile
        if total_tiles % depth:
            raise ValueError(
                f"{total_tiles} input tiles is not a multiple of the "
                f"{depth} tiles accumulated per Dest"
            )

        check_source_layout(total_tiles, cfg.geometry)
        chain = self.build_chain(cfg)
        self.last_chain = chain
        packed_blocks: List[int] = []
        for block in range(total_tiles // depth):
            regs = Registers()
            for tile in range(depth):
                index = block * depth + tile
                for operand, (values, fmt) in enumerate(zip(stimuli, in_formats)):
                    chunk = values.reshape(-1)[
                        index * per_tile : (index + 1) * per_tile
                    ]
                    regs[self.source(operand, tile)] = self.blocks.pack_to_l1(
                        chunk, fmt, **cfg.geometry
                    )
            packed_blocks.extend(chain.run(regs, result="out", trace=trace))
        return self.blocks.unpack_from_l1(packed_blocks, cfg.out_format, **cfg.geometry)

    def run_l1(
        self, l1_buffers: Sequence, cfg: OpConfig, *, trace=None
    ) -> Sequence[int]:
        """Run on L1 buffers and return an L1 buffer, for chaining ops together."""
        regs = Registers(**{f"in{i}": b for i, b in enumerate(l1_buffers)})
        self.last_chain = self.build_chain(cfg)
        return self.last_chain.run(regs, result="out", trace=trace)
