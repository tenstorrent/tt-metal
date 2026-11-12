# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Per-operation goldens.

An operation declares its own pipeline in :meth:`Golden.build_chain` — the
sequence of data-transfer blocks and maths the hardware runs.
:meth:`Golden.run` executes it.

The block methods here return a :class:`~.chain.Step` rather than doing the
work, so an op composes them into a chain and the chain runs it later. They live
here rather than in :mod:`.chain` because they need the architecture's blocks,
and a chain deliberately knows nothing about hardware.

Writing a test never involves the data-transfer blocks: name the operation and
the architecture, then call it with tensors.

    golden = get_golden(Op.DataCopy, QUASAR)
    result = golden.run(stimuli, in_format, out_format)
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from helpers.data_format_inference import is_format_combination_outlier
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, PackerReluType, StochasticRounding
from helpers.tile_constants import MAX_FACE_R_DIM, MAX_NUM_FACES, MAX_TILE_ELEMENTS

from ..data_transfer_blocks import PackEdgeMask
from ..data_transfer_blocks.data_transfer_blocks import (
    DEST_32_BIT_FORMATS,
    DataTransferBlocks,
    as_dest_acc,
)
from ..data_transfer_blocks.l1_codec import datums_per_tile
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
    tiles_per_output: int = 1
    relu_type: PackerReluType = PackerReluType.NoRelu
    relu_threshold: float = 0.0
    edge_mask: Optional[PackEdgeMask] = None
    stoch_rnd: StochasticRounding = StochasticRounding.No


def check_source_layout(
    tile_count: int, geometry: Dict, dense_layout: bool = False
) -> None:
    """Refuse a stimuli layout this model cannot infer.

    ``pack_to_l1`` lays tiles out back to back, ``datums_per_tile`` apart. The
    harness has two writers and only one of them matches that:

    * ``write_matrix_w_tile_dimensions`` (``use_dense_tile_dimensions=True``)
      strides the source by the tile's own datum count, same as here.
    * ``write_matrix``, the default, always strides the source by
      ``MAX_TILE_ELEMENTS`` regardless of tile size, writing only
      ``num_faces * face_r_dim * 16`` of each stride.

    They coincide only when a tile *is* ``MAX_TILE_ELEMENTS`` datums, or when
    there is a single tile and the stride never applies.
    """
    per_tile = datums_per_tile(**geometry)
    if not dense_layout and tile_count > 1 and per_tile != MAX_TILE_ELEMENTS:
        raise ValueError(
            f"{tile_count} tiles of {per_tile} datums is ambiguous: this packs "
            f"tiles {per_tile} apart, but StimuliConfig.write_matrix strides the "
            f"source by {MAX_TILE_ELEMENTS} for any tile size, so the two agree "
            f"only at {MAX_TILE_ELEMENTS} datums per tile or on a single tile. "
            f"If the stimuli came from use_dense_tile_dimensions=True, which "
            f"strides by the tile's own size, say so with dense_layout=True -- "
            f"this cannot tell from the tensor alone. Otherwise hand over real "
            f"L1 buffers with run_l1."
        )


PACK_EFFECT_KEYWORDS = frozenset(
    {"relu_type", "relu_threshold", "edge_mask", "stoch_rnd"}
)


def check_pack_effects(pack_effects: Dict) -> Dict:
    """Reject a keyword that would otherwise fail later, inside OpConfig.

    ``run`` funnels its surplus keywords into :class:`OpConfig`, so a misspelled
    argument surfaces as an OpConfig error naming a class the caller never
    mentioned. Checking here names the caller's own options instead.
    """
    unknown = sorted(set(pack_effects) - PACK_EFFECT_KEYWORDS)
    if unknown:
        raise TypeError(
            f"got unexpected keyword argument(s) {unknown}. The surplus "
            f"keywords are the pack effects "
            f"{sorted(PACK_EFFECT_KEYWORDS)};"
        )
    return pack_effects


class Golden:
    """An operation, expressed as a chain of data-transfer blocks."""

    op_name: str = "golden"

    def __init__(self, arch=None, blocks: Optional[DataTransferBlocks] = None):
        """Bind the architecture this golden runs on.

        `arch` is an :class:`.arch.ArchProfile`, which says which blocks to use. Use one
        of the factories in :mod:`.arch` rather than passing it by hand.
        `blocks` overrides the profile's blocks, which a test uses to drive one
        architecture's op through another's blocks.
        """
        if arch is None and blocks is None:
            raise TypeError(
                f"{type(self).__name__} has no architecture. Build it with "
                f"get_golden(op, arch, ...), or pass arch= / blocks= explicitly."
            )
        self.arch = arch
        self.blocks = blocks if blocks is not None else arch.blocks()
        self.last_chain: Optional[Chain] = None
        self.last_dest_format: Optional[DataFormat] = None

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

    def _default_src_format(self, cfg: OpConfig, index: int) -> DataFormat:
        """The src format the kernel unpacks input `index` into, when none is named."""
        l1_format = cfg.in_formats[index]
        if l1_format in (DataFormat.Float32, DataFormat.Tf32) and cfg.dest_format in (
            DataFormat.Float16,
            DataFormat.Float16_b,
        ):
            return cfg.dest_format
        return self.blocks.src_format(l1_format)

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
        fmt = src_format or self._default_src_format(cfg, index)
        # A real attribute reference rather than a name built from a string:
        # a rename now breaks where grep and the linter can see it, instead of
        # at run time on whichever path happens to use that register.
        unpack = {
            "srcA": self.blocks.l1_to_srcA,
            "srcB": self.blocks.l1_to_srcB,
        }[register]

        def run(regs: Registers) -> None:
            regs[into] = unpack(regs[source], l1_format, fmt, **cfg.geometry)

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
        fmt = src_format or self._default_src_format(cfg, 0)
        convert = {
            "srcA": self.blocks.dest_to_srcA,
            "srcB": self.blocks.dest_to_srcB,
        }[register]

        def run(regs: Registers) -> None:
            # The conversion branches on the **Dest** format, and the src format
            # only decides whether a wide Dest gets its exponent rebiased.
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
        """Unpack the L1 buffer in `source` into srcA."""
        return self._l1_to_register(cfg, "srcA", source, into, index, src_format)

    def l1_to_srcB(
        self,
        cfg: OpConfig,
        source: str = "in1",
        into: str = "srcB",
        index: int = 1,
        src_format: Optional[DataFormat] = None,
    ) -> Step:
        """Unpack the L1 buffer in `source` into srcB."""
        return self._l1_to_register(cfg, "srcB", source, into, index, src_format)

    def l1_to_srcS(
        self,
        cfg: OpConfig,
        source: str = "in0",
        into: str = "srcS",
        index: int = 0,
        src_format: Optional[DataFormat] = None,
    ) -> Step:
        """Unpack the L1 buffer in `source` into srcS."""
        l1_format = cfg.in_formats[index]
        dest_acc = cfg.dest_format in DEST_32_BIT_FORMATS

        def run(regs: Registers) -> None:
            regs[into] = self.blocks.l1_to_srcS(
                regs[source], l1_format, src_format, dest_acc=dest_acc, **cfg.geometry
            )

        return Step(f"l1_to_srcS({source})", run, reads=(source,), writes=(into,))

    def l1_to_dest(
        self, cfg: OpConfig, source: str = "in0", into: str = "dest", index: int = 0
    ) -> Step:
        """Seed Dest straight from an L1 buffer, bypassing the src registers.

        Not shared with :meth:`_l1_to_register`
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

        `fn` supplies the arithmetic, which is the operation's business, and the
        block decides how the result lands in a Dest slot.

        `name` labels the step in a trace. It defaults to the operation's name,
        which is right for the step that *is* the operation. Pass it when a
        chain has a second src-to-Dest step that is something else, such as the
        datacopy that seeds Dest in a reuse-dest chain.

        With `accumulate`, the result is added to what Dest already holds rather
        than replacing it, and each pass rounds to Dest precision which is what
        lets a multi-pass op be written as a loop.
        """

        # Accumulating reads Dest as well as the src registers, so declare it:
        # otherwise dry_run calls an accumulate-before-any-write chain sound.
        declared_reads = tuple(reads)
        if accumulate and into not in declared_reads:
            declared_reads += (into,)

        def run(regs: Registers) -> None:
            # Subscript, not .get, since a missing Dest here means the chain put an
            # accumulating step before anything wrote the slot.
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

    def _promote_dest_acc(
        self,
        in_format: DataFormat,
        out_format: DataFormat,
        dest_acc: Union[bool, DestAccumulation],
    ) -> bool:
        """`dest_acc` as the device ran it, not as the caller asked for it."""
        if not isinstance(dest_acc, DestAccumulation):
            dest_acc = DestAccumulation.Yes if dest_acc else DestAccumulation.No
        if (
            self.blocks.promotes_outlier_to_32_bit_dest
            and is_format_combination_outlier(
                input_format=in_format,
                output_format=out_format,
                is_fp32_dest_acc_en=dest_acc,
            )
        ):
            return True
        return as_dest_acc(dest_acc)

    def _make_config(
        self,
        in_formats: Union[DataFormat, Sequence[DataFormat]],
        out_format: DataFormat,
        *,
        operands: int,
        geometry: Dict,
        tiles_per_output: int,
        dest_format: Optional[DataFormat],
        dest_acc: Union[bool, DestAccumulation],
        pack_effects: Dict,
    ) -> Tuple[List[DataFormat], OpConfig]:
        """Normalise one run's arguments into an :class:`OpConfig`.

        Returns the expanded `in_formats` alongside the config, since a single
        format given for several operands has to be broadcast before use.

        Also applies the architecture's `dest_acc` promotion, so a caller may
        pass the `dest_acc` its test asked for rather than the one
        ``TestConfig`` quietly substituted. See :meth:`_promote_dest_acc`.
        """
        if isinstance(in_formats, DataFormat):
            in_formats = [in_formats] * operands
        in_formats = list(in_formats)
        dest_acc = self._promote_dest_acc(in_formats[0], out_format, dest_acc)
        cfg = OpConfig(
            in_formats=in_formats,
            out_format=out_format,
            dest_format=self.blocks.resolve_dest_format(
                dest_format, in_formats[0], dest_acc
            ),
            geometry=geometry,
            tiles_per_output=tiles_per_output,
            **check_pack_effects(pack_effects),
        )
        self.last_dest_format = cfg.dest_format
        return in_formats, cfg

    def _tile_to_l1(
        self, values: torch.Tensor, index: int, fmt: DataFormat, geometry: Dict
    ) -> List[int]:
        """Tile `index` of `values`, packed the way the harness writes L1."""
        per_tile = datums_per_tile(**geometry)
        chunk = values.reshape(-1)[index * per_tile : (index + 1) * per_tile]
        return self.blocks.pack_to_l1(chunk, fmt, **geometry)

    def run(
        self,
        stimuli: Union[torch.Tensor, Sequence[torch.Tensor]],
        in_formats: Union[DataFormat, Sequence[DataFormat]],
        out_format: DataFormat,
        *,
        dest_acc: Union[bool, DestAccumulation] = False,
        dest_format: Optional[DataFormat] = None,
        num_faces: int = MAX_NUM_FACES,
        face_r_dim: int = MAX_FACE_R_DIM,
        num_tiles_per_output: int = 1,
        dense_layout: bool = False,
        trace: Optional[List[StageRecord]] = None,
        **pack_effects,
    ) -> torch.Tensor:
        """Run the operation on stimuli tensors and return the result values.

        Takes and returns what a test has: tensors. The blocks only ever handle
        L1 bytes, so this is the adapter on either side of the chain.
        """
        if isinstance(stimuli, torch.Tensor):
            stimuli = [stimuli]
        geometry = dict(num_faces=num_faces, face_r_dim=face_r_dim)

        per_tile = datums_per_tile(**geometry)
        for operand, tensor in enumerate(stimuli):
            if tensor.numel() % per_tile:
                raise ValueError(
                    f"operand {operand} has {tensor.numel()} datums, which is not "
                    f"a whole number of {per_tile}-datum tiles"
                )
        if len({tensor.numel() for tensor in stimuli}) > 1:
            raise ValueError(
                f"operands differ in size: "
                f"{[tensor.numel() for tensor in stimuli]} datums"
            )
        in_formats, cfg = self._make_config(
            in_formats,
            out_format,
            operands=len(stimuli),
            geometry=geometry,
            tiles_per_output=num_tiles_per_output,
            dest_format=dest_format,
            dest_acc=dest_acc,
            pack_effects=pack_effects,
        )
        if num_tiles_per_output > 1:
            return self._run_blocked(stimuli, in_formats, cfg, trace, dense_layout)
        # Lay the stimuli out in L1 the way the harness does, so the chain
        # reads the bytes the hardware read.
        check_source_layout(
            stimuli[0].numel() // datums_per_tile(**geometry), geometry, dense_layout
        )
        regs = Registers(
            **{
                f"in{i}": self.blocks.pack_to_l1(t, f, **geometry)
                for i, (t, f) in enumerate(zip(stimuli, in_formats))
            }
        )
        self.last_chain = self.build_chain(cfg)
        ignored = self.last_chain.unread([self.source(i) for i in range(len(stimuli))])
        if ignored:
            raise ValueError(
                f"{type(self).__name__} was given {len(stimuli)} operands but "
                f"its chain never reads {ignored}, so they would be dropped "
                f"and the result computed from the rest. Pass only the "
                f"operands this operation takes."
            )
        l1_out = self.last_chain.run(regs, result="out", trace=trace)
        return self.blocks.unpack_from_l1(l1_out, out_format, **geometry)

    def _run_blocked(
        self,
        stimuli: Sequence[torch.Tensor],
        in_formats: Sequence[DataFormat],
        cfg: OpConfig,
        trace: Optional[List[StageRecord]],
        dense_layout: bool = False,
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

        check_source_layout(total_tiles, cfg.geometry, dense_layout)
        chain = self.build_chain(cfg)
        self.last_chain = chain

        staged = [
            self.source(operand, tile)
            for tile in range(depth)
            for operand in range(len(stimuli))
        ]
        ignored = chain.unread(staged)
        if ignored:
            raise ValueError(
                f"{type(self).__name__} was given {depth} tiles per output but "
                f"its chain never reads {ignored}, so those tiles would be "
                f"dropped and the result would be computed from tile 0 alone. "
                f"This op does not fold tiles into one Dest: pass "
                f"num_tiles_per_output=1, or give it a chain that consumes "
                f"every staged tile."
            )
        packed_blocks: List[int] = []
        for block in range(total_tiles // depth):
            regs = Registers()
            for tile in range(depth):
                index = block * depth + tile
                for operand, (values, fmt) in enumerate(zip(stimuli, in_formats)):
                    regs[self.source(operand, tile)] = self._tile_to_l1(
                        values, index, fmt, cfg.geometry
                    )
            packed_blocks.extend(chain.run(regs, result="out", trace=trace))
        return self.blocks.unpack_from_l1(packed_blocks, cfg.out_format, **cfg.geometry)

    def run_l1(
        self,
        l1_buffers: Union[Sequence, Mapping[str, Sequence[int]], Registers],
        cfg: OpConfig,
        *,
        trace=None,
    ) -> Sequence[int]:
        """Run on L1 buffers and return an L1 buffer, for chaining ops together."""

        if isinstance(l1_buffers, Registers):
            regs = Registers(**{n: l1_buffers[n] for n in l1_buffers.names()})
        elif isinstance(l1_buffers, Mapping):
            regs = Registers(**l1_buffers)
        else:
            regs = Registers(**{f"in{i}": b for i, b in enumerate(l1_buffers)})
        chain = self.build_chain(cfg)

        # The chain's inputs: what a step reads before any earlier step wrote it.
        inputs, written = [], set()
        for step in chain:
            inputs += [r for r in step.reads if r not in written and r not in inputs]
            written.update(step.writes)
        missing = [name for name in inputs if name not in regs]
        unused = chain.unread(regs.names())
        if missing or unused:
            raise ValueError(
                f"{type(self).__name__}'s chain reads {inputs}; "
                + (f"missing {missing}" if missing else "")
                + ("; " if missing and unused else "")
                + (f"never reads {unused}" if unused else "")
                + ". Pass the buffers as a mapping keyed by register name."
            )

        self.last_chain = chain
        # run_l1 takes a prebuilt cfg, so _make_config never ran for it.
        self.last_dest_format = cfg.dest_format
        return chain.run(regs, result="out", trace=trace)
