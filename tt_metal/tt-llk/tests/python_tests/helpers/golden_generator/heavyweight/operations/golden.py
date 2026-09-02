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

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import PackerReluType, StochasticRounding

from ..data_transfer_blocks.data_transfer_blocks import DataTransferBlocks
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
    relu_type: PackerReluType = PackerReluType.NoRelu
    relu_threshold: float = 0.0
    edge_mask: Optional[PackEdgeMask] = None
    stoch_rnd: StochasticRounding = StochasticRounding.No


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

    # ------------------------------------------------------------------
    # The data-transfer blocks, as chainable steps
    # ------------------------------------------------------------------

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
        l1_format = cfg.in_formats[index]

        def run(regs: Registers) -> None:
            regs[into] = self.blocks.l1_to_srcA(
                regs[source], l1_format, src_format, **cfg.geometry
            )

        return Step(f"l1_to_srcA({source})", run, reads=(source,), writes=(into,))

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
        l1_format = cfg.in_formats[index]

        def run(regs: Registers) -> None:
            regs[into] = self.blocks.l1_to_srcB(
                regs[source], l1_format, src_format, **cfg.geometry
            )

        return Step(f"l1_to_srcB({source})", run, reads=(source,), writes=(into,))

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
        l1_format = cfg.in_formats[index]

        def run(regs: Registers) -> None:
            regs[into] = self.blocks.l1_to_srcS(
                regs[source], l1_format, src_format, **cfg.geometry
            )

        return Step(f"l1_to_srcS({source})", run, reads=(source,), writes=(into,))

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
        fn: Callable[[Registers], torch.Tensor],
        *,
        reads: tuple = ("srcA",),
        into: str = "dest",
    ) -> Step:
        """The maths between the src registers and Dest."""

        def run(regs: Registers) -> None:
            regs[into] = fn(regs)

        return Step(self.op_name, run, reads=reads, writes=(into,))

    def accumulate_into_dest(
        self,
        fn: Callable[[Registers], torch.Tensor],
        *,
        reads: tuple = ("srcA",),
        into: str = "dest",
    ) -> Step:
        """Like :meth:`src_to_dest`, but adds into Dest instead of replacing it.

        This is what lets a multi-pass op be written as a loop.
        """

        def run(regs: Registers) -> None:
            value = fn(regs)
            regs[into] = value if into not in regs else regs[into] + value

        return Step(f"{self.op_name}+=", run, reads=reads, writes=(into,))

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
        num_faces: int = 4,
        face_r_dim: int = 16,
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
            dest_format=dest_format
            or self.blocks.dest_format_for(in_formats[0], dest_acc),
            geometry=geometry,
            **pack_effects,
        )
        # Lay the stimuli out in L1 exactly as the test harness does before
        # writing them to the device, so the chain reads the bytes the hardware
        # read. Note this packs tiles contiguously while
        # ``StimuliConfig.write_matrix`` strides its source buffer at 1024
        # elements; the two agree only because every entry in
        # SUPPORTED_TILE_SIZES has num_faces * face_r_dim * 16 == tile rows *
        # cols. Suspect this first if a partial-face multi-tile case disagrees
        # with hardware.
        regs = Registers(
            **{
                f"in{i}": self.blocks.pack_to_l1(t, f, **geometry)
                for i, (t, f) in enumerate(zip(stimuli, in_formats))
            }
        )
        self.last_chain = self.build_chain(cfg)
        l1_out = self.last_chain.run(regs, result="out", trace=trace)
        return self.blocks.unpack_from_l1(l1_out, out_format, **geometry)

    def run_l1(
        self, l1_buffers: Sequence, cfg: OpConfig, *, trace=None
    ) -> Sequence[int]:
        """Run on L1 buffers and return an L1 buffer, for chaining ops together."""
        regs = Registers(**{f"in{i}": b for i, b in enumerate(l1_buffers)})
        self.last_chain = self.build_chain(cfg)
        return self.last_chain.run(regs, result="out", trace=trace)
