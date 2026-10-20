# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Value-level model of the L1 <-> register-file data-transfer blocks.

Each block answers one question: given a buffer sitting in L1, what does the
consumer see after the hardware has moved it? Blocks take and return the things
the hardware takes and returns — **L1 takes bytes** — so a golden op can chain
them the way the pipeline does:

    l1_bytes -> l1_to_srcA -> [math] -> dest_to_l1 -> l1_bytes

There is deliberately no quantization step. Storing a tensor as MxFp8R *is* the
quantization; by the time the L1 bytes exist the loss has happened, and reading
them back just reports it. Use :func:`.l1_codec.pack_to_l1` to build a buffer.

That leaves one real conversion for these blocks to model: the unpacker lands
every L1 format in one of a small number of src-register storage families. The
family, not the L1 format, is what the FPU reads.

**This class is the façade, not the implementation.** The transfers live on the
units in :mod:`.units` -- one per hardware stage -- and an architecture is a
set of those units (:mod:`.arch`), injected here. Nothing subclasses this and
nothing overrides a method: there is one implementation of each transfer, and
architectures differ in the data their units carry. See :mod:`.arch` for why.
"""

from functools import partial
from typing import FrozenSet, Optional, Union

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, PackerReluType, StochasticRounding

from .arch import (
    BLOCK_FLOAT_FORMATS,
    MX_FORMATS,
    ArchGolden,
    BlackholeArch,
    QuasarArch,
    WormholeArch,
)
from .l1_support import L1Buffer, L1Support
from .pack_effects import PackEdgeMask

# Re-exported so the operations layer keeps importing the register rules from
# the blocks that apply them, rather than reaching past them into storage.
# Listed in __all__ because autoflake strips an import nothing in this module
# body uses, which is exactly what a re-export is.
from .storage import (  # noqa: F401
    BF16_MANT_BITS,
    DEST_32_BIT_FORMATS,
    DEST_STORAGE_FORMATS,
    INT32_DEST_COMPATIBLE,
    SRC_MANT_BITS,
    SRC_STORAGE_FORMATS,
    SRCS_STORAGE_FORMATS,
    as_dest_acc,
    flush_subnormals,
    saturate_to_integer,
    truncate_mantissa,
    truncate_src_mantissa,
)
from .units.pack import UnmodelledHardwareWarning  # noqa: F401


class DataTransferBlocks:
    """The data-transfer blocks for one architecture.

    Construct with an :class:`.arch.ArchGolden`, or use one of the
    architecture factories below.
    """

    def __init__(self, arch: ArchGolden) -> None:
        self.arch = arch

    # ------------------------------------------------------------------
    # L1 -> registers
    # ------------------------------------------------------------------

    def l1_to_srcA(
        self,
        l1_bytes: L1Buffer,
        l1_format: DataFormat,
        src_format: Optional[DataFormat] = None,
        **geometry,
    ) -> torch.Tensor:
        """Values visible in SrcA after unpacking `l1_bytes` from L1."""
        return self.arch.unpack.l1_to_src(l1_bytes, l1_format, src_format, **geometry)

    def l1_to_srcB(
        self,
        l1_bytes: L1Buffer,
        l1_format: DataFormat,
        src_format: Optional[DataFormat] = None,
        **geometry,
    ) -> torch.Tensor:
        """Values visible in SrcB. SrcA and SrcB share the datum layout."""
        return self.arch.unpack.l1_to_src(l1_bytes, l1_format, src_format, **geometry)

    def l1_to_srcS(
        self,
        l1_bytes: L1Buffer,
        l1_format: DataFormat,
        src_format: Optional[DataFormat] = None,
        *,
        dest_acc: Union[bool, DestAccumulation] = False,
        **geometry,
    ) -> torch.Tensor:
        """Values visible in SrcS. Quasar only."""
        return self.arch.unpack.l1_to_srcs(
            l1_bytes, l1_format, src_format, dest_acc=dest_acc, **geometry
        )

    def l1_to_dest(
        self,
        l1_bytes: L1Buffer,
        l1_format: DataFormat,
        dest_format: Optional[DataFormat] = None,
        *,
        dest_acc: Union[bool, DestAccumulation] = False,
        **geometry,
    ) -> torch.Tensor:
        """Load an L1 buffer straight into Dest, bypassing the src registers."""
        return self.arch.unpack.l1_to_dest(
            l1_bytes, l1_format, dest_format, dest_acc=dest_acc, **geometry
        )

    # ------------------------------------------------------------------
    # Src registers <-> Dest
    # ------------------------------------------------------------------

    def src_to_dest(
        self,
        values: torch.Tensor,
        dest_format: DataFormat = DataFormat.Float32,
        current: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Write a math result into Dest, replacing or accumulating."""
        return self.arch.fpu.src_to_dest(values, dest_format, current)

    def dest_to_srcA(
        self,
        dest_values: torch.Tensor,
        dest_format: DataFormat = DataFormat.Float32,
        src_format: DataFormat = DataFormat.Float16_b,
    ) -> torch.Tensor:
        """Move Dest back into SrcA, so an op can use its own result as an operand."""
        return self.arch.fpu.dest_to_src(dest_values, dest_format, src_format)

    def dest_to_srcB(
        self,
        dest_values: torch.Tensor,
        dest_format: DataFormat = DataFormat.Float32,
        src_format: DataFormat = DataFormat.Float16_b,
    ) -> torch.Tensor:
        """Move Dest back into SrcB. See :meth:`dest_to_srcA`."""
        return self.arch.fpu.dest_to_src(dest_values, dest_format, src_format)

    # ------------------------------------------------------------------
    # Dest -> L1
    # ------------------------------------------------------------------

    def dest_to_l1(
        self,
        dest_values: torch.Tensor,
        l1_format: DataFormat,
        dest_format: DataFormat = DataFormat.Float32,
        *,
        relu_type: PackerReluType = PackerReluType.NoRelu,
        relu_threshold: float = 0.0,
        edge_mask: Optional[PackEdgeMask] = None,
        stoch_rnd: StochasticRounding = StochasticRounding.No,
        **geometry,
    ) -> list:
        """Pack Dest into L1 — the T2 half, mirroring :meth:`l1_to_srcA`."""
        return self.arch.pack.dest_to_l1(
            dest_values,
            l1_format,
            dest_format,
            relu_type=relu_type,
            relu_threshold=relu_threshold,
            edge_mask=edge_mask,
            stoch_rnd=stoch_rnd,
            **geometry,
        )

    # ------------------------------------------------------------------
    # L1 access and format queries
    # ------------------------------------------------------------------

    def pack_to_l1(
        self, tensor: torch.Tensor, l1_format: DataFormat, **geometry
    ) -> list:
        """Lay `tensor` out in L1 as `l1_format`. Where precision is lost."""
        return self.arch.l1.write(tensor, l1_format, **geometry)

    def unpack_from_l1(
        self, l1_bytes: L1Buffer, l1_format: DataFormat, **geometry
    ) -> torch.Tensor:
        """Read `l1_bytes` back as values, before any src-register conversion."""
        return self.arch.l1.read(l1_bytes, l1_format, **geometry)

    def supports(self, l1_format: DataFormat) -> bool:
        """Whether this golden can move `l1_format` through L1 on this arch."""
        return self.arch.l1.supports(l1_format)

    @property
    def supported_dest_formats(self) -> FrozenSet[DataFormat]:
        """Dest formats available on this architecture."""
        return self.arch.l1.dest_formats

    @property
    def promotes_outlier_to_32_bit_dest(self) -> bool:
        """Whether the harness promotes this architecture's outlier combination.

        An exponent-B input with a Float16 output runs with a 32-bit Dest
        everywhere except Quasar. The packer owns the rule; the operations
        layer needs the answer when it resolves ``dest_acc``.
        """
        return self.arch.pack.promotes_outlier_to_32_bit_dest

    def src_format(self, l1_format: DataFormat) -> DataFormat:
        """The src-register storage format the unpacker lands `l1_format` in."""
        return self.arch.unpack.src_format(l1_format)

    def srcs_format(
        self, l1_format: DataFormat, dest_acc: Union[bool, DestAccumulation] = False
    ) -> DataFormat:
        """The SrcS storage format the unpacker lands `l1_format` in."""
        return self.arch.unpack.srcs_format(l1_format, dest_acc)

    def dest_format_for(
        self,
        l1_input_format: DataFormat,
        dest_acc: Union[bool, DestAccumulation] = False,
    ) -> DataFormat:
        """The Dest format for an op whose *input* was `l1_input_format`."""
        return self.arch.unpack.dest_format_for(l1_input_format, dest_acc)

    def resolve_dest_format(
        self,
        dest_format: Optional[DataFormat],
        l1_input_format: DataFormat,
        dest_acc: Union[bool, DestAccumulation],
    ) -> DataFormat:
        """The Dest format to run with: derived from the input, or validated."""
        return self.arch.unpack.resolve_dest_format(
            dest_format, l1_input_format, dest_acc
        )


# ----------------------------------------------------------------------
# Architecture factories.
#
# Kept under the original names so `blocks_class = Quasar...` and
# `self.blocks_class()` in the operations layer keep working unchanged.
#
# ``partial`` rather than a plain function, and deliberately not a subclass.
# A plain function assigned to a class attribute becomes a *method*: functions
# are descriptors, so ``self.blocks_class()`` would pass ``self`` as the arch
# and raise. ``partial`` is not a descriptor, so it survives the assignment
# intact. A subclass would work too, but it would reintroduce somewhere to
# override a transfer -- the thing this structure exists to remove.
# ----------------------------------------------------------------------

QuasarDataTransferBlocks = partial(DataTransferBlocks, QuasarArch())
QuasarDataTransferBlocks.__doc__ = (
    "Quasar: MX formats, no block float, and the only SrcS register."
)

WormholeDataTransferBlocks = partial(DataTransferBlocks, WormholeArch())
WormholeDataTransferBlocks.__doc__ = "Wormhole: block float, no MX."

BlackholeDataTransferBlocks = partial(DataTransferBlocks, BlackholeArch())
BlackholeDataTransferBlocks.__doc__ = "Blackhole: block float, no MX, plus Fp8_e4m3."


__all__ = [
    "BF16_MANT_BITS",
    "BLOCK_FLOAT_FORMATS",
    "BlackholeArch",
    "BlackholeDataTransferBlocks",
    "DEST_32_BIT_FORMATS",
    "DEST_STORAGE_FORMATS",
    "DataTransferBlocks",
    "INT32_DEST_COMPATIBLE",
    "L1Buffer",
    "L1Support",
    "MX_FORMATS",
    "PackEdgeMask",
    "QuasarArch",
    "QuasarDataTransferBlocks",
    "SRC_MANT_BITS",
    "SRC_STORAGE_FORMATS",
    "SRCS_STORAGE_FORMATS",
    "UnmodelledHardwareWarning",
    "WormholeArch",
    "WormholeDataTransferBlocks",
    "as_dest_acc",
    "flush_subnormals",
    "saturate_to_integer",
    "truncate_mantissa",
    "truncate_src_mantissa",
]
