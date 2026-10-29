# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Value-level model of the L1 <-> register-file data-transfer blocks.

Each block answers one question: given a buffer sitting in L1, what does the
consumer see after the hardware has moved it? Blocks take and return what the
hardware does, and **L1 takes bytes**, so a golden op can chain them:

    l1_bytes -> l1_to_srcA -> [math] -> dest_to_l1 -> l1_bytes

There is no quantization step. Storing a tensor as MxFp8R *is* the
quantization, and reading it back just reports the loss.

**This class is the facade, not the implementation.** The transfers live in
units, one per hardware stage. An architecture is a set of those units
(`.arch`), injected here. Nothing subclasses this and nothing overrides a
method.

See `README.md` for what each block models and why.
"""

from functools import partial
from typing import FrozenSet, Optional, Tuple, Union

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import (
    DestAccumulation,
    MathFidelity,
    MathOperation,
    PackerReluType,
    StochasticRounding,
)

from .arch import (
    BLOCK_FLOAT_FORMATS,
    MX_FORMATS,
    ArchGolden,
    BlackholeArch,
    QuasarArch,
    WormholeArch,
)
from .l1_support import L1Buffer, L1Support

# Re-exported so the operations layer imports the register rules from the
# blocks that apply them rather than reaching past into storage. The noqa and
# the __all__ entry both stop autoflake deleting what looks unused.
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
from .units.pack_effects import PackEdgeMask


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
        everywhere except Quasar. The packer owns the rule, and the operations
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

    # ------------------------------------------------------------------
    # The FPU: what it computes, and how its multiplier behaves.
    #
    # Forwarded one method at a time, so an operation never holds an Fpu and
    # the units stay private. The fpu_ prefix says which unit decides the
    # answer, which a bare phases() or operand_halves() would not.
    # ------------------------------------------------------------------

    def fpu_elementwise(
        self, operation: MathOperation, srcA: torch.Tensor, srcB: torch.Tensor
    ) -> torch.Tensor:
        """One element-wise FPU operation on two src registers."""
        return self.arch.fpu.elementwise(operation, srcA, srcB)

    def fpu_sum_of_products(
        self, srcB_rows: torch.Tensor, srcA_cols: torch.Tensor
    ) -> torch.Tensor:
        """One matrix multiply, accumulated wider than its operands."""
        return self.arch.fpu.sum_of_products(srcB_rows, srcA_cols)

    def fpu_models_fidelity(self, in_format: DataFormat) -> bool:
        """Whether a multiply on `in_format` decomposes into fidelity phases."""
        return self.arch.fpu.models_fidelity(in_format)

    def fpu_phases(self, math_fidelity: MathFidelity) -> range:
        """The phase indices a multiply at `math_fidelity` runs."""
        return self.arch.fpu.phases(math_fidelity)

    def fpu_operand_halves(
        self, srcA: torch.Tensor, srcB: torch.Tensor, phase: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The slice of each operand this multiplier takes on `phase`."""
        return self.arch.fpu.operand_halves(srcA, srcB, phase)

    def fpu_partial_product(
        self,
        srcA: torch.Tensor,
        srcB: torch.Tensor,
        phase: int,
        dest_format: Optional[DataFormat] = None,
    ) -> torch.Tensor:
        """One fidelity phase of a multiply: window, multiply, flush."""
        return self.arch.fpu.partial_product(srcA, srcB, phase, dest_format)

    def fpu_warn_split_unmodelled(self, op_name: str, stacklevel: int = 4) -> None:
        """Say so when a multiply here falls back to the exact product.

        `stacklevel` is 4 rather than the FPU's own 3 because this wrapper adds
        a frame: the count starts at ``warnings.warn`` and has to reach past
        the warning helper, the FPU method and this one to land on the caller.
        A caller that adds a frame of its own raises it again.
        """
        self.arch.fpu.warn_split_unmodelled(op_name, stacklevel=stacklevel)


# ----------------------------------------------------------------------
# Architecture factories, under the names the architecture classes used to
# have, so callers construct blocks the same way they always did.
#
# Not subclasses: a subclass would reintroduce somewhere to override a
# transfer, which is what this structure exists to remove.
#
# ``partial`` rather than a plain function, so one assigned as a *class
# attribute* does not bind as a method and pass ``self`` as the architecture.
# A plain function is a descriptor, and ``partial`` is not.
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
