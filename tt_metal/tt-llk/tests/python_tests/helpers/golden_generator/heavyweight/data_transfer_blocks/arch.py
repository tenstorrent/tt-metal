# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The architectures, assembled from units.

An architecture is not a subclass of the blocks. It is a small object holding
one unit per hardware stage, injected into :class:`.DataTransferBlocks`.

The architectures differ in three things today: which L1 formats they hold, the
unpacker's legality tables, and how wide the FPU's multiplier is per fidelity
phase. They are listed side by side here on purpose, so "how does Blackhole
differ from Wormhole" is a diff rather than a walk through two files.
"""

from abc import ABC, abstractmethod

from helpers.format_config import DataFormat

from .l1_support import L1Support
from .storage import SRC_STORAGE_FORMATS
from .units import Fpu, Pack, Unpack

BLOCK_FLOAT_FORMATS = frozenset(f for f in DataFormat if f.is_block_float())
MX_FORMATS = frozenset(f for f in DataFormat if f.is_mx_format())

_QUASAR_TO_SRC_FLOAT = frozenset(
    {DataFormat.Tf32, DataFormat.Float16, DataFormat.Float16_b}
)

_QUASAR_NARROW_FLOAT_DEST = frozenset({DataFormat.Float16, DataFormat.Float16_b})

# Formats the packer can write from any float Dest: the narrow floats and
# every MX format. Fp8_e4m3 is L1-only and belongs here too.
_QUASAR_PACK_NARROW = (
    MX_FORMATS | _QUASAR_NARROW_FLOAT_DEST | frozenset({DataFormat.Fp8_e4m3})
)


class ArchGolden(ABC):
    """One unit per hardware stage, for one architecture.

    No SFPU unit yet: nothing here models the SFPU, so the member would be an
    empty class every architecture had to supply.
    """

    @property
    @abstractmethod
    def l1(self) -> L1Support:
        """Which L1 formats this architecture holds, and the checks on them."""

    @property
    @abstractmethod
    def unpack(self) -> Unpack:
        """L1 -> SrcA/SrcB/SrcS/Dest."""

    @property
    @abstractmethod
    def fpu(self) -> Fpu:
        """Src registers -> Dest, and Dest back to a src register."""

    @property
    @abstractmethod
    def pack(self) -> Pack:
        """Dest -> L1."""


# ----------------------------------------------------------------------
# Quasar: MX formats, no block float. Asking for a Bfp here raises rather
# than quietly quantizing to something the hardware cannot store.
# ----------------------------------------------------------------------

_QUASAR_L1 = L1Support(
    name="Quasar",
    formats=MX_FORMATS
    | frozenset(
        {
            DataFormat.Float32,
            DataFormat.Tf32,
            DataFormat.Float16,
            DataFormat.Float16_b,
            DataFormat.Fp8_e4m3,
            DataFormat.Int32,
            DataFormat.Int16,
            DataFormat.Int8,
            DataFormat.UInt8,
        }
    ),
)


class QuasarArch(ArchGolden):
    """Quasar, the only architecture with a SrcS register."""

    l1 = _QUASAR_L1

    # Integers stay in their own width. Int32 is deliberately absent: it
    # unpacks to Dest or SrcS only, never SrcA/SrcB.
    #
    # MxFp4's 2x register formats are omitted.
    unpack = Unpack(
        _QUASAR_L1,
        to_src_formats={
            **{f: _QUASAR_TO_SRC_FLOAT for f in MX_FORMATS},
            DataFormat.Float32: _QUASAR_TO_SRC_FLOAT,
            DataFormat.Tf32: _QUASAR_TO_SRC_FLOAT,
            DataFormat.Float16: _QUASAR_TO_SRC_FLOAT,
            DataFormat.Float16_b: _QUASAR_TO_SRC_FLOAT,
            DataFormat.Fp8_e4m3: _QUASAR_TO_SRC_FLOAT,
            DataFormat.Int16: frozenset({DataFormat.Int16}),
            DataFormat.Int8: frozenset({DataFormat.Int8}),
            DataFormat.UInt8: frozenset({DataFormat.UInt8}),
        },
        to_dest_formats={
            **{f: _QUASAR_NARROW_FLOAT_DEST for f in MX_FORMATS},
            DataFormat.Float32: frozenset({DataFormat.Float32})
            | _QUASAR_NARROW_FLOAT_DEST,
            DataFormat.Tf32: frozenset({DataFormat.Float32})
            | _QUASAR_NARROW_FLOAT_DEST,
            DataFormat.Float16: _QUASAR_NARROW_FLOAT_DEST,
            DataFormat.Float16_b: _QUASAR_NARROW_FLOAT_DEST,
            DataFormat.Fp8_e4m3: _QUASAR_NARROW_FLOAT_DEST,
            DataFormat.Int32: frozenset({DataFormat.Int32}),
            DataFormat.Int16: frozenset({DataFormat.Int16}),
            DataFormat.Int8: frozenset({DataFormat.Int8}),
            DataFormat.UInt8: frozenset({DataFormat.UInt8}),
        },
        has_srcs=True,
    )
    #: 7x7 bits per fidelity phase, so a src datum's 10 explicit mantissa bits
    #: split 7 high / 3 low on **both** operands.
    fpu = Fpu(_QUASAR_L1, mantissa_split=(7, 7))

    #: Quasar inverts the edge-mask register, so a set bit masks the datum,
    #: the opposite of Wormhole and Blackhole. And ``TestConfig`` skips the
    #: outlier promotion here, so a Float16_b Dest packed to Float16 is a
    #: state this device really runs.
    pack = Pack(
        _QUASAR_L1,
        to_l1_formats={
            DataFormat.Float32: _QUASAR_PACK_NARROW
            | frozenset({DataFormat.Float32, DataFormat.Tf32}),
            DataFormat.Float16: _QUASAR_PACK_NARROW,
            DataFormat.Float16_b: _QUASAR_PACK_NARROW,
            DataFormat.Int32: frozenset(
                {DataFormat.Int32, DataFormat.Int8, DataFormat.UInt8}
            ),
            DataFormat.Int16: frozenset({DataFormat.Int16}),
            DataFormat.Int8: frozenset({DataFormat.Int8}),
            DataFormat.UInt8: frozenset({DataFormat.UInt8}),
        },
        edge_mask_masked_when_set=True,
        promotes_outlier_to_32_bit_dest=False,
    )


# ----------------------------------------------------------------------
# Wormhole and Blackhole: block float, no MX.
# ----------------------------------------------------------------------

_WORMHOLE_L1 = L1Support(
    name="Wormhole",
    formats=BLOCK_FLOAT_FORMATS
    | frozenset(
        {
            DataFormat.Float32,
            DataFormat.Tf32,
            DataFormat.Float16,
            DataFormat.Float16_b,
            DataFormat.Int32,
            DataFormat.UInt32,
            DataFormat.UInt16,
            DataFormat.Int8,
            DataFormat.UInt8,
        }
    ),
)


class WormholeArch(ArchGolden):
    """Wormhole: block float, **no MX**.

    ``Fp8_e4m3`` is absent, because Wormhole's only fp8 is Lf8 (e5m2), which has no
    ``DataFormat`` in the harness and so no codec.

    The unpacker's pair table and the FPU's mantissa split are both left empty:
    unmodelled at that level of detail, not a claim that everything is legal.
    A multiply here is the exact product and warns.
    """

    l1 = _WORMHOLE_L1
    unpack = Unpack(_WORMHOLE_L1)
    fpu = Fpu(_WORMHOLE_L1)
    pack = Pack(_WORMHOLE_L1)


_BLACKHOLE_L1 = L1Support(
    name="Blackhole",
    formats=BLOCK_FLOAT_FORMATS
    | frozenset(
        {
            DataFormat.Float32,
            DataFormat.Tf32,
            DataFormat.Float16,
            DataFormat.Float16_b,
            DataFormat.Fp8_e4m3,
            DataFormat.Int32,
            DataFormat.UInt32,
            DataFormat.UInt16,
            DataFormat.Int8,
            DataFormat.UInt8,
        }
    ),
)


class BlackholeArch(ArchGolden):
    """Blackhole: block float, **no MX**.

    Differs from Wormhole in one format, ``Fp8_e4m3``, and nothing
    else, which is the point of assembling an architecture rather than
    subclassing one.
    """

    l1 = _BLACKHOLE_L1
    unpack = Unpack(_BLACKHOLE_L1)
    fpu = Fpu(_BLACKHOLE_L1)
    pack = Pack(_BLACKHOLE_L1)


__all__ = [
    "BLOCK_FLOAT_FORMATS",
    "MX_FORMATS",
    "SRC_STORAGE_FORMATS",
    "ArchGolden",
    "BlackholeArch",
    "QuasarArch",
    "WormholeArch",
]
