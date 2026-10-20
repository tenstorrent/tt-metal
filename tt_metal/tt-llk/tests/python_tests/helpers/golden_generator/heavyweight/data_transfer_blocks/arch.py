# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The architectures, assembled from units.

An architecture is not a subclass of the blocks -- it is a small object holding
one unit per hardware stage, injected into :class:`.DataTransferBlocks`. Two
consequences, both deliberate:

* **Nothing overrides a method.** In Python an override is invisible at the
  call site, so a reader of ``blocks.l1_to_srcA(...)`` cannot tell which
  implementation runs without walking the MRO. There is exactly one
  implementation of each transfer, and what varies between architectures is
  the data each unit is constructed with.
* **A unit that did not change between generations is reused as-is.** When one
  eventually does change, it gets a second implementation of that unit only --
  not a subclass carrying every other transfer along with it.

The three architectures differ in two things today: which L1 formats they hold,
and the unpacker's legality tables. They are listed side by side here on
purpose, so "how does Blackhole differ from Wormhole" is a diff rather than a
walk through two files.
"""

from abc import ABC, abstractmethod

from helpers.format_config import DataFormat

from .l1_support import L1Support
from .storage import SRC_STORAGE_FORMATS
from .units import Fpu, Pack, Unpack

#: The two format families an architecture has one of, derived from the
#: predicates in :mod:`helpers.format_config` rather than relisted here, so a
#: new member reaches every architecture without a second copy of the list.
#: Quasar has MX and no block float; Wormhole and Blackhole the reverse.
BLOCK_FLOAT_FORMATS = frozenset(f for f in DataFormat if f.is_block_float())
MX_FORMATS = frozenset(f for f in DataFormat if f.is_mx_format())

#: Narrow floats and Tf32 are the src targets for every float-ish input.
_TO_SRC_FLOAT = frozenset({DataFormat.Tf32, DataFormat.Float16, DataFormat.Float16_b})

#: The 16-bit float Dest formats a narrow input can land in.
_NARROW_FLOAT_DEST = frozenset({DataFormat.Float16, DataFormat.Float16_b})

#: Formats the Quasar packer can write from any float Dest: the narrow floats
#: and every MX format. Fp8_e4m3 is L1-only and belongs here too.
_PACK_NARROW = MX_FORMATS | _NARROW_FLOAT_DEST | frozenset({DataFormat.Fp8_e4m3})


class ArchGolden(ABC):
    """One unit per hardware stage, for one architecture.

    No SFPU unit yet: nothing here models the SFPU, so the member would be an
    empty class every architecture had to supply. It is added with the first
    real SFPU block.
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
# Quasar: MX formats, no block float.
#
# Bfp8/Bfp8_b/Bfp4_b/Bfp2_b are Wormhole/Blackhole only -- Quasar dropped block
# float in favour of the MX family, so asking for one here raises rather than
# quietly quantizing to something the hardware cannot store. MxFp4_2x_A/B are
# also absent: they are src-register storage formats, never L1 formats.
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

    #: Integers stay in their own width. Int32 is deliberately absent: it
    #: unpacks to Dest or SrcS only, never SrcA/SrcB.
    #:
    #: The same facts live in ``constraints._QUASAR_UNPACK_TO_SRCA_FORMATS``.
    #: Kept separately so heavyweight does not depend on the old
    #: test-generation constraints; they were equivalent for every format in
    #: the L1 set when this was written. MxFp4's 2x register formats are
    #: omitted because this golden has no storage model for them, so asking for
    #: one should fail rather than quietly produce a plain value.
    unpack = Unpack(
        _QUASAR_L1,
        to_src_formats={
            **{f: _TO_SRC_FLOAT for f in MX_FORMATS},
            DataFormat.Float32: _TO_SRC_FLOAT,
            DataFormat.Tf32: _TO_SRC_FLOAT,
            DataFormat.Float16: _TO_SRC_FLOAT,
            DataFormat.Float16_b: _TO_SRC_FLOAT,
            DataFormat.Fp8_e4m3: _TO_SRC_FLOAT,
            DataFormat.Int16: frozenset({DataFormat.Int16}),
            DataFormat.Int8: frozenset({DataFormat.Int8}),
            DataFormat.UInt8: frozenset({DataFormat.UInt8}),
        },
        to_dest_formats={
            **{f: _NARROW_FLOAT_DEST for f in MX_FORMATS},
            DataFormat.Float32: frozenset({DataFormat.Float32}) | _NARROW_FLOAT_DEST,
            DataFormat.Tf32: frozenset({DataFormat.Float32}) | _NARROW_FLOAT_DEST,
            DataFormat.Float16: _NARROW_FLOAT_DEST,
            DataFormat.Float16_b: _NARROW_FLOAT_DEST,
            DataFormat.Fp8_e4m3: _NARROW_FLOAT_DEST,
            DataFormat.Int32: frozenset({DataFormat.Int32}),
            DataFormat.Int16: frozenset({DataFormat.Int16}),
            DataFormat.Int8: frozenset({DataFormat.Int8}),
            DataFormat.UInt8: frozenset({DataFormat.UInt8}),
        },
        has_srcs=True,
    )
    fpu = Fpu(_QUASAR_L1)

    #: Quasar's packer inverts the edge-mask register before the gasket applies
    #: it, so a set bit masks the datum: 0xFFFF masks a whole row and 0x0000
    #: passes it through, the opposite of Wormhole and Blackhole.
    #:
    #: ``TestConfig`` skips the outlier promotion on Quasar by name, so a
    #: Float16_b Dest packed to Float16 is a state this device really runs.
    pack = Pack(
        _QUASAR_L1,
        to_l1_formats={
            DataFormat.Float32: _PACK_NARROW
            | frozenset({DataFormat.Float32, DataFormat.Tf32}),
            DataFormat.Float16: _PACK_NARROW,
            DataFormat.Float16_b: _PACK_NARROW,
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

    ``Fp8_e4m3`` is absent: Wormhole's only fp8 is Lf8 (e5m2), and the harness
    agrees -- ``test_eltwise_unary_datacopy`` skips with "Fp8_e4m3 not
    supported on wormhole". Lf8 is not listed because the harness has no
    ``DataFormat`` for it, so there is no codec to pack or unpack one.

    The unpacker's pair table is empty, so the weaker "is it a src storage
    format at all" check applies -- unmodelled at that level of detail, not a
    claim that every pair is legal.
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

    Differs from Wormhole in one format -- it has ``Fp8_e4m3`` -- and in
    nothing else. Every unit is constructed the same way, which is the whole
    point of assembling an architecture rather than subclassing one.
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
