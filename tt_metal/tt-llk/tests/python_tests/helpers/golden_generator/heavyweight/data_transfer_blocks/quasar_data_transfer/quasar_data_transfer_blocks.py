# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Quasar L1 -> Src register data-transfer blocks."""

from typing import ClassVar, FrozenSet, Mapping

from helpers.format_config import DataFormat

from ..data_transfer_blocks import MX_FORMATS, DataTransferBlocks

_NARROW_FLOAT_DEST = frozenset({DataFormat.Float16, DataFormat.Float16_b})
_TO_SRC_FLOAT = frozenset({DataFormat.Tf32, DataFormat.Float16, DataFormat.Float16_b})
#: Formats the packer can write from any Dest format: the narrow floats and
#: every MX format. Fp8_e4m3 is L1-only and belongs here too.
_PACK_NARROW = MX_FORMATS | _NARROW_FLOAT_DEST | frozenset({DataFormat.Fp8_e4m3})


class QuasarDataTransferBlocks(DataTransferBlocks):
    """Quasar: MX formats, **no block float**.

    Bfp8/Bfp8_b/Bfp4_b/Bfp2_b are Wormhole/Blackhole only — Quasar dropped block
    float in favour of the MX family, so asking for one here raises rather than
    quietly quantizing to something the hardware cannot store. MxFp4_2x_A/B are
    also absent: they are src-register storage formats, never L1 formats.
    """

    SUPPORTED_L1_FORMATS: ClassVar[FrozenSet[DataFormat]] = MX_FORMATS | frozenset(
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
    )

    #: Quasar's packer inverts the edge-mask register before the gasket applies
    #: it ("Flip polarity as Packer Gasket logic uses inverted polarity for
    #: masking", ``tt_pack_row.sv``), so a set bit masks the datum: 0xFFFF masks
    #: a whole row and 0x0000 passes it through, the opposite of Wormhole and
    #: Blackhole. ``EDGE_MASK_ROW_DATUMS_*`` in ``cpack_common.h`` agree.
    #: What the packer can write from each Dest format. A 32-bit Dest reaches
    #: everything narrow plus the two wide floats; a 16-bit float Dest reaches
    #: only the narrow formats, never Float32; and each integer width reaches
    #: only its own, except Int32, which the packer can narrow to Int8/UInt8.
    #:
    #: The same facts live in ``constraints._QUASAR_PACK_OUTPUTS``, duplicated
    #: for the reason the rest of this package duplicates: heavyweight owns its
    #: model of the hardware. No block-float entries, because Quasar has none.
    PACK_TO_L1_FORMATS: ClassVar[Mapping[DataFormat, FrozenSet[DataFormat]]] = {
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
    }

    #: Quasar is the architecture with a SrcS register; the base class has
    #: none, so the SrcS blocks refuse there.
    HAS_SRCS: ClassVar[bool] = True

    #: TestConfig skips the outlier promotion on Quasar by name, so a Float16_b
    #: Dest packed to Float16 is a state this device really does run.
    PROMOTES_OUTLIER_TO_32_BIT_DEST: ClassVar[bool] = False

    EDGE_MASK_MASKED_WHEN_SET: ClassVar[bool] = True

    #: Quasar is the only modelled architecture with a SrcS register.
    HAS_SRCS: ClassVar[bool] = True

    #: Narrow floats and Tf32 are the src targets for every float-ish input;
    #: integers stay in their own width. Int32 is deliberately absent: it
    #: unpacks to Dest or SrcS only, never SrcA/SrcB.
    #:
    #: The same facts live in ``constraints._QUASAR_UNPACK_TO_SRCA_FORMATS``.
    #: Kept separately so heavyweight does not depend on the old test-generation
    #: constraints. They were equivalent for every format in
    #: ``SUPPORTED_L1_FORMATS`` when this was written, and nothing checks that
    #: they still are. MxFp4's 2x register
    #: formats are omitted because this golden has no storage model for them,
    #: so asking for one should fail rather than quietly produce a plain value.
    UNPACK_TO_SRC_FORMATS: ClassVar[Mapping[DataFormat, FrozenSet[DataFormat]]] = {
        **{f: _TO_SRC_FLOAT for f in MX_FORMATS},
        DataFormat.Float32: _TO_SRC_FLOAT,
        DataFormat.Tf32: _TO_SRC_FLOAT,
        DataFormat.Float16: _TO_SRC_FLOAT,
        DataFormat.Float16_b: _TO_SRC_FLOAT,
        DataFormat.Fp8_e4m3: _TO_SRC_FLOAT,
        DataFormat.Int16: frozenset({DataFormat.Int16}),
        DataFormat.Int8: frozenset({DataFormat.Int8}),
        DataFormat.UInt8: frozenset({DataFormat.UInt8}),
    }

    #: The unpack-to-Dest table. Narrower than the src one in the way that
    #: matters: the unpacker does not widen, so a narrow input has no Float32
    #: Dest target, and an integer input stays in its own width. Mirrors
    #: ``constraints._QUASAR_UNPACK_TO_DEST_FORMATS``; see the note above about
    #: why it is kept separately.
    UNPACK_TO_DEST_FORMATS: ClassVar[Mapping[DataFormat, FrozenSet[DataFormat]]] = {
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
    }
