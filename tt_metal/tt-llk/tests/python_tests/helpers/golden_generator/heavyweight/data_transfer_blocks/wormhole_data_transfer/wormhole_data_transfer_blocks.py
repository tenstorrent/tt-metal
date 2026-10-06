# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Wormhole L1 -> Src register data-transfer blocks."""

from typing import ClassVar, FrozenSet

from helpers.format_config import DataFormat

from ..data_transfer_blocks import BLOCK_FLOAT_FORMATS, DataTransferBlocks


class WormholeDataTransferBlocks(DataTransferBlocks):
    """Wormhole: block-float formats, **no MX**.

    The MX family is Quasar-only, so it is absent here. ``Fp8_e4m3`` is absent
    too: Wormhole's only fp8 is Lf8 (e5m2), and the harness agrees --
    ``test_eltwise_unary_datacopy`` skips with "Fp8_e4m3 not supported on
    wormhole". Lf8 is not listed because the harness has no ``DataFormat`` for
    it, so there is no codec to pack or unpack one.
    """

    SUPPORTED_L1_FORMATS: ClassVar[FrozenSet[DataFormat]] = (
        BLOCK_FLOAT_FORMATS
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
        )
    )
