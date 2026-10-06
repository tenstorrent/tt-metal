# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Blackhole L1 -> Src register data-transfer blocks."""

from typing import ClassVar, FrozenSet

from helpers.format_config import DataFormat

from ..data_transfer_blocks import BLOCK_FLOAT_FORMATS, DataTransferBlocks


class BlackholeDataTransferBlocks(DataTransferBlocks):
    """Blackhole: block-float formats, **no MX**.

    The MX family is Quasar-only, so it is absent here.
    """

    SUPPORTED_L1_FORMATS: ClassVar[FrozenSet[DataFormat]] = (
        BLOCK_FLOAT_FORMATS
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
        )
    )
