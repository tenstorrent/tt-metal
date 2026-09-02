# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Quasar L1 -> Src register data-transfer blocks."""

from typing import ClassVar, FrozenSet, Optional

import torch
from helpers.format_config import DataFormat

from ..data_transfer_blocks import DataTransferBlocks, L1Buffer

#: MX formats replace block float on Quasar.
_MX_FORMATS = frozenset(
    {
        DataFormat.MxFp8R,
        DataFormat.MxFp8P,
        DataFormat.MxFp4,
        DataFormat.MxInt8,
        DataFormat.MxInt4,
        DataFormat.MxInt2,
    }
)


class QuasarDataTransferBlocks(DataTransferBlocks):
    """Quasar: MX formats, **no block float**.

    Bfp8/Bfp8_b/Bfp4_b/Bfp2_b are Wormhole/Blackhole only — Quasar dropped block
    float in favour of the MX family, so asking for one here raises rather than
    quietly quantizing to something the hardware cannot store. MxFp4_2x_A/B are
    also absent: they are src-register storage formats, never L1 formats.
    """

    SUPPORTED_L1_FORMATS: ClassVar[FrozenSet[DataFormat]] = _MX_FORMATS | frozenset(
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

    def _src_format(self, l1_format: DataFormat) -> DataFormat:
        if l1_format is DataFormat.Fp8_e4m3:
            # L1-only encoding; the unpacker converts it into the Float16 family.
            return DataFormat.Float16
        return super()._src_format(l1_format)

    def l1_to_srcA(
        self,
        l1_bytes: L1Buffer,
        l1_format: DataFormat,
        src_format: Optional[DataFormat] = None,
        **geometry,
    ) -> torch.Tensor:
        return self._l1_to_src(l1_bytes, l1_format, src_format, **geometry)
