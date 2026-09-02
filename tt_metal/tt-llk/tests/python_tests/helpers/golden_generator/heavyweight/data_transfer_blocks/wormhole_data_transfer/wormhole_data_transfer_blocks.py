# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Wormhole L1 -> Src register data-transfer blocks."""

from typing import ClassVar, FrozenSet, Optional

import torch
from helpers.format_config import DataFormat

from ..data_transfer_blocks import DataTransferBlocks, L1Buffer

#: Block float is the Wormhole/Blackhole answer to what MX does on Quasar.
_BFP_FORMATS = frozenset(
    {
        DataFormat.Bfp8,
        DataFormat.Bfp8_b,
        DataFormat.Bfp4_b,
        DataFormat.Bfp2_b,
    }
)


class WormholeDataTransferBlocks(DataTransferBlocks):
    """Wormhole: block-float formats, **no MX**.

    The MX family is Quasar-only, so it is absent here.
    """

    SUPPORTED_L1_FORMATS: ClassVar[FrozenSet[DataFormat]] = _BFP_FORMATS | frozenset(
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

    def l1_to_srcA(
        self,
        l1_bytes: L1Buffer,
        l1_format: DataFormat,
        src_format: Optional[DataFormat] = None,
        **geometry,
    ) -> torch.Tensor:
        return self._l1_to_src(l1_bytes, l1_format, src_format, **geometry)
