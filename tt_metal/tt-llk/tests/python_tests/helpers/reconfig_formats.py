# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Format and Dest-mode parameters shared by the reconfig tests."""

from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation

# NEXT_SIZE in sources/state/reconfig/{pack,unpack,unpack_tile_shape}_reconfig_test.cpp.
NEXT_TILE_SIZE = 16 * 16 * 4

RECONFIG_FORMATS = [
    DataFormat.Float16,
    DataFormat.Float16_b,
    DataFormat.Bfp8,
    DataFormat.Bfp8_b,
    DataFormat.Float32,
    DataFormat.Int32,
    DataFormat.Tf32,
    DataFormat.UInt32,
    DataFormat.UInt16,
    DataFormat.Int8,
    DataFormat.UInt8,
]


def reconfig_dest_acc(
    prev_format: DataFormat, next_format: DataFormat
) -> list[DestAccumulation]:
    """Integer math requires FP32 Dest accumulation (tt-metal#34499), so integer pairs run only ``Yes``."""
    return (
        [DestAccumulation.Yes]
        if prev_format.is_integer() or next_format.is_integer()
        else [DestAccumulation.No, DestAccumulation.Yes]
    )
