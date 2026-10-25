# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.constraints import is_valid_quasar_dest_mode_conversion
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation


@pytest.mark.parametrize(
    "in_fmt, out_fmt",
    [
        (DataFormat.Float16_b, DataFormat.Float32),
        (DataFormat.Int8, DataFormat.UInt8),
        (DataFormat.Int4, DataFormat.Int4),
        (DataFormat.UInt8, DataFormat.Int32),
    ],
)
def test_dest_mode_conversion_any_pair_with_dest_acc(in_fmt, out_fmt):
    assert is_valid_quasar_dest_mode_conversion(in_fmt, out_fmt, DestAccumulation.Yes)


@pytest.mark.parametrize(
    "in_fmt, out_fmt, valid",
    [
        (DataFormat.Float16_b, DataFormat.Float16_b, True),
        (DataFormat.Float16_b, DataFormat.Float16, True),
        (DataFormat.Float32, DataFormat.Float32, True),
        (DataFormat.Float32, DataFormat.Float16_b, True),
        (DataFormat.Float16_b, DataFormat.Float32, False),
        (DataFormat.Int8, DataFormat.Int32, False),
        (DataFormat.Int8, DataFormat.Int8, True),
        (DataFormat.UInt8, DataFormat.UInt8, True),
        (DataFormat.Int8, DataFormat.UInt8, False),
        (DataFormat.UInt8, DataFormat.Int8, False),
        (DataFormat.Int8, DataFormat.Float16, False),
        (DataFormat.Int4, DataFormat.Int8, True),
        (DataFormat.UInt4, DataFormat.UInt8, True),
        (DataFormat.Int4, DataFormat.Int4, False),
        (DataFormat.UInt4, DataFormat.Int8, False),
    ],
)
def test_dest_mode_conversion_without_dest_acc(in_fmt, out_fmt, valid):
    assert (
        is_valid_quasar_dest_mode_conversion(in_fmt, out_fmt, DestAccumulation.No)
        == valid
    )
