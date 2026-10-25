# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.constraints import is_valid_quasar_dest_mode_conversion
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, format_dict
from helpers.utils import result_tensor


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


@pytest.mark.parametrize("output_format", [DataFormat.Float16_b, DataFormat.Float32])
def test_result_tensor_dtype(output_format):
    res = result_tensor([1.0, 2.5, -3.0], output_format)
    assert res.dtype == format_dict[output_format]
    assert res.tolist() == [1.0, 2.5, -3.0]


def test_result_tensor_without_golden_skips_length_check():
    assert result_tensor([1.0, 2.0], DataFormat.Float32).numel() == 2


def test_result_tensor_matching_golden_length():
    golden = torch.zeros(3)
    assert result_tensor([1.0, 2.0, 3.0], DataFormat.Float32, golden).numel() == 3


def test_result_tensor_rejects_length_mismatch():
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        AssertionError, match="result has 2 datums, golden has 3"
    ):
        result_tensor([1.0, 2.0], DataFormat.Float32, torch.zeros(3))
