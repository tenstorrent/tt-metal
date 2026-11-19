# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from helpers.constraints import (
    get_valid_dest_accumulation_modes,
    is_valid_quasar_dest_mode_conversion,
)
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, format_dict
from helpers.param_config import quasar_dest_sync_modes
from helpers.test_variant_parameters import fp32_bits
from helpers.utils import calculate_pcc, passed_test, result_tensor


@pytest.mark.parametrize(
    "value, bits",
    [(1.0, 0x3F800000), (-2.0, 0xC0000000), (0.0, 0), (float("inf"), 0x7F800000)],
)
def test_fp32_bits(value, bits):
    assert fp32_bits(value) == bits


@pytest.mark.parametrize(
    "in_fmt, out_fmt",
    [
        (DataFormat.Float32, DataFormat.Float32),
        (DataFormat.Float32, DataFormat.Float16_b),
        (DataFormat.Int32, DataFormat.Int32),
        (DataFormat.UInt32, DataFormat.UInt8),
    ],
)
def test_get_valid_dest_accumulation_modes_32bit_input(in_fmt, out_fmt):
    formats = InputOutputFormat(in_fmt, out_fmt)
    assert get_valid_dest_accumulation_modes(formats) == [DestAccumulation.Yes]


def test_quasar_dest_sync_modes():
    assert quasar_dest_sync_modes() == [DestSync.Half, DestSync.Full]
    assert quasar_dest_sync_modes(is_perf=True) == [DestSync.Half]


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


_PCC_DTYPES = [torch.float32, torch.bfloat16, torch.float16]
_PCC_NUM_ELEMENTS = 1024


@pytest.mark.parametrize("dtype", _PCC_DTYPES, ids=str)
@pytest.mark.parametrize("constant_side", ["golden", "input"])
def test_calculate_pcc_one_constant_tensor_is_uncorrelated(dtype, constant_side):
    """A constant tensor has zero standard deviation; it cannot track a varying one."""
    varying = torch.linspace(-4.0, 4.0, _PCC_NUM_ELEMENTS).to(dtype)
    constant = torch.full((_PCC_NUM_ELEMENTS,), 3.0, dtype=dtype)
    golden, input = (
        (constant, varying) if constant_side == "golden" else (varying, constant)
    )
    assert calculate_pcc(golden, input) == 0.0


def test_calculate_pcc_constant_plus_one_outlier_is_still_low():
    """One lane off a constant golden is a real (weak) correlation, not a degenerate one."""
    golden = torch.full((_PCC_NUM_ELEMENTS,), 3.0)
    golden[0] = 4.0
    input = torch.linspace(-4.0, 4.0, _PCC_NUM_ELEMENTS)
    pcc = calculate_pcc(golden, input)
    assert pcc != 0.0
    assert pcc == pytest.approx(np.corrcoef(golden.numpy(), input.numpy())[0, 1])
    assert pcc < 0.99


def test_calculate_pcc_both_constant_is_degenerate_match():
    """Neither side varies, so there is no shape to disagree about; magnitude is the
    tolerance check's job."""
    assert (
        calculate_pcc(
            torch.full((_PCC_NUM_ELEMENTS,), 1.0),
            torch.full((_PCC_NUM_ELEMENTS,), 1.0078125),
        )
        == 1.0
    )


def test_calculate_pcc_varying_tensors_unchanged():
    golden = torch.linspace(-4.0, 4.0, _PCC_NUM_ELEMENTS)
    assert calculate_pcc(golden, golden * 2.0 + 1.0) == pytest.approx(1.0)
    assert calculate_pcc(golden, -golden) == pytest.approx(-1.0)


def _bf16_steps_around_one(n=_PCC_NUM_ELEMENTS):
    """Distinct bf16 values just above 1.0, all within Float16_b tolerance of 1.0."""
    steps = torch.arange(n) % 4
    return (1.0 + steps * 2.0**-7).to(torch.bfloat16)


def test_passed_test_a_constant_result_does_not_track_a_varying_golden():
    """Within tolerance lane by lane, but it ignores the golden's variation entirely:
    the PCC check is there to catch exactly this."""
    golden = _bf16_steps_around_one()
    result = torch.ones(golden.shape, dtype=torch.bfloat16)
    assert not passed_test(golden, result, DataFormat.Float16_b, print_errors=False)


def test_passed_test_a_constant_golden_is_judged_by_tolerance():
    """A constant golden gives PCC nothing to measure, so lane noise within tolerance
    still passes and noise beyond it still fails."""
    golden = torch.ones(_PCC_NUM_ELEMENTS, dtype=torch.bfloat16)
    assert passed_test(golden, _bf16_steps_around_one(), DataFormat.Float16_b)
    assert not passed_test(
        golden, _bf16_steps_around_one() * 2, DataFormat.Float16_b, print_errors=False
    )


def test_passed_test_a_zero_result_fails_a_small_constant_golden():
    """Every lane of a stuck-at-zero output is within atol of a 0.03 golden; the
    all-zero check in calculate_pcc must still fail it."""
    golden = torch.full((_PCC_NUM_ELEMENTS,), 0.03, dtype=torch.bfloat16)
    result = torch.zeros(_PCC_NUM_ELEMENTS, dtype=torch.bfloat16)
    assert not passed_test(golden, result, DataFormat.Float16_b, print_errors=False)


def test_passed_test_a_constant_golden_with_non_finite_lanes_is_judged_by_tolerance():
    """Like log over {0, 1}: the finite golden lanes are all 0 next to -Inf/NaN lanes.
    Those lanes are zeroed for PCC, so the golden is constant there and the tolerance
    check decides."""
    golden = torch.zeros(_PCC_NUM_ELEMENTS, dtype=torch.bfloat16)
    golden[::4] = float("-inf")
    golden[1::4] = float("nan")
    result = (torch.arange(_PCC_NUM_ELEMENTS) % 3 * 2.0**-10).to(torch.bfloat16)
    result[::4] = float("-inf")
    result[1::4] = float("nan")
    assert passed_test(golden, result, DataFormat.Float16_b)
    result[2] = 1.0
    assert not passed_test(golden, result, DataFormat.Float16_b, print_errors=False)
