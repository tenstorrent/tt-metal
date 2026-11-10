# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tt_py_test_utils_common.utility_functions import comp_allclose_and_pcc
from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_arange import run_moreh_arange

pytestmark = pytest.mark.use_module_device


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "start_end_step, dtype, tilized",
    [
        ([10.9, -13, -0.3], "bfloat16", True),
        ([-100, 320, 1], "int32", True),
        ([2.3, 15.3, 0.5], "float32", True),
        ([10.9, -13, -0.3], "bfloat16", False),
        ([-100, 320, 1], "int32", False),
        ([2.3, 15.3, 0.5], "float32", False),
        ([0, 4763, 1], "int32", True),
        ([0, 4763, 1], "int32", False),
    ],
    ids=[
        "bfloat16",
        "int32",
        "float32",
        "bfloat16_row_major",
        "int32_row_major",
        "float32_row_major",
        "multi_tile_per_core",
        "multi_tile_per_core_row_major",
    ],
)
def test_moreh_arange(start_end_step, dtype, tilized, device):
    torch.manual_seed(0)
    run_moreh_arange(start_end_step, False, dtype, tilized, device)


@pytest.mark.merge_gate
def test_moreh_arange_provided_output(device):
    torch.manual_seed(0)
    # Not the nightly helper: it fills the output with torch.empty, which can already hold the expected values.
    expected = torch.arange(10.9, -13, -0.3).to(torch.bfloat16)
    tt_output = ttnn.from_torch(
        torch.full([1, len(expected)], float("nan")), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    ttnn.moreh_arange(10.9, -13, -0.3, device, output=tt_output, untilize_out=False, dtype=ttnn.bfloat16)
    actual = ttnn.to_torch(tt_output).reshape(expected.shape)

    passing, output_pcc = comp_allclose_and_pcc(expected, actual, rtol=0.1, atol=0.1)
    assert passing, output_pcc


@pytest.mark.parametrize(
    "start, end, step",
    [
        (16777184, 16777248, 1),
        (16777216, 16777280, 1),
        (33554433, 33554497, 1),
        (-33554497, -33554433, 1),
        (-16777250, 16777250, 1),
        (16777217, 16777217 + 3 * 100, 3),
        (33554433, 33554433 - 2 * 70, -2),
        (2147483000, 2147483000 + 600, 1),
    ],
    ids=[
        "below_2p24",
        "at_2p24",
        "above_2p25",
        "negative_large",
        "crosses_2p24_both_signs",
        "step_3",
        "negative_step",
        "near_int32_max",
    ],
)
@pytest.mark.parametrize("tilized", [True, False], ids=["tile", "row_major"])
def test_moreh_arange_int32_large_values(start, end, step, tilized, device):
    expected = torch.arange(start, end, step, dtype=torch.int32)
    L = expected.shape[0]

    output = ttnn.moreh_arange(
        start,
        end,
        step,
        device,
        untilize_out=not tilized,
        dtype=ttnn.int32,
    )
    actual = ttnn.to_torch(output).reshape([L]).to(torch.int32)

    assert actual.shape == expected.shape
    assert torch.equal(actual, expected), f"{int((actual != expected).sum())} of {L} values differ"
