# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc

pytestmark = pytest.mark.use_module_device


def run_moreh_arange_test(start, end, step, device, dtype=ttnn.bfloat16, untilize_out=False, provide_output=False):
    torch_dtype = torch.int32 if dtype == ttnn.int32 else torch.bfloat16
    expected = torch.arange(start=start, end=end, step=step).to(torch_dtype)
    layout = ttnn.ROW_MAJOR_LAYOUT if untilize_out else ttnn.TILE_LAYOUT
    tt_output = (
        ttnn.from_torch(torch.zeros([1, len(expected)]), device=device, dtype=dtype, layout=layout)
        if provide_output
        else None
    )
    tt_output = ttnn.moreh_arange(start, end, step, device, output=tt_output, untilize_out=untilize_out, dtype=dtype)
    actual = ttnn.to_torch(tt_output).reshape(expected.shape)

    passing, output_pcc = comp_allclose_and_pcc(expected, actual, rtol=0.1, atol=0.1)
    assert passing, output_pcc


@pytest.mark.merge_gate
def test_moreh_arange(device):
    torch.manual_seed(0)
    # Negative fractional step; the 80 outputs span three tiles, the last one partly filled.
    run_moreh_arange_test(10.9, -13, -0.3, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "start, end, step, dtype, untilize_out",
    [
        (2.3, 15.3, 0.5, ttnn.bfloat16, False),
        (-100, 32 * 10, 1, ttnn.int32, False),
        # Row-major output uses a separate writer kernel.
        (10.9, -13, -0.3, ttnn.bfloat16, True),
    ],
    ids=["positive_step", "int32", "row_major"],
)
def test_moreh_arange_corner_cases(start, end, step, dtype, untilize_out, device):
    torch.manual_seed(0)
    run_moreh_arange_test(start, end, step, device, dtype=dtype, untilize_out=untilize_out)


@pytest.mark.merge_gate
def test_moreh_arange_provided_output(device):
    torch.manual_seed(0)
    run_moreh_arange_test(10.9, -13, -0.3, device, provide_output=True)


@pytest.mark.merge_gate
def test_moreh_arange_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_arange_test(10.9, -13, -0.3, device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Holding this tensor moves the next allocations, so the cache hit must update the output address.
    tt_placeholder = ttnn.from_torch(torch.zeros([1, 80]), device=device, layout=ttnn.TILE_LAYOUT)
    run_moreh_arange_test(10.9, -13, -0.3, device)
    assert device.num_program_cache_entries() == num_program_cache_entries
