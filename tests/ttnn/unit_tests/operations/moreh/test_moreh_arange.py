# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc
from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_arange import run_moreh_arange

pytestmark = pytest.mark.use_module_device


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "start_end_step, dtype, tilized",
    [
        # Every length leaves the last tile partly filled.
        ([10.9, -13, -0.3], "bfloat16", True),
        ([-100, 320, 1], "int32", True),
        ([2.3, 15.3, 0.5], "float32", True),
        # Row-major output uses a separate writer kernel, which clamps the partial last chunk.
        ([10.9, -13, -0.3], "bfloat16", False),
        ([-100, 320, 1], "int32", False),
        ([2.3, 15.3, 0.5], "float32", False),
        # 149 tiles: more than any device's core count, so cores write several tiles each.
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
    # Not the nightly helper: it fills the provided output with torch.empty, which can already hold the
    # expected values. NaN makes an output the op never writes fail.
    expected = torch.arange(10.9, -13, -0.3).to(torch.bfloat16)
    tt_output = ttnn.from_torch(
        torch.full([1, len(expected)], float("nan")), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    ttnn.moreh_arange(10.9, -13, -0.3, device, output=tt_output, untilize_out=False, dtype=ttnn.bfloat16)
    actual = ttnn.to_torch(tt_output).reshape(expected.shape)

    passing, output_pcc = comp_allclose_and_pcc(expected, actual, rtol=0.1, atol=0.1)
    assert passing, output_pcc
