# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from tests.ttnn.unit_tests.operations.reduce.test_topk import run_topk_test


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
    ids=[
        "BFLOAT16_B",
    ],
)
@pytest.mark.parametrize(
    "N, C, H, W, dim, k",
    ((1, 1, 32, 16 * 1024, 3, 32),),
)
@pytest.mark.parametrize(
    "sorted",
    [
        True,
    ],
)
@pytest.mark.parametrize(
    "largest",
    [
        True,
    ],
)
@pytest.mark.parametrize(
    "pass_indices_tensor",
    [
        True,
    ],
)
@pytest.mark.parametrize(
    "sub_core_grids",
    [
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ]
        ),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
def test_topk_sub_core_grids(
    N, C, H, W, dim, k, dtype, sorted, largest, device, sub_core_grids, pass_indices_tensor, galaxy_type
):
    # skip if galaxy_type is None
    if galaxy_type is None:
        pytest.skip("Test is not applicable for non-galaxy devices")
    if dim == 0 or dim == 1:
        # As of now, when we try to get top-k for dim = 0 or 1, we get following error from transpose_op.cpp's validate():
        # input_tensor.get_dtype() == DataType::BFLOAT16 || input_tensor.get_dtype() == DataType::FLOAT32
        # this is because, transpose.cpp always typecasts bf8 to bf16
        # and when dim = 0 or 1, transpose converts it into TransposeOpDim::HC & this dim doesnt support bf16 or fp32
        pytest.skip()
    run_topk_test(N, C, H, W, k, dtype, dim, sorted, largest, device, sub_core_grids, pass_indices_tensor)
