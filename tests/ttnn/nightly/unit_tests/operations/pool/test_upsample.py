# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.unit_tests.operations.pool.test_upsample import upsample_multicore_common


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 640, 16, 32],
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("scale_h", [2])
@pytest.mark.parametrize("scale_w", [2])
@pytest.mark.parametrize(
    "core_range",
    [
        [((0, 0), (4, 3))],
    ],
)
def test_rectangle_core_grid_bs(device, input_shape, scale_h, scale_w, core_range):
    (torch_result, output_tensor) = upsample_multicore_common(
        device=device,
        input_shape=input_shape,
        scale_h=scale_h,
        scale_w=scale_w,
        shard_strategy=ttnn.ShardStrategy.BLOCK,
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
        core_range=core_range,
    )
    ## compare the results
    torch_result = torch_result.permute(0, 2, 3, 1)

    isequal = torch.equal(output_tensor, torch_result)

    assert isequal
