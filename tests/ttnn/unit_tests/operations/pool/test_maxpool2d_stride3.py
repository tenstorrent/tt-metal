# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
from tests.ttnn.nightly.unit_tests.operations.pool.test_maxpool2d import run_max_pool2d


@pytest.fixture(scope="module")
def tensor_map(request):
    tensor_map = {}
    return tensor_map


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_max_pool2d_stride_3x3_no_padding(device, tensor_map):
    run_max_pool2d(
        [1, 32, 9, 9],
        (3, 3),
        (0, 0),
        (3, 3),
        (1, 1),
        device,
        tensor_map,
        ttnn.bfloat16,
        shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=False,
        nightly_skips=False,
    )
