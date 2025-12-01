# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
from tests.ttnn.nightly.unit_tests.operations.pool.test_avgpool2d import run_avg_pool2d


# Cache map used for torch tensor reuse - the tensor will not be generated if a tensor of the same dimensions has already been generated
@pytest.fixture(scope="module")
def tensor_map(request):
    tensor_map = {}

    return tensor_map


parameters = {
    "avg_pool2d_short_sweep_suite": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_specs": [
            # Contains following parameters
            # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w, ceil_mode, count_include_pad]
            [1, 1056, 14, 14, 2, 2, 2, 2, 0, 0, False, True],
            [1, 128, 56, 56, 2, 2, 2, 2, 0, 0, False, True],
            [1, 160, 7, 7, 2, 2, 2, 2, 0, 0, False, True],
            [1, 192, 56, 56, 2, 2, 2, 2, 0, 0, False, True],
            [1, 256, 28, 28, 2, 2, 2, 2, 0, 0, False, True],
            [1, 384, 28, 28, 2, 2, 2, 2, 0, 0, False, True],
            [1, 512, 14, 14, 2, 2, 2, 2, 0, 0, False, True],
            [1, 640, 14, 14, 2, 2, 2, 2, 0, 0, False, True],
            [1, 896, 14, 14, 2, 2, 2, 2, 0, 0, False, True],
            [1, 1024, 17, 17, 3, 3, 1, 1, 1, 1, False, False],
            [1, 112, 14, 14, 2, 2, 2, 2, 0, 0, False, True],
            [1, 1536, 8, 8, 3, 3, 1, 1, 1, 1, False, False],
            [1, 24, 56, 56, 2, 2, 2, 2, 0, 0, False, True],
            [1, 384, 35, 35, 3, 3, 1, 1, 1, 1, False, False],
            [1, 40, 28, 28, 2, 2, 2, 2, 0, 0, False, True],
            [1, 80, 14, 14, 2, 2, 2, 2, 0, 0, False, True],
        ],
    },
}


@pytest.mark.parametrize("input_spec", parameters["avg_pool2d_short_sweep_suite"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["avg_pool2d_short_sweep_suite"]["dtype"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_ttnn_pytorch_sweep(device, tensor_map, input_spec, dtype):
    (
        in_n,
        in_c,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        ceil_mode,
        count_include_pad,
    ) = input_spec

    run_avg_pool2d(
        device=device,
        tensor_map=tensor_map,
        input_shape=[in_n, in_c, in_h, in_w],
        kernel_size=(kernel_h, kernel_w),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        ceil_mode=ceil_mode,
        divisor_override=None,
        count_include_pad=count_include_pad,
        shard_scheme=None,
        run_twice=False,
        in_dtype=dtype,
        nightly_skips=False,
    )
