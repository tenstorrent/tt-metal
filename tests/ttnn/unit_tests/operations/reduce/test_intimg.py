# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from tests.ttnn.unit_tests.operations.reduce.numeric_check import (
    collect_and_dump_numeric_metrics,
)
from tests.ttnn.utils_for_testing import assert_numeric_metrics


def ttnn_integral_image_cumsum_channel_last(features_nhwc):
    assert len(features_nhwc.shape) == 4, "Input tensor must be 4D"
    assert features_nhwc.shape[0] == 1, "Batch size must be 1"
    tmp = ttnn.cumsum(features_nhwc, dim=1, dtype=features_nhwc.dtype)
    ttnn.deallocate(features_nhwc)
    tmp = ttnn.move(tmp)
    return ttnn.cumsum(tmp, dim=2, dtype=features_nhwc.dtype)


def ttnn_integral_image_channel_last(features_nhwc):
    assert len(features_nhwc.shape) == 4, "Input tensor must be 4D"
    assert features_nhwc.shape[0] == 1, "Batch size must be 1"
    return ttnn.experimental.intimg(features_nhwc)


@pytest.mark.parametrize(
    "input_shape_nhwc",
    [
        # fmt: off
        ([1, 12, 40, 256]),
        ([1, 24, 80, 256]),
        ([1, 48, 160, 256]),
        ([1, 96, 160, 256])
    ],
    ids=["OFT32", "OFT16", "OFT8", "big_one"],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bfloat16", "float32"])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["DRAM"])
def test_cumsum_channel_last(device, input_shape_nhwc, dtype, memory_config):
    torch.manual_seed(0)

    if dtype != ttnn.bfloat16 and dtype != ttnn.float32:
        pytest.skip("Unsupported dtype")

    torch_input_tensor = torch.relu(torch.rand(input_shape_nhwc, dtype=torch.bfloat16))
    # golden
    torch_output_tensor = torch.cumsum(torch.cumsum(torch_input_tensor, dim=1), dim=2)

    # reference
    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype, memory_config=memory_config
    )
    output_tensor = ttnn_integral_image_cumsum_channel_last(input_tensor)
    ttnn_output_tensor = ttnn.to_torch(output_tensor)
    # Collect numeric metrics and dump to CSV using reusable function
    test_name = (
        f"test_cumsum_channel_last[input_shape_nhwc={input_shape_nhwc},dtype={dtype},memory_config={memory_config}]"
    )
    collect_and_dump_numeric_metrics(
        torch_output_tensor,
        ttnn_output_tensor,
        test_name=test_name,
        csv_filename="test_intimg_numeric_results.csv",
        test_params=None,
    )

    # Thresholds from test_intimg_numeric_results.csv (test_cumsum_channel_last); headroom on extrema.
    if dtype == ttnn.bfloat16:
        assert_numeric_metrics(
            torch_output_tensor,
            ttnn_output_tensor,
            pcc_threshold=0.9983,
            rtol=0.156,
            atol=1180.0,
            frobenius_threshold=0.0496,
        )
    else:
        assert_numeric_metrics(
            torch_output_tensor,
            ttnn_output_tensor,
            pcc_threshold=0.99989,
            rtol=0.0145,
            atol=66.0,
            frobenius_threshold=0.0035,
        )

    # experimental intimg
    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype, memory_config=memory_config
    )
    output_tensor_2 = ttnn_integral_image_channel_last(input_tensor)
    ttnn_output_tensor_2 = ttnn.to_torch(output_tensor_2)
    # Collect numeric metrics and dump to CSV using reusable function
    test_name_2 = (
        f"test_intimg_channel_last[input_shape_nhwc={input_shape_nhwc},dtype={dtype},memory_config={memory_config}]"
    )
    collect_and_dump_numeric_metrics(
        torch_output_tensor,
        ttnn_output_tensor_2,
        test_name=test_name_2,
        csv_filename="test_intimg_numeric_results.csv",
        test_params=None,
    )
    # Thresholds from test_intimg_numeric_results.csv (test_intimg_channel_last).
    if dtype == ttnn.bfloat16:
        assert_numeric_metrics(
            torch_output_tensor,
            ttnn_output_tensor_2,
            pcc_threshold=0.99988,
            rtol=0.039,
            atol=131.0,
            frobenius_threshold=0.011,
        )
    else:
        assert_numeric_metrics(
            torch_output_tensor,
            ttnn_output_tensor_2,
            pcc_threshold=0.99999,
            rtol=0.0111,
            atol=33.0,
            frobenius_threshold=0.00235,
        )
