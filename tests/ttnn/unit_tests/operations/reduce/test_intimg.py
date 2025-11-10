# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc
from loguru import logger


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
        ([1, 48, 160, 256])
    ],
    ids=["OFT32", "OFT16", "OFT8"],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bfloat16"])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["DRAM"])
def test_cumsum_channel_last(device, input_shape_nhwc, dtype, memory_config):
    torch.manual_seed(0)

    if dtype == ttnn.bfloat16:
        torch_input_tensor = torch.relu(torch.rand(input_shape_nhwc, dtype=torch.bfloat16))
    else:
        pytest.skip("Unsupported dtype")

    # golden
    torch_output_tensor = torch.cumsum(torch.cumsum(torch_input_tensor, dim=1), dim=2)

    # reference  (test hangs if removed)
    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype, memory_config=memory_config
    )
    output_tensor = ttnn_integral_image_cumsum_channel_last(input_tensor)
    ttnn_output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, ttnn_output_tensor, pcc=0.999)

    # check L1 just in case
    buffers = ttnn._ttnn.reports.get_buffers(device)
    l1_buffers = [buf for buf in buffers if buf.buffer_type == ttnn.BufferType.L1]
    for i, buf in enumerate(l1_buffers):
        logger.warning(f"L1 Buffer {i}: addr={buf.address}, size={buf.max_size_per_bank}, layout={buf.buffer_layout}")

    # experimental intimg
    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype, memory_config=memory_config
    )
    output_tensor_2 = ttnn_integral_image_channel_last(input_tensor)
    ttnn_output_tensor_2 = ttnn.to_torch(output_tensor_2)
    assert_with_pcc(torch_output_tensor, ttnn_output_tensor_2, pcc=0.999)
