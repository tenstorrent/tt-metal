# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.yunet.common import YUNET_L1_SMALL_SIZE, YUNET_TRACE_REGION_SIZE
from models.experimental.yunet.runner.performant_runner import YunetPerformantRunner


def run_yunet_e2e(device, input_height=320, input_width=320):
    """Run YUNet E2E performance test with trace+2CQ."""
    input_shape = (1, input_height, input_width, 3)  # NHWC format
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)

    performant_runner = YunetPerformantRunner(
        device,
        input_height=input_height,
        input_width=input_width,
    )

    num_iter = 1000
    t0 = time.time()
    for _ in range(num_iter):
        output_tensors = performant_runner.run(torch_input_tensor)
    ttnn.synchronize_device(device)
    t1 = time.time()

    performant_runner.release()

    inference_time_avg = round((t1 - t0) / num_iter, 6)
    fps = round(1.0 / inference_time_avg)
    logger.info(
        f"ttnn_yunet_{input_height}x{input_width}. One inference iteration time (sec): {inference_time_avg}, FPS: {fps}"
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YUNET_L1_SMALL_SIZE, "trace_region_size": YUNET_TRACE_REGION_SIZE, "num_command_queues": 2}],
    indirect=True,
)
def test_yunet_e2e_performant(device):
    run_yunet_e2e(device, input_height=320, input_width=320)
