# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole


def run_sam3_vit_backbone(device, batch_size, sam3_vit_backbone):
    input_shape = (batch_size, 3, 1024, 1024)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)

    inference_iter_count = 10
    t0 = time.time()
    for _ in range(inference_iter_count):
        with torch.no_grad():
            output = sam3_vit_backbone(torch_input_tensor)
    t1 = time.time()

    inference_time_avg = round((t1 - t0) / inference_iter_count, 6)
    fps = round(batch_size / inference_time_avg)
    logger.info(
        f"ttnn_sam3_vit_backbone_1024x1024_batch_size_{batch_size}. "
        f"One inference iteration time (sec): {inference_time_avg}, FPS: {fps}"
    )


@run_for_blackhole()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size",
    ((1),),
)
@pytest.mark.models_performance_bare_metal
def test_perf_sam3_vit_backbone(device, batch_size, sam3_vit_backbone):
    return run_sam3_vit_backbone(device, batch_size, sam3_vit_backbone)
