# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import time
import torch
from loguru import logger

from models.utility_functions import run_for_wormhole_b0
from models.demos.yolov4.tests.yolov4_perfomant_webdemo import Yolov4Trace2CQ


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 40960, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
@pytest.mark.parametrize(
    "resolution",
    [
        (320, 320),
        (640, 640),
    ],
)
def test_run_yolov4_trace_2cqs_inference(
    device,
    use_program_cache,
    batch_size,
    act_dtype,
    weight_dtype,
    enable_async_mode,
    model_location_generator,
    resolution,
):
    yolov4_trac2_2cq = Yolov4Trace2CQ()

    yolov4_trac2_2cq.initialize_yolov4_trace_2cqs_inference(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        model_location_generator=None,
    )

    inference_iter_count = 10
    inference_time_iter = []
    for iter in range(0, inference_iter_count):
        input_shape = (1, 3, *resolution)

        torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
        n, c, h, w = torch_input_tensor.shape
        torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)
        torch_input_tensor = torch_input_tensor.reshape(1, 1, h * w * n, c)
        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_inputs_host = ttnn.pad(tt_inputs_host, [1, 1, n * h * w, 16], [0, 0, 0, 0], 0)

        t0 = time.time()
        output = yolov4_trac2_2cq.execute_yolov4_trace_2cqs_inference(tt_inputs_host)
        t1 = time.time()
        inference_time_iter.append(t1 - t0)
    yolov4_trac2_2cq.release_yolov4_trace_2cqs_inference()
    inference_time_avg = round(sum(inference_time_iter) / len(inference_time_iter), 6)
    # print(batch_size/inference_time_avg)
    logger.info(
        f"ttnn_yolov4_batch_size: {batch_size}, resolution: {resolution}. One inference iteration time (sec): {inference_time_avg}, FPS: {round(batch_size/inference_time_avg)}"
    )
