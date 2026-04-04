# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.yolov8l.common import YOLOV8L_L1_SMALL_SIZE, YOLOV8L_TRACE_REGION_SIZE_1CQ
from models.demos.yolov8l.runner.yolov8l_performant import (
    run_yolov8l_inference,
    run_yolov8l_trace_2cqs_inference,
    run_yolov8l_trace_inference,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV8L_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("device_batch_size", [(1)])
def test_run_yolov8l_inference(
    device,
    device_batch_size,
):
    run_yolov8l_inference(
        device,
        device_batch_size,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV8L_L1_SMALL_SIZE, "trace_region_size": YOLOV8L_TRACE_REGION_SIZE_1CQ}],
    indirect=True,
)
@pytest.mark.parametrize("device_batch_size", [(1)])
def test_run_yolov8l_trace_inference(
    device,
    device_batch_size,
):
    run_yolov8l_trace_inference(
        device,
        device_batch_size,
    )


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": YOLOV8L_L1_SMALL_SIZE,
            "trace_region_size": YOLOV8L_TRACE_REGION_SIZE_1CQ,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_batch_size", [(1)])
def test_run_yolov8l_trace_2cq_inference(
    device,
    device_batch_size,
):
    run_yolov8l_trace_2cqs_inference(
        device=device,
        device_batch_size=device_batch_size,
    )
