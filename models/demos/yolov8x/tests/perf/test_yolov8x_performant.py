# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest

from models.demos.yolov8x.common import YOLOV8X_L1_SMALL_SIZE
from models.demos.yolov8x.runner.yolov8x_performant import (
    run_yolov8x_inference,
    run_yolov8x_trace_2cqs_inference,
    run_yolov8x_trace_inference,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV8X_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("device_batch_size", [(1)])
def test_run_yolov8x_inference(
    device,
    device_batch_size,
):
    run_yolov8x_inference(
        device,
        device_batch_size,
    )


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": YOLOV8X_L1_SMALL_SIZE, "trace_region_size": 3686400}], indirect=True
)
@pytest.mark.parametrize("device_batch_size", [(1)])
def test_run_yolov8x_trace_inference(
    device,
    device_batch_size,
):
    run_yolov8x_trace_inference(
        device,
        device_batch_size,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV8X_L1_SMALL_SIZE, "trace_region_size": 3686400, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize("device_batch_size", [(1)])
def test_run_yolov8x_trace_2cq_inference(
    device,
    device_batch_size,
):
    run_yolov8x_trace_2cqs_inference(
        device=device,
        device_batch_size=device_batch_size,
    )
