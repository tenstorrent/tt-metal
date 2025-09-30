# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest

import ttnn
from models.common.utility_functions import run_for_wormhole_b0
from models.demos.yolov9c.common import YOLOV9C_L1_SMALL_SIZE
from models.demos.yolov9c.tests.perf.test_e2e_performant_detect import run_yolov9c_inference


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV9C_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size, resolution",
    [
        (
            1,
            (640, 640),
        ),
    ],
)
@pytest.mark.parametrize(
    "expected_inference_throughput",
    [
        50 if ttnn.get_num_devices() < 2 else 45,
    ],
)
def test_e2e_performant(model_location_generator, device, batch_size, resolution, expected_inference_throughput):
    run_yolov9c_inference(
        model_location_generator,
        device,
        batch_size,
        resolution,
        model_task="segment",
        expected_inference_throughput=expected_inference_throughput,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV9C_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size, resolution",
    [
        (
            1,
            (640, 640),
        ),
    ],
)
@pytest.mark.parametrize(
    "expected_inference_throughput",
    [
        92,
    ],
)
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_e2e_performant_dp(
    model_location_generator, mesh_device, batch_size, resolution, expected_inference_throughput
):
    if ttnn.get_num_devices() < 2:
        pytest.skip()

    run_yolov9c_inference(
        model_location_generator,
        mesh_device,
        batch_size,
        resolution,
        model_task="segment",
        expected_inference_throughput=expected_inference_throughput,
    )
