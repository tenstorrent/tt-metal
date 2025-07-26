# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest

import ttnn
from models.demos.yolov9c.tests.perf.test_e2e_performant_detect import run_yolov9c_inference
from models.utility_functions import run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 23887872, "num_command_queues": 2}], indirect=True
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
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_e2e_performant(
    device,
    batch_size,
    resolution,
    act_dtype=ttnn.bfloat8_b,
    weight_dtype=ttnn.bfloat8_b,
):
    run_yolov9c_inference(
        device,
        batch_size,
        resolution,
        model_task="segment",
        act_dtype=ttnn.bfloat8_b,
        weight_dtype=ttnn.bfloat8_b,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 23887872, "num_command_queues": 2}], indirect=True
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
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_e2e_performant_dp(
    mesh_device,
    batch_size,
    resolution,
    act_dtype=ttnn.bfloat8_b,
    weight_dtype=ttnn.bfloat8_b,
):
    run_yolov9c_inference(
        mesh_device,
        batch_size,
        resolution,
        model_task="segment",
        act_dtype=ttnn.bfloat8_b,
        weight_dtype=ttnn.bfloat8_b,
    )
