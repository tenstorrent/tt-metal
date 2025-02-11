# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import run_for_wormhole_b0
from models.experimental.functional_yolov11.test.yolov11_perfomant import (
    run_yolov11_trace_inference,
    run_yolov11_trace_2cqs_inference,
)


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 1843200}], indirect=True)
# @pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
def test_run_yolov11_trace_inference(
    device,
    use_program_cache,
    # enable_async_mode,
    model_location_generator,
):
    run_yolov11_trace_inference(
        device,
        model_location_generator,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 3686400, "num_command_queues": 2}], indirect=True
)
def test_run_yolov11_trace_2cqs_inference(
    device,
    use_program_cache,
    # enable_async_mode,
    model_location_generator,
):
    run_yolov11_trace_2cqs_inference(
        device,
        model_location_generator,
    )
