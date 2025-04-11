# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.utility_functions import run_for_wormhole_b0
from models.experimental.functional_vgg_unet.tests.vgg_unet_performant import (
    run_vgg_unet_trace_inference,
    run_vgg_unet_trace_2cqs_inference,
    run_vgg_unet_inference,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_run_vgg_unet_inference(device, use_program_cache, model_location_generator):
    run_vgg_unet_inference(device, model_location_generator)


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "trace_region_size": 1843200}], indirect=True)
@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
def test_run_vgg_unet_trace_inference(
    device,
    use_program_cache,
    enable_async_mode,
    model_location_generator,
):
    run_vgg_unet_trace_inference(
        device,
        model_location_generator,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "trace_region_size": 3686400, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
def test_run_vgg_unet_trace_2cqs_inference(
    device,
    use_program_cache,
    enable_async_mode,
    model_location_generator,
):
    run_vgg_unet_trace_2cqs_inference(
        device,
        model_location_generator,
    )
