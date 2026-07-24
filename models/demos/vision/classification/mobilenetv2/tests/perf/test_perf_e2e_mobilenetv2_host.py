# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Non-trace HOST end-to-end perf test. Runs the model for several iterations: iteration 0 is
# compile + program-cache miss, later iterations are cache hits. A descriptor rebuild-on-cache-hit
# regression (see #48928) balloons the steady-state cache-hit host time, so this guards against it.

import time

import pytest
from loguru import logger

import ttnn
from models.demos.vision.classification.mobilenetv2.common import (
    MOBILENETV2_BATCH_SIZE,
    MOBILENETV2_L1_SMALL_SIZE,
    load_torch_model,
)
from models.demos.vision.classification.mobilenetv2.reference.mobilenetv2 import Mobilenetv2
from models.demos.vision.classification.mobilenetv2.tt import ttnn_mobilenetv2
from models.demos.vision.classification.mobilenetv2.tt.model_preprocessing import (
    create_mobilenetv2_input_tensors,
    create_mobilenetv2_model_parameters,
)
from models.perf.perf_utils import prep_perf_report


def get_expected_times():
    # (compile+miss, steady-state cache-hit). Calibrated on wormhole_b0; cache-hit time carries the
    # margin that a rebuild-on-cache-hit regression would blow through.
    return (
        45.0,
        0.025,
    )  # wh_b0: measured compile ~33s, cache-hit ~0.019s (Move x64 still rebuilds pre-Move-fix); tighten once Move is bound


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": MOBILENETV2_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("batch_size", [MOBILENETV2_BATCH_SIZE])
def test_perf_e2e_mobilenetv2(device, batch_size, reset_seeds, model_location_generator):
    torch_model = Mobilenetv2()
    torch_model = load_torch_model(torch_model, model_location_generator)
    torch_model.eval()

    model_parameters = create_mobilenetv2_model_parameters(torch_model, device=device)
    ttnn_model = ttnn_mobilenetv2.TtMobileNetV2(model_parameters, device, batchsize=batch_size)

    iterations = 4
    durations = []
    for _ in range(iterations):
        _, ttnn_input_tensor = create_mobilenetv2_input_tensors(batch=batch_size, input_height=224, input_width=224)
        start = time.time()
        output = ttnn_model(ttnn_input_tensor)
        output = ttnn.from_device(output)
        durations.append(time.time() - start)

    inference_and_compile_time = durations[0]
    inference_time = min(durations[1:])  # steady-state cache-hit

    expected_compile_time, expected_inference_time = get_expected_times()
    prep_perf_report(
        model_name="MobileNetV2",
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="non-trace host e2e",
        inference_time_cpu=0.0,
    )
    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Cache-hit inference time (avg): {inference_time}")
