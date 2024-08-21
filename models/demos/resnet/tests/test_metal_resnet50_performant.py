# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from models.demos.resnet.tests.test_metal_resnet50 import (
    run_resnet50_inference,
    run_model,
    run_trace_model,
    run_2cq_model,
    run_trace_2cq_model,
)
from models.utility_functions import skip_for_wormhole_b0


@skip_for_wormhole_b0("This test is not supported on WHB0, please use the TTNN version.")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("batch_size", [20], ids=["batch_20"])
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b],
    ids=["weights_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat8_b],
    ids=["activations_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "math_fidelity",
    [ttnn.MathFidelity.LoFi],
    ids=["LoFi"],
)
@pytest.mark.parametrize("enable_async_mode", [True, False], indirect=True)
def test_run_resnet50_inference(
    device,
    use_program_cache,
    batch_size,
    weights_dtype,
    activations_dtype,
    math_fidelity,
    imagenet_sample_input,
    enable_async_mode,
    model_location_generator,
):
    run_resnet50_inference(
        device,
        batch_size,
        weights_dtype,
        activations_dtype,
        math_fidelity,
        imagenet_sample_input,
        run_model,
        model_location_generator,
    )


@skip_for_wormhole_b0("This test is not supported on WHB0, please use the TTNN version.")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 1500000}], indirect=True)
@pytest.mark.parametrize("batch_size", [20], ids=["batch_20"])
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b],
    ids=["weights_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat8_b],
    ids=["activations_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "math_fidelity",
    [ttnn.MathFidelity.LoFi],
    ids=["LoFi"],
)
@pytest.mark.parametrize("enable_async_mode", [True, False], indirect=True)
def test_run_resnet50_trace_inference(
    device,
    use_program_cache,
    batch_size,
    weights_dtype,
    activations_dtype,
    math_fidelity,
    imagenet_sample_input,
    enable_async_mode,
    model_location_generator,
):
    run_resnet50_inference(
        device,
        batch_size,
        weights_dtype,
        activations_dtype,
        math_fidelity,
        imagenet_sample_input,
        run_trace_model,
        model_location_generator,
    )


@skip_for_wormhole_b0("This test is not supported on WHB0, please use the TTNN version.")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "num_hw_cqs": 2}], indirect=True)
@pytest.mark.parametrize("batch_size", [20], ids=["batch_20"])
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b],
    ids=["weights_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat8_b],
    ids=["activations_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "math_fidelity",
    [ttnn.MathFidelity.LoFi],
    ids=["LoFi"],
)
@pytest.mark.parametrize("enable_async_mode", [True, False], indirect=True)
def test_run_resnet50_2cqs_inference(
    device,
    use_program_cache,
    batch_size,
    weights_dtype,
    activations_dtype,
    math_fidelity,
    imagenet_sample_input,
    enable_async_mode,
    model_location_generator,
):
    run_resnet50_inference(
        device,
        batch_size,
        weights_dtype,
        activations_dtype,
        math_fidelity,
        imagenet_sample_input,
        run_2cq_model,
        model_location_generator,
    )


@skip_for_wormhole_b0("This test is not supported on WHB0, please use the TTNN version.")
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "num_hw_cqs": 2, "trace_region_size": 1500000}], indirect=True
)
@pytest.mark.parametrize("batch_size", [20], ids=["batch_20"])
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b],
    ids=["weights_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat8_b],
    ids=["activations_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "math_fidelity",
    [ttnn.MathFidelity.LoFi],
    ids=["LoFi"],
)
@pytest.mark.parametrize("enable_async_mode", [True, False], indirect=True)
def test_run_resnet50_trace_2cqs_inference(
    device,
    use_program_cache,
    batch_size,
    weights_dtype,
    activations_dtype,
    math_fidelity,
    imagenet_sample_input,
    enable_async_mode,
    model_location_generator,
):
    run_resnet50_inference(
        device,
        batch_size,
        weights_dtype,
        activations_dtype,
        math_fidelity,
        imagenet_sample_input,
        run_trace_2cq_model,
        model_location_generator,
    )
