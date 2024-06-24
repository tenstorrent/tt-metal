# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.resnet.tests.test_perf_resnet import run_perf_resnet
from models.utility_functions import skip_for_wormhole_b0


@skip_for_wormhole_b0(reason_str="Not tested on single WH")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "num_hw_cqs": 2}], indirect=True)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time",
    ((20, 0.0042, 16),),
)
def test_perf_2cqs_bare_metal(
    device,
    use_program_cache,
    batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
):
    run_perf_resnet(
        batch_size, expected_inference_time, expected_compile_time, hf_cat_image_sample_input, device, "resnet50_2cqs"
    )


@skip_for_wormhole_b0(reason_str="Not tested on single WH")
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "num_hw_cqs": 2, "trace_region_size": 1332224}], indirect=True
)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time",
    ((20, 0.0042, 16),),
)
def test_perf_trace_2cqs_bare_metal(
    device,
    use_program_cache,
    batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
):
    run_perf_resnet(
        batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        device,
        "resnet50_trace_2cqs",
    )
