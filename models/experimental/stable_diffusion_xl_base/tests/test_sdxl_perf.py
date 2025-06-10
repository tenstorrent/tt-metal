# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report


@pytest.mark.models_device_performance_bare_metal
def test_sdxl_unet_perf_device():
    command = f"pytest tests/nightly/single_card/stable_diffusion_xl_base/test_module_tt_unet.py::test_unet[device_params0-transformer_weights_dtype0-conv_weights_dtype0-input_shape0-timestep_shape0-encoder_shape0-temb_shape0-time_ids_shape0]"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL DURATION [ns]"
    post_processed_results = run_device_perf(command, subdir="sdxl_unet", num_iterations=3, cols=cols, batch_size=1)
    expected_perf_cols = {inference_time_key: 478272848}
    expected_results = check_device_perf(
        post_processed_results, margin=0.015, expected_perf_cols=expected_perf_cols, assert_on_fail=True
    )
    prep_device_perf_report(
        model_name=f"sdxl_unet_single_iter",
        batch_size=1,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
