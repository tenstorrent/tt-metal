# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report

from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE
from models.experimental.stable_diffusion_xl_base.tests.pcc.test_module_tt_unet import run_unet_model

UNET_DEVICE_TEST_TOTAL_ITERATIONS = 3


@pytest.mark.parametrize(
    "input_shape, timestep_shape, encoder_shape, temb_shape, time_ids_shape",
    [
        ((1, 4, 128, 128), (1,), (1, 77, 2048), (1, 1280), (1, 6)),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("iterations", [UNET_DEVICE_TEST_TOTAL_ITERATIONS])
def test_unet(
    device,
    input_shape,
    timestep_shape,
    encoder_shape,
    temb_shape,
    time_ids_shape,
    iterations,
    is_ci_env,
    reset_seeds,
):
    run_unet_model(
        device,
        input_shape,
        timestep_shape,
        encoder_shape,
        temb_shape,
        time_ids_shape,
        is_ci_env,
        iterations=iterations,
    )


@pytest.mark.models_device_performance_bare_metal
def test_sdxl_unet_perf_device():
    expected_device_perf_cycles_per_iteration = 258_957_806

    command = f"pytest models/experimental/stable_diffusion_xl_base/tests/test_sdxl_perf.py::test_unet"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    batch_size = 1
    total_batch_size = batch_size * UNET_DEVICE_TEST_TOTAL_ITERATIONS

    inference_time_key = "AVG DEVICE KERNEL DURATION [ns]"
    post_processed_results = run_device_perf(
        command, subdir="sdxl_unet", num_iterations=1, cols=cols, batch_size=total_batch_size
    )
    expected_perf_cols = {
        inference_time_key: expected_device_perf_cycles_per_iteration * UNET_DEVICE_TEST_TOTAL_ITERATIONS
    }
    expected_results = check_device_perf(
        post_processed_results, margin=0.015, expected_perf_cols=expected_perf_cols, assert_on_fail=True
    )
    prep_device_perf_report(
        model_name=f"sdxl_unet",
        batch_size=total_batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=f"iterations={UNET_DEVICE_TEST_TOTAL_ITERATIONS}",
    )
