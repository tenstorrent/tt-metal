# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report

from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE
from models.experimental.stable_diffusion_xl_base.tests.pcc.test_module_tt_unet import run_unet_model
from models.experimental.stable_diffusion_xl_base.refiner.tests.pcc.test_module_tt_unet import run_refiner_unet_model

VAE_DEVICE_TEST_TOTAL_ITERATIONS = 1
UNET_DEVICE_TEST_TOTAL_ITERATIONS = 1


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
    is_ci_v2_env,
    model_location_generator,
    reset_seeds,
):
    run_unet_model(
        device,
        input_shape,
        timestep_shape,
        encoder_shape,
        temb_shape,
        time_ids_shape,
        debug_mode=False,
        is_ci_env=is_ci_env,
        is_ci_v2_env=is_ci_v2_env,
        model_location_generator=model_location_generator,
        iterations=iterations,
    )


@pytest.mark.models_device_performance_bare_metal
def test_sdxl_unet_perf_device():
    expected_device_perf_cycles_per_iteration = 200_766_079

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


@pytest.mark.parametrize(
    "input_shape, timestep_shape, encoder_shape, temb_shape, time_ids_shape",
    [
        ((1, 4, 128, 128), (1,), (1, 77, 1280), (1, 1280), (1, 5)),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("iterations", [UNET_DEVICE_TEST_TOTAL_ITERATIONS])
def test_refiner_unet(
    device,
    input_shape,
    timestep_shape,
    encoder_shape,
    temb_shape,
    time_ids_shape,
    iterations,
    is_ci_env,
    is_ci_v2_env,
    model_location_generator,
    reset_seeds,
):
    run_refiner_unet_model(
        device,
        input_shape,
        timestep_shape,
        encoder_shape,
        temb_shape,
        time_ids_shape,
        debug_mode=False,
        is_ci_env=is_ci_env,
        is_ci_v2_env=is_ci_v2_env,
        model_location_generator=model_location_generator,
        iterations=iterations,
    )


@pytest.mark.models_device_performance_bare_metal
def test_sdxl_refiner_unet_perf_device():
    expected_device_perf_cycles_per_iteration = 640_816_818

    command = f"pytest models/experimental/stable_diffusion_xl_base/tests/test_sdxl_perf.py::test_refiner_unet"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    batch_size = 1
    total_batch_size = batch_size * UNET_DEVICE_TEST_TOTAL_ITERATIONS

    inference_time_key = "AVG DEVICE KERNEL DURATION [ns]"
    post_processed_results = run_device_perf(
        command, subdir="sdxl_refiner_unet", num_iterations=1, cols=cols, batch_size=total_batch_size
    )
    expected_perf_cols = {
        inference_time_key: expected_device_perf_cycles_per_iteration * UNET_DEVICE_TEST_TOTAL_ITERATIONS
    }
    expected_results = check_device_perf(
        post_processed_results, margin=0.015, expected_perf_cols=expected_perf_cols, assert_on_fail=True
    )
    prep_device_perf_report(
        model_name=f"sdxl_refiner_unet",
        batch_size=total_batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=f"iterations={UNET_DEVICE_TEST_TOTAL_ITERATIONS}",
    )


@pytest.mark.models_device_performance_bare_metal
def test_sdxl_vae_decode_perf_device():
    expected_device_perf_cycles_per_iteration = 932_430_106
    command = f"pytest models/experimental/stable_diffusion_xl_base/vae/tests/pcc/test_module_tt_autoencoder_kl.py::test_vae -k 'test_decode'"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    batch_size = 1

    inference_time_key = "AVG DEVICE KERNEL DURATION [ns]"
    post_processed_results = run_device_perf(
        command, subdir="sdxl_vae", num_iterations=VAE_DEVICE_TEST_TOTAL_ITERATIONS, cols=cols, batch_size=batch_size
    )
    expected_perf_cols = {inference_time_key: expected_device_perf_cycles_per_iteration}
    expected_results = check_device_perf(
        post_processed_results, margin=0.015, expected_perf_cols=expected_perf_cols, assert_on_fail=True
    )
    prep_device_perf_report(
        model_name=f"sdxl_vae_decode",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )


@pytest.mark.models_device_performance_bare_metal
def test_sdxl_vae_encode_perf_device():
    expected_device_perf_cycles_per_iteration = 492_473_727
    command = f"pytest models/experimental/stable_diffusion_xl_base/vae/tests/pcc/test_module_tt_autoencoder_kl.py::test_vae -k 'test_encode'"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    batch_size = 1

    inference_time_key = "AVG DEVICE KERNEL DURATION [ns]"
    post_processed_results = run_device_perf(
        command, subdir="sdxl_vae", num_iterations=VAE_DEVICE_TEST_TOTAL_ITERATIONS, cols=cols, batch_size=batch_size
    )
    expected_perf_cols = {inference_time_key: expected_device_perf_cycles_per_iteration}
    expected_results = check_device_perf(
        post_processed_results, margin=0.015, expected_perf_cols=expected_perf_cols, assert_on_fail=True
    )
    prep_device_perf_report(
        model_name=f"sdxl_vae_encode",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
