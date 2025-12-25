# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import pytest

from models.perf.device_perf_utils import run_model_device_perf_test

from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE
from models.experimental.stable_diffusion_xl_base.tests.pcc.test_module_tt_unet import run_unet_model
from models.experimental.stable_diffusion_xl_base.refiner.tests.pcc.test_module_tt_unet import run_refiner_unet_model

VAE_DEVICE_TEST_TOTAL_ITERATIONS = 1
UNET_DEVICE_TEST_TOTAL_ITERATIONS = 1
CLIP_ENCODER_DEVICE_TEST_TOTAL_ITERATIONS = 1


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


@pytest.mark.parametrize(
    "command, expected_device_perf_ns_per_iteration, subdir, model_name, num_iterations, batch_size, margin, comments",
    [
        (
            "pytest models/experimental/stable_diffusion_xl_base/tests/test_sdxl_perf.py::test_unet",
            191_651_771 * UNET_DEVICE_TEST_TOTAL_ITERATIONS,
            "sdxl_unet",
            "sdxl_unet",
            1,
            1 * UNET_DEVICE_TEST_TOTAL_ITERATIONS,
            0.015,
            f"iterations={UNET_DEVICE_TEST_TOTAL_ITERATIONS}",
        ),
        (
            "pytest models/experimental/stable_diffusion_xl_base/tests/test_sdxl_perf.py::test_refiner_unet",
            610_108_504 * UNET_DEVICE_TEST_TOTAL_ITERATIONS,
            "sdxl_refiner_unet",
            "sdxl_refiner_unet",
            1,
            1 * UNET_DEVICE_TEST_TOTAL_ITERATIONS,
            0.06,
            f"iterations={UNET_DEVICE_TEST_TOTAL_ITERATIONS}",
        ),
        (
            "pytest models/experimental/stable_diffusion_xl_base/vae/tests/pcc/test_module_tt_autoencoder_kl.py::test_vae -k 'test_decode'",
            691_874_774,
            "sdxl_vae",
            "sdxl_vae_decode",
            VAE_DEVICE_TEST_TOTAL_ITERATIONS,
            1,
            0.015,
            "",
        ),
        (
            "pytest models/experimental/stable_diffusion_xl_base/vae/tests/pcc/test_module_tt_autoencoder_kl.py::test_vae -k 'test_encode'",
            348_815_743,
            "sdxl_vae",
            "sdxl_vae_encode",
            VAE_DEVICE_TEST_TOTAL_ITERATIONS,
            1,
            0.015,
            "",
        ),
        (
            "pytest models/experimental/stable_diffusion_xl_base/tests/pcc/test_sdxl_clip_encoders.py::test_clip_encoder -k 'encoder_1'",
            13_112_562,
            "sdxl_clip_encoder_1",
            "sdxl_clip_encoder_1",
            CLIP_ENCODER_DEVICE_TEST_TOTAL_ITERATIONS,
            1,
            0.015,
            "",
        ),
        (
            "pytest models/experimental/stable_diffusion_xl_base/tests/pcc/test_sdxl_clip_encoders.py::test_clip_encoder -k 'encoder_2'",
            63_023_709,  # Note: this is an average value of 5 test runs due to high variability
            "sdxl_clip_encoder_2",
            "sdxl_clip_encoder_2",
            CLIP_ENCODER_DEVICE_TEST_TOTAL_ITERATIONS,
            1,
            0.015,
            "",
        ),
    ],
    ids=[
        "test_sdxl_unet",
        "test_sdxl_refiner_unet",
        "test_sdxl_vae_decode",
        "test_sdxl_vae_encode",
        "test_sdxl_clip_encoder_1",
        "test_sdxl_clip_encoder_2",
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_sdxl_perf_device(
    command, expected_device_perf_ns_per_iteration, subdir, model_name, num_iterations, batch_size, margin, comments
):
    os.environ["TT_MM_THROTTLE_PERF"] = "5"
    run_model_device_perf_test(
        command=command,
        expected_device_perf_ns_per_iteration=expected_device_perf_ns_per_iteration,
        subdir=subdir,
        model_name=model_name,
        num_iterations=num_iterations,
        batch_size=batch_size,
        margin=margin,
        comments=comments,
    )
