# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import pytest

from models.perf.device_perf_utils import run_model_device_perf_test

from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE
from models.experimental.stable_diffusion_xl_base.tests.pcc.test_module_tt_unet import run_unet_model
from models.experimental.stable_diffusion_xl_base.refiner.tests.pcc.test_module_tt_unet import run_refiner_unet_model
from models.common.utility_functions import is_wormhole_b0, is_blackhole

VAE_DEVICE_TEST_TOTAL_ITERATIONS = 1
UNET_DEVICE_TEST_TOTAL_ITERATIONS = 1
CLIP_ENCODER_DEVICE_TEST_TOTAL_ITERATIONS = 1


@pytest.mark.parametrize(
    "image_resolution, input_shape, timestep_shape, encoder_shape, temb_shape, time_ids_shape, pcc",
    [
        # 1024x1024 image resolution
        ((1024, 1024), (1, 4, 128, 128), (1,), (1, 77, 2048), (1, 1280), (1, 6), 0.9968),
        # 512x512 image resolution
        ((512, 512), (1, 4, 64, 64), (1,), (1, 77, 2048), (1, 1280), (1, 6), 0.9958),
    ],
    ids=["1024x1024", "512x512"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("iterations", [UNET_DEVICE_TEST_TOTAL_ITERATIONS])
def test_unet(
    device,
    image_resolution,
    input_shape,
    timestep_shape,
    encoder_shape,
    temb_shape,
    time_ids_shape,
    pcc,
    iterations,
    is_ci_env,
    is_ci_v2_env,
    sdxl_base_unet_location,
    sdxl_inpainting_unet_location,
    reset_seeds,
):
    run_unet_model(
        device,
        image_resolution,
        input_shape,
        timestep_shape,
        encoder_shape,
        temb_shape,
        time_ids_shape,
        pcc,
        debug_mode=False,
        is_ci_env=is_ci_env,
        is_ci_v2_env=is_ci_v2_env,
        sdxl_base_unet_location=sdxl_base_unet_location,
        sdxl_inpainting_unet_location=sdxl_inpainting_unet_location,
        iterations=iterations,
    )


@pytest.mark.parametrize(
    "image_resolution, input_shape, timestep_shape, encoder_shape, temb_shape, time_ids_shape, pcc",
    [
        # 1024x1024 image resolution
        ((1024, 1024), (1, 4, 128, 128), (1,), (1, 77, 1280), (1, 1280), (1, 5), 0.997),
        # 512x512 image resolution
        ((512, 512), (1, 4, 64, 64), (1,), (1, 77, 1280), (1, 1280), (1, 5), 0.997),
    ],
    ids=["1024x1024", "512x512"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("iterations", [UNET_DEVICE_TEST_TOTAL_ITERATIONS])
def test_refiner_unet(
    device,
    image_resolution,
    input_shape,
    timestep_shape,
    encoder_shape,
    temb_shape,
    time_ids_shape,
    pcc,
    iterations,
    is_ci_env,
    is_ci_v2_env,
    sdxl_refiner_unet_location,
    reset_seeds,
):
    run_refiner_unet_model(
        device,
        image_resolution,
        input_shape,
        timestep_shape,
        encoder_shape,
        temb_shape,
        time_ids_shape,
        pcc,
        debug_mode=False,
        is_ci_env=is_ci_env,
        is_ci_v2_env=is_ci_v2_env,
        sdxl_refiner_unet_location=sdxl_refiner_unet_location,
        iterations=iterations,
    )


# Device-specific performance expectations (ns per iteration)
# Format: {test_id: {"wormhole": perf_ns, "blackhole": perf_ns}}
DEVICE_PERF_EXPECTATIONS = {
    "unet_1024x1024": {
        "wormhole": 191_651_771 * UNET_DEVICE_TEST_TOTAL_ITERATIONS,
        "blackhole": 119_138_750 * UNET_DEVICE_TEST_TOTAL_ITERATIONS,
    },
    "unet_512x512": {
        "wormhole": 91_463_635 * UNET_DEVICE_TEST_TOTAL_ITERATIONS,
        "blackhole": None,  # Only 1024x1024 tested on Blackhole
    },
    "refiner_unet_1024x1024": {
        "wormhole": 244_107_203 * UNET_DEVICE_TEST_TOTAL_ITERATIONS,
        "blackhole": 131_795_269 * UNET_DEVICE_TEST_TOTAL_ITERATIONS,
    },
    "refiner_unet_512x512": {
        "wormhole": 79_843_092 * UNET_DEVICE_TEST_TOTAL_ITERATIONS,
        "blackhole": None,  # Only 1024x1024 tested on Blackhole
    },
    "vae_decode_1024x1024": {
        "wormhole": 663_083_865,
        "blackhole": 708_023_460,  # Used to be 369_666_097, group_norm welford disabled caused regression
    },
    "vae_decode_512x512": {
        "wormhole": 171_560_642,
        "blackhole": None,  # Only 1024x1024 tested on Blackhole
    },
    "vae_encode_1024x1024": {
        "wormhole": 324_271_938,
        "blackhole": 170_093_216,
    },
    "vae_encode_512x512": {
        "wormhole": 83_537_085,
        "blackhole": None,  # Only 1024x1024 tested on Blackhole
    },
    "clip_encoder_1": {
        "wormhole": 13_112_562,
        "blackhole": 6_795_180,
    },
    "clip_encoder_2": {
        "wormhole": 63_591_763,  # Note: this is an average value of 30 test runs due to high variability
        "blackhole": 31_220_061,
    },
}


def get_device_perf(test_id):
    """Get expected performance for the current device."""
    perfs = DEVICE_PERF_EXPECTATIONS.get(test_id, {})
    if is_blackhole():
        return perfs.get("blackhole")
    elif is_wormhole_b0():
        return perfs.get("wormhole")
    return None


@pytest.mark.parametrize(
    "test_id, command, subdir, model_name, num_iterations, batch_size, margin, comments",
    [
        (
            "unet_1024x1024",
            'pytest models/experimental/stable_diffusion_xl_base/tests/test_sdxl_perf.py::test_unet -k "1024x1024"',
            "sdxl_unet_1024x1024",
            "sdxl_unet_1024x1024",
            1,
            1 * UNET_DEVICE_TEST_TOTAL_ITERATIONS,
            0.015,
            f"iterations={UNET_DEVICE_TEST_TOTAL_ITERATIONS}",
        ),
        (
            "unet_512x512",
            'pytest models/experimental/stable_diffusion_xl_base/tests/test_sdxl_perf.py::test_unet -k "512x512"',
            "sdxl_unet_512x512",
            "sdxl_unet_512x512",
            1,
            1 * UNET_DEVICE_TEST_TOTAL_ITERATIONS,
            0.015,
            f"iterations={UNET_DEVICE_TEST_TOTAL_ITERATIONS}",
        ),
        (
            "refiner_unet_1024x1024",
            'pytest models/experimental/stable_diffusion_xl_base/tests/test_sdxl_perf.py::test_refiner_unet -k "1024x1024"',
            "sdxl_refiner_unet_1024x1024",
            "sdxl_refiner_unet_1024x1024",
            1,
            1 * UNET_DEVICE_TEST_TOTAL_ITERATIONS,
            0.06,
            f"iterations={UNET_DEVICE_TEST_TOTAL_ITERATIONS}",
        ),
        (
            "refiner_unet_512x512",
            'pytest models/experimental/stable_diffusion_xl_base/tests/test_sdxl_perf.py::test_refiner_unet -k "512x512"',
            "sdxl_refiner_unet_512x512",
            "sdxl_refiner_unet_512x512",
            1,
            1 * UNET_DEVICE_TEST_TOTAL_ITERATIONS,
            0.06,
            f"iterations={UNET_DEVICE_TEST_TOTAL_ITERATIONS}",
        ),
        (
            "vae_decode_1024x1024",
            "pytest models/experimental/stable_diffusion_xl_base/vae/tests/pcc/test_module_tt_autoencoder_kl.py::test_vae -k 'test_1024x1024_decode'",
            "sdxl_vae",
            "sdxl_vae_decode_1024x1024",
            VAE_DEVICE_TEST_TOTAL_ITERATIONS,
            1,
            0.015,
            "",
        ),
        (
            "vae_decode_512x512",
            "pytest models/experimental/stable_diffusion_xl_base/vae/tests/pcc/test_module_tt_autoencoder_kl.py::test_vae -k 'test_512x512_decode'",
            "sdxl_vae",
            "sdxl_vae_decode_512x512",
            VAE_DEVICE_TEST_TOTAL_ITERATIONS,
            1,
            0.015,
            "",
        ),
        (
            "vae_encode_1024x1024",
            "pytest models/experimental/stable_diffusion_xl_base/vae/tests/pcc/test_module_tt_autoencoder_kl.py::test_vae -k 'test_1024x1024_encode'",
            "sdxl_vae",
            "sdxl_vae_encode_1024x1024",
            VAE_DEVICE_TEST_TOTAL_ITERATIONS,
            1,
            0.015,
            "",
        ),
        (
            "vae_encode_512x512",
            "pytest models/experimental/stable_diffusion_xl_base/vae/tests/pcc/test_module_tt_autoencoder_kl.py::test_vae -k 'test_512x512_encode'",
            "sdxl_vae",
            "sdxl_vae_encode_512x512",
            VAE_DEVICE_TEST_TOTAL_ITERATIONS,
            1,
            0.015,
            "",
        ),
        (
            "clip_encoder_1",
            "pytest models/experimental/stable_diffusion_xl_base/tests/pcc/test_sdxl_clip_encoders.py::test_clip_encoder -k 'encoder_1'",
            "sdxl_clip_encoder_1",
            "sdxl_clip_encoder_1",
            CLIP_ENCODER_DEVICE_TEST_TOTAL_ITERATIONS,
            1,
            0.015,
            "",
        ),
        (
            "clip_encoder_2",
            "pytest models/experimental/stable_diffusion_xl_base/tests/pcc/test_sdxl_clip_encoders.py::test_clip_encoder -k 'encoder_2'",
            "sdxl_clip_encoder_2",
            "sdxl_clip_encoder_2",
            CLIP_ENCODER_DEVICE_TEST_TOTAL_ITERATIONS,
            1,
            0.020,
            "",
        ),
    ],
    ids=[
        "test_sdxl_unet_1024x1024",
        "test_sdxl_unet_512x512",
        "test_sdxl_refiner_unet_1024x1024",
        "test_sdxl_refiner_unet_512x512",
        "test_sdxl_vae_decode_1024x1024",
        "test_sdxl_vae_decode_512x512",
        "test_sdxl_vae_encode_1024x1024",
        "test_sdxl_vae_encode_512x512",
        "test_sdxl_clip_encoder_1",
        "test_sdxl_clip_encoder_2",
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_sdxl_perf_device(test_id, command, subdir, model_name, num_iterations, batch_size, margin, comments):
    expected_perf = get_device_perf(test_id)
    if expected_perf is None:
        pytest.skip(f"Test {test_id} not configured for current device")

    if is_wormhole_b0():
        os.environ["TT_MM_THROTTLE_PERF"] = "0" if "clip_encoder" in command else "5"

    run_model_device_perf_test(
        command=command,
        expected_device_perf_ns_per_iteration=expected_perf,
        subdir=subdir,
        model_name=model_name,
        num_iterations=num_iterations,
        batch_size=batch_size,
        margin=margin,
        comments=comments,
    )
