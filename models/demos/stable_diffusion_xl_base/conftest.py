# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from loguru import logger

from conftest import is_galaxy
from models.common.utility_functions import is_blackhole, is_wormhole_b0
from models.demos.stable_diffusion_xl_base.lora.config import TEST_LORA_FILENAME, TEST_LORA_REPO_ID
from models.demos.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE, SDXL_L1_SMALL_SIZE_BH

# =============================================================================
# SDXL Model Location Fixtures (Session-scoped for CIv2 download efficiency)
# =============================================================================
# These fixtures download models once per pytest session and cache the location.
# This prevents redundant downloads when running multiple SDXL tests.
# =============================================================================

# --- Base Model Locations ---


@pytest.fixture(scope="session")
def sdxl_base_unet_location(model_location_generator, is_ci_v2_env):
    """
    Returns the location for SDXL base UNet model weights.
    In CIv2: Downloads once per session from large file cache.
    In CIv1/local: Returns HF repo ID (resolved via HF_HUB_CACHE to local cache or MLPerf mount).
    """
    if is_ci_v2_env:
        return model_location_generator(
            "stable-diffusion-xl-base-1.0/unet",
            download_if_ci_v2=True,
            ci_v2_timeout_in_s=1800,
        )
    else:
        return "stabilityai/stable-diffusion-xl-base-1.0"


@pytest.fixture(scope="session")
def sdxl_base_pipeline_location(model_location_generator, is_ci_v2_env):
    """
    Returns the location for full SDXL base pipeline.
    In CIv2: Downloads once per session from large file cache.
    In CIv1/local: Returns HF repo ID (resolved via HF_HUB_CACHE to local cache or MLPerf mount).
    """
    if is_ci_v2_env:
        return model_location_generator(
            "stable-diffusion-xl-base-1.0",
            download_if_ci_v2=True,
            ci_v2_timeout_in_s=1800,
        )
    else:
        return "stabilityai/stable-diffusion-xl-base-1.0"


@pytest.fixture(scope="session")
def sdxl_base_vae_location(model_location_generator, is_ci_v2_env):
    """
    Returns the location for SDXL base VAE model weights.
    In CIv2: Downloads once per session from large file cache.
    In CIv1/local: Returns HF repo ID (resolved via HF_HUB_CACHE to local cache or MLPerf mount).
    """
    if is_ci_v2_env:
        return model_location_generator(
            "stable-diffusion-xl-base-1.0/vae",
            download_if_ci_v2=True,
            ci_v2_timeout_in_s=1800,
        )
    else:
        return "stabilityai/stable-diffusion-xl-base-1.0"


@pytest.fixture(scope="session")
def sdxl_base_text_encoder_location(model_location_generator, is_ci_v2_env):
    """
    Returns the location for SDXL base text_encoder (CLIP) model weights.
    In CIv2: Downloads once per session from large file cache.
    In CIv1/local: Returns HF repo ID (resolved via HF_HUB_CACHE to local cache or MLPerf mount).
    """
    if is_ci_v2_env:
        return model_location_generator(
            "stable-diffusion-xl-base-1.0/text_encoder",
            download_if_ci_v2=True,
            ci_v2_timeout_in_s=1800,
        )
    else:
        return "stabilityai/stable-diffusion-xl-base-1.0"


@pytest.fixture(scope="session")
def sdxl_base_text_encoder_2_location(model_location_generator, is_ci_v2_env):
    """
    Returns the location for SDXL base text_encoder_2 (CLIP with projection) model weights.
    In CIv2: Downloads once per session from large file cache.
    In CIv1/local: Returns HF repo ID (resolved via HF_HUB_CACHE to local cache or MLPerf mount).
    """
    if is_ci_v2_env:
        return model_location_generator(
            "stable-diffusion-xl-base-1.0/text_encoder_2",
            download_if_ci_v2=True,
            ci_v2_timeout_in_s=1800,
        )
    else:
        return "stabilityai/stable-diffusion-xl-base-1.0"


@pytest.fixture(scope="session")
def sdxl_base_tokenizer_location(model_location_generator, is_ci_v2_env):
    """
    Returns the location for SDXL base tokenizer.
    In CIv2: Downloads once per session from large file cache.
    In CIv1/local: Returns HF repo ID (resolved via HF_HUB_CACHE to local cache or MLPerf mount).
    """
    if is_ci_v2_env:
        return model_location_generator(
            "stable-diffusion-xl-base-1.0/tokenizer",
            download_if_ci_v2=True,
            ci_v2_timeout_in_s=1800,
        )
    else:
        return "stabilityai/stable-diffusion-xl-base-1.0"


@pytest.fixture(scope="session")
def sdxl_base_tokenizer_2_location(model_location_generator, is_ci_v2_env):
    """
    Returns the location for SDXL base tokenizer_2.
    In CIv2: Downloads once per session from large file cache.
    In CIv1/local: Returns HF repo ID (resolved via HF_HUB_CACHE to local cache or MLPerf mount).
    """
    if is_ci_v2_env:
        return model_location_generator(
            "stable-diffusion-xl-base-1.0/tokenizer_2",
            download_if_ci_v2=True,
            ci_v2_timeout_in_s=1800,
        )
    else:
        return "stabilityai/stable-diffusion-xl-base-1.0"


# --- Inpainting Model Locations ---


@pytest.fixture(scope="session")
def sdxl_inpainting_unet_location(model_location_generator, is_ci_v2_env):
    """
    Returns the location for SDXL inpainting UNet model weights.
    In CIv2: Downloads once per session from large file cache.
    In CIv1/local: Returns HF repo ID.
    """
    if is_ci_v2_env:
        return model_location_generator(
            "stable-diffusion-xl-1.0-inpainting-0.1/unet",
            download_if_ci_v2=True,
            ci_v2_timeout_in_s=1800,
        )
    else:
        return "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"


@pytest.fixture(scope="session")
def sdxl_inpainting_pipeline_location(model_location_generator, is_ci_v2_env):
    """
    Returns the location for full SDXL inpainting pipeline.
    In CIv2: Downloads once per session from large file cache.
    In CIv1/local: Returns HF repo ID.
    """
    if is_ci_v2_env:
        return model_location_generator(
            "stable-diffusion-xl-1.0-inpainting-0.1",
            download_if_ci_v2=True,
            ci_v2_timeout_in_s=1800,
        )
    else:
        return "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"


# --- Refiner Model Locations ---


@pytest.fixture(scope="session")
def sdxl_refiner_unet_location(model_location_generator, is_ci_v2_env):
    """
    Returns the location for SDXL refiner UNet model weights.
    In CIv2: Downloads once per session from large file cache.
    In CIv1/local: Returns HF repo ID (resolved via HF_HUB_CACHE to local cache or MLPerf mount).
    """
    if is_ci_v2_env:
        return model_location_generator(
            "stable-diffusion-xl-refiner-1.0/unet",
            download_if_ci_v2=True,
            ci_v2_timeout_in_s=1800,
        )
    else:
        return "stabilityai/stable-diffusion-xl-refiner-1.0"


@pytest.fixture(scope="session")
def sdxl_refiner_pipeline_location(model_location_generator, is_ci_v2_env):
    """
    Returns the location for full SDXL refiner pipeline.
    In CIv2: Downloads once per session from large file cache.
    In CIv1/local: Returns HF repo ID (resolved via HF_HUB_CACHE to local cache or MLPerf mount).
    """
    if is_ci_v2_env:
        return model_location_generator(
            "stable-diffusion-xl-refiner-1.0",
            download_if_ci_v2=True,
            ci_v2_timeout_in_s=1800,
        )
    else:
        return "stabilityai/stable-diffusion-xl-refiner-1.0"


def pytest_configure(config):
    """Override global timeout setting for SDXL tests"""
    config.option.timeout = 0


def pytest_addoption(parser):
    parser.addoption(
        "--start-from",
        action="store",
        default=0,
        help="Start from prompt number (0-4999)",
    )
    parser.addoption(
        "--num-prompts",
        action="store",
        default=5000,
        help="Number of prompts to process (default: 5000)",
    )
    parser.addoption(
        "--reset-bool",
        action="store",
        type=int,
        default=1,
        help="Whether to reset periodically (1 or 0), default: 1",
    )
    parser.addoption(
        "--reset-period",
        action="store",
        default=200,
        type=int,
        help="How often to reset (default: 200 (images))",
    )
    parser.addoption(
        "--loop-iter-num",
        action="store",
        default=10,
        help="Number of iterations of denoising loop (default: 10)",
    )
    parser.addoption(
        "--debug-mode",
        action="store_true",
        default=False,
        help="Run SDXL in debug mode (default: False)",
    )
    parser.addoption(
        "--lora-weights",
        action="store",
        default=None,
        help="Full path to a local .safetensors file with LoRA weights. Overrides --lora-hf-repo and --lora-hf-filename",
    )
    parser.addoption(
        "--lora-hf-repo",
        action="store",
        default=None,
        help="Hugging Face repo id for LoRA (e.g. 'user/repo'). Required together with --lora-hf-filename",
    )
    parser.addoption(
        "--lora-hf-filename",
        action="store",
        default=None,
        help="Filename in the Hugging Face repo (e.g. 'lora.safetensors'). Required together with --lora-hf-repo",
    )


@pytest.fixture
def evaluation_range(request):
    start_from = request.config.getoption("--start-from")
    num_prompts = request.config.getoption("--num-prompts")
    if start_from is not None:
        start_from = int(start_from)
    else:
        start_from = 0
    if num_prompts is not None:
        num_prompts = int(num_prompts)
    else:
        num_prompts = 5000
    return start_from, num_prompts


@pytest.fixture
def reset_config(request):
    reset_bool_val = request.config.getoption("--reset-bool")
    reset_period = request.config.getoption("--reset-period")
    if reset_bool_val is not None:
        reset_bool = bool(reset_bool_val)
    else:
        reset_bool = True
    if reset_period is not None:
        reset_period = int(reset_period)
    else:
        reset_period = 200
    return reset_bool, reset_period


def get_device_name():
    import ttnn

    num_devices = ttnn.GetNumAvailableDevices()
    if is_galaxy():
        if is_blackhole():
            return "bh galaxy"
        return "galaxy"
    elif num_devices == 0:
        return "cpu"
    elif num_devices == 1:
        if is_blackhole():
            return "p150"
        return "n150"
    elif num_devices == 2:
        if is_blackhole():
            return "p300"
        return "n300"
    elif num_devices == 8:
        if is_blackhole():
            return "bh t3k"
        return "t3k"


@pytest.fixture
def loop_iter_num(request):
    return int(request.config.getoption("--loop-iter-num"))


@pytest.fixture
def debug_mode(request):
    return request.config.getoption("--debug-mode")


@pytest.fixture(scope="function")
def validate_fabric_compatibility(request):
    """
    Validate that fabric configuration is compatible with the requested mesh device configuration.
    This fixture runs before mesh_device creation to catch incompatibilities early.
    It is needed to be able to gracefully fail if the configuration is not possible.
    """
    import ttnn

    params = getattr(request.node, "callspec", {}).params
    use_cfg_parallel = params.get("use_cfg_parallel", None)
    mesh_device_param = params.get("mesh_device", None)

    if not use_cfg_parallel:
        return

    if mesh_device_param is not None:
        total_devices = ttnn.GetNumAvailableDevices()

        if isinstance(mesh_device_param, int):
            requested_devices = mesh_device_param
        elif isinstance(mesh_device_param, tuple):
            requested_devices = mesh_device_param[0] * mesh_device_param[1]
        else:
            requested_devices = total_devices

        assert requested_devices == total_devices, "Requested devices must be equal to total devices"


@pytest.fixture
def sdxl_l1_small_size():
    """
    Returns the appropriate L1_SMALL_SIZE value based on device architecture.
    """
    return SDXL_L1_SMALL_SIZE if is_wormhole_b0() else SDXL_L1_SMALL_SIZE_BH


@pytest.fixture(scope="function")
def device_params(request, sdxl_l1_small_size):
    """
    Override the global device_params fixture to automatically inject SDXL L1_SMALL_SIZE.

    If the parametrized device_params dict doesn't contain 'l1_small_size',
    it will be automatically added based on the device architecture (Wormhole vs Blackhole).

    NOTE: `device_params` fixture exists in conftest.py in root but this one will take precedence.
    Fixture in conftest.py closest to the test in hierarchy will take precedence.
    We can still set L1_SMALL_SIZE inside tests for some specific cases if needed.
    Otherwise, when we do not specify it inside parentheses in test params, it will get set
    inside this device_params fixture.
    """
    params = getattr(request, "param", {})

    # Auto-inject l1_small_size if not already specified
    if "l1_small_size" not in params:
        params = {**params, "l1_small_size": sdxl_l1_small_size}

    return params


def _resolve_local_lora_file_path(path_input):
    if not path_input or not path_input.strip():
        return None
    resolved_path = Path(path_input).expanduser().resolve()
    if not resolved_path.exists() or not resolved_path.is_file():
        return None
    return str(resolved_path)


@pytest.fixture(scope="function")
def lora_path(request, is_ci_env, is_ci_v2_env):
    """
    Resolve LoRA weights path.
    1) --lora-weights: full path to a local .safetensors file.
    2) --lora-hf-repo and --lora-hf-filename: download from Hugging Face.
    3) If nothing provided: use default weights (HF download).
    """
    lora_weights_cli_path = request.config.getoption("--lora-weights", default=None)
    hf_repo_id = request.config.getoption("--lora-hf-repo", default=None)
    hf_filename = request.config.getoption("--lora-hf-filename", default=None)

    # Local file path via --lora-weights
    if lora_weights_cli_path is not None and str(lora_weights_cli_path).strip():
        resolved_lora_path = _resolve_local_lora_file_path(lora_weights_cli_path)
        if resolved_lora_path:
            return resolved_lora_path
        pytest.skip(
            f"LoRA path must be an existing .safetensors file: {lora_weights_cli_path}. "
            f"Provide a full path to the file (not a directory)."
        )
        return

    if not (hf_repo_id and hf_filename):
        logger.warning(
            f"No LoRA weights provided. Using default weights. Repo: {TEST_LORA_REPO_ID}, File: {TEST_LORA_FILENAME}"
        )
        hf_repo_id = TEST_LORA_REPO_ID
        hf_filename = TEST_LORA_FILENAME

    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=hf_repo_id, filename=hf_filename, local_files_only=is_ci_env and not is_ci_v2_env
        )
    except Exception as _:
        pytest.skip(
            f"LoRA weights not available from HF ({hf_repo_id}, {hf_filename}). "
            f"Use --lora-weights for a local file path, or ensure network/cache for HF."
        )
        return
