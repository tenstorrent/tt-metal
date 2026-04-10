# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from loguru import logger

from models.common.utility_functions import is_wormhole_b0
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


# --- Pipeline Loading ---


@pytest.fixture(scope="function")
def load_sdxl_base_pipeline(sdxl_base_pipeline_location, is_ci_env, is_ci_v2_env):
    """
    Returns a callable that loads a DiffusionPipeline from the resolved SDXL base model location.
    Handles CIv2 (large file cache), CIv1 (MLPerf/HF cache), and local (HF download) transparently.

    Usage in tests::

        pipeline = load_sdxl_base_pipeline()
    """
    import torch
    from diffusers import DiffusionPipeline

    def _load():
        return DiffusionPipeline.from_pretrained(
            sdxl_base_pipeline_location,
            torch_dtype=torch.float32,
            use_safetensors=True,
            local_files_only=is_ci_v2_env or is_ci_env,
        )

    return _load


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


def is_galaxy():
    import ttnn

    return (
        ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.GALAXY
        or ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.BLACKHOLE_GALAXY
    )


def get_device_name():
    import ttnn

    cluster_type = ttnn.cluster.get_cluster_type()
    cluster_type_to_name = {
        ttnn.cluster.ClusterType.N150: "n150",
        ttnn.cluster.ClusterType.N300: "n300",
        ttnn.cluster.ClusterType.N300_2x2: "n300_2x2",
        ttnn.cluster.ClusterType.T3K: "t3k",
        ttnn.cluster.ClusterType.GALAXY: "galaxy",
        ttnn.cluster.ClusterType.BLACKHOLE_GALAXY: "bh_galaxy",
        ttnn.cluster.ClusterType.P100: "p100",
        ttnn.cluster.ClusterType.P150: "p150",
        ttnn.cluster.ClusterType.P150_X2: "p150x2",
        ttnn.cluster.ClusterType.P150_X4: "p150x4",
        ttnn.cluster.ClusterType.P150_X8: "p150x8",
        ttnn.cluster.ClusterType.P300: "p300",
        ttnn.cluster.ClusterType.P300_X2: "p300x2",
    }
    return cluster_type_to_name.get(cluster_type, "unknown")


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
def lora_path(request, model_location_generator, is_ci_env, is_ci_v2_env):
    """
    Resolve LoRA weights path.
    1) --lora-weights: full path to a local .safetensors file.
    2) CIv2: download from large file cache via model_location_generator.
    3) CIv1/local: download from Hugging Face via hf_hub_download.
    4) If nothing provided: use default weights.
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

    if is_ci_v2_env:
        # In CIv2, download from large file cache. LFC uses repo name without owner prefix.
        repo_name = hf_repo_id.split("/")[-1] if "/" in hf_repo_id else hf_repo_id
        lora_dir = model_location_generator(repo_name, download_if_ci_v2=True, ci_v2_timeout_in_s=300)
        lora_file = Path(lora_dir) / hf_filename
        if lora_file.exists():
            return str(lora_file)
        pytest.skip(
            f"LoRA weights not found in CIv2 large file cache at {lora_file}. "
            f"Upload them to LFC under '{repo_name}/{hf_filename}'."
        )
        return

    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=hf_repo_id, filename=hf_filename, local_files_only=is_ci_env)
    except Exception as _:
        pytest.skip(
            f"LoRA weights not available from HF ({hf_repo_id}, {hf_filename}). "
            f"Use --lora-weights for a local file path, or ensure network/cache for HF."
        )
        return
