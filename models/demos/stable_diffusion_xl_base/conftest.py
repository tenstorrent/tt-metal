# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from loguru import logger

from conftest import is_galaxy
from models.demos.stable_diffusion_xl_base.lora.config import TEST_LORA_FILENAME, TEST_LORA_REPO_ID


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
        return "galaxy"
    elif num_devices == 0:
        return "cpu"
    elif num_devices == 1:
        return "n150"
    elif num_devices == 2:
        return "n300"
    elif num_devices == 8:
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
