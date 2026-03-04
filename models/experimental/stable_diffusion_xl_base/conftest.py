# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from pathlib import Path
from conftest import is_galaxy

from models.experimental.stable_diffusion_xl_base.lora.config import (
    LORA_FILENAME,
    LORA_REPO_ID,
)


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
        help="Path to LoRA weights: .safetensors file or directory containing the file. Overrides env and HF.",
    )
    parser.addoption(
        "--lora-hf-repo",
        action="store",
        default=None,
        help="Hugging Face repo id for LoRA (e.g. 'user/repo'). Used with --lora-hf-filename or when loading from HF.",
    )
    parser.addoption(
        "--lora-hf-filename",
        action="store",
        default=None,
        help="Filename in the HF repo or in --lora-weights directory (e.g. 'lora.safetensors').",
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


def _resolve_local_lora_path(path_str, filename=None):
    """
    Resolve a path (file or directory) to a single LoRA .safetensors file.
    Returns absolute path string or None if not found.
    """
    if not path_str or not path_str.strip():
        return None
    p = Path(path_str).expanduser().resolve()
    if not p.exists():
        return None
    if p.is_file():
        return str(p)
    name = filename or LORA_FILENAME
    candidate = p / name
    return str(candidate) if candidate.exists() else None


def _get_lora_filename_from_options(request):
    """Filename for LoRA: pytest option or default."""
    if request is not None:
        opt = request.config.getoption("--lora-hf-filename", default=None)
        if opt:
            return opt
    return LORA_FILENAME


@pytest.fixture(scope="function")
def lora_path(request, model_location_generator, is_ci_v2_env, is_ci_env):
    """
    Resolve LoRA weights path for tests. Priority:
    1. --lora-weights (pytest): file or directory
    2. CI: model_location_generator (Weka/MLPerf or CI v2)
    3. Hugging Face: --lora-hf-repo + --lora-hf-filename, or config defaults
    """
    filename = _get_lora_filename_from_options(request)

    # 1. Explicit path from CLI (--lora-weights)
    opt_path = request.config.getoption("--lora-weights", default=None)
    if opt_path is not None and str(opt_path).strip():
        resolved = _resolve_local_lora_path(opt_path, filename=filename)
        if resolved:
            return resolved
        pytest.skip(
            f"LoRA path not found: {opt_path!r}. "
            f"If it is a directory, ensure it contains a file named {filename!s}."
        )

    # # 2. CI
    # if is_ci_env:
    #     lora_dir = model_location_generator(
    #         LORA_CI_MODEL_VERSION,
    #         download_if_ci_v2=is_ci_v2_env,
    #     )
    #     lora_file = Path(lora_dir) / LORA_FILENAME
    #     if not lora_file.exists():
    #         pytest.skip(
    #             f"LoRA weights not found at {lora_file}. "
    #             f"Ensure {LORA_FILENAME} is under {LORA_CI_MODEL_VERSION} (Weka/MLPerf or CI v2 cache)."
    #         )
    #     return str(lora_file)

    # 3. Hugging Face: repo + filename from options or config defaults
    repo_id = request.config.getoption("--lora-hf-repo", default=None) or LORA_REPO_ID
    hf_filename = filename

    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=repo_id, filename=hf_filename)
    except Exception as e:
        pytest.skip(
            f"LoRA weights not available: {e}. "
            f"Use --lora-weights PATH for a local file/dir, or --lora-hf-repo/--lora-hf-filename for HF; or ensure network for default HF."
        )
