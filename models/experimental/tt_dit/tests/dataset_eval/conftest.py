# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from ...pipelines.flux1.pipeline_flux1 import Flux1Pipeline
from ...pipelines.motif.pipeline_motif import MotifPipeline
from ...pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import StableDiffusion3Pipeline
from ...pipelines.mochi.pipeline_mochi import MochiPipeline
from ...pipelines.wan.pipeline_wan import WanPipeline

# with open(os.path.join(os.path.dirname(__file__), "eval_targets.json"), "r") as f:
#    targets_setup = json.load(f)
targets_setup = {
    "stable-diffusion-3.5-large": {
        "hf_id": "stabilityai/stable-diffusion-3.5-large",
        "num_inference_steps": 28,
        "pipeline": StableDiffusion3Pipeline,
        "device_params": {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "l1_small_size": 32768,
            "trace_region_size": 25000000,
        },
        "accuracy": {
            "5000": {"fid_valid_range": [22.63592, 24.0361780112], "clip_valid_range": [30.38525, 32.264881565]},
            "500": {"fid_valid_range": [79.734, 84.66634524], "clip_valid_range": [30.423468, 32.30546373048]},
            "100": {
                "fid_valid_range": [175.35951, 186.2072492886],
                "clip_valid_range": [30.162635, 32.028495601100005],
            },
        },
    },
    "flux.1-dev": {
        "hf_id": "black-forest-labs/FLUX.1-dev",
        "num_inference_steps": 28,
        "pipeline": Flux1Pipeline,
        "device_params": {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "l1_small_size": 32768,
            "trace_region_size": 50000000,
        },
        "accuracy": {
            "5000": {"fid_valid_range": [28.14067, 29.8814518462], "clip_valid_range": [29.7014, 31.538728604]},
            "500": {"fid_valid_range": [82.37143, 87.46692665980001], "clip_valid_range": [29.79258, 31.6355489988]},
            "100": {"fid_valid_range": [174.97054, 185.79421760440002], "clip_valid_range": [29.86339, 31.7107393054]},
        },
    },
    "flux.1-schnell": {
        "hf_id": "black-forest-labs/FLUX.1-schnell",
        "num_inference_steps": 4,
        "pipeline": Flux1Pipeline,
        "device_params": {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "l1_small_size": 32768,
            "trace_region_size": 50000000,
        },
        "accuracy": {
            "5000": {
                "fid_valid_range": [22.195637, 23.568659104820004],
                "clip_valid_range": [30.010539, 31.866990942540003],
            },
            "500": {"fid_valid_range": [77.21976, 81.9965743536], "clip_valid_range": [30.19901, 32.067120758600005]},
            "100": {"fid_valid_range": [171.065611, 181.64772969646], "clip_valid_range": [30.41629, 32.2978416994]},
        },
    },
    "motif-image-6b-preview": {
        "hf_id": "Motif-Technologies/Motif-Image-6B-Preview",
        "num_inference_steps": 28,
        "pipeline": MotifPipeline,
        "device_params": {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "l1_small_size": 32768,
            "trace_region_size": 31000000,
        },
        "accuracy": {
            "5000": {"fid_valid_range": [33.85203, 35.9461165758], "clip_valid_range": [29.6529, 31.487228394]},
            "500": {"fid_valid_range": [86.02833, 91.3500424938], "clip_valid_range": [29.65581, 31.4903184066]},
            "100": {"fid_valid_range": [180.97872, 192.1740636192], "clip_valid_range": [29.65581, 31.4903184066]},
        },
    },
    "mochi-1-preview": {
        "hf_id": "genmo/mochi-1-preview",
        "num_inference_steps": 50,
        "pipeline": MochiPipeline,
        "device_params": {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        },
        "accuracy": {"16": {"clip_valid_range": [26.20977, 27.83099]}},
        "is_video": True,
        "extra_args": {"num_frames": 168},
    },
    "wan2.2": {
        "hf_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "num_inference_steps": 40,
        "pipeline": WanPipeline,
        "device_params": {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        },
        "accuracy": {"16": {"clip_valid_range": [28.25276, 30.00036]}},
        "is_video": True,
        "extra_args": {"output_type": "pil"},
    },
}


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

    model_id_choices = targets_setup.keys()
    parser.addoption(
        "--model-id",
        action="store",
        required=True,
        choices=model_id_choices,
        help=f"Model ID to use for evaluation. Options:{','.join(model_id_choices)}",
    )

    parser.addoption(
        "--num-inference-steps",
        action="store",
        default=None,
        type=int,
        help="Number of inference steps (default: 28)",
    )

    parser.addoption(
        "--24hrs-test",
        action="store_true",
        default=False,
        help="Whether to run a 24-hour test",
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
        num_prompts = 5
    return start_from, num_prompts


@pytest.fixture
def model_id(request):
    return request.config.getoption("--model-id")


@pytest.fixture
def is_24hrs_test(request):
    return request.config.getoption("--24hrs-test")


@pytest.fixture
def num_inference_steps(request):
    return (
        request.config.getoption("--num-inference-steps")
        or targets_setup[request.config.getoption("--model-id")]["num_inference_steps"]
    )


@pytest.fixture
def dit_pipeline(model_id):
    return targets_setup[model_id]["pipeline"]


@pytest.fixture
def model_metadata(model_id):
    return targets_setup[model_id]


@pytest.fixture
def device_params(request, model_id):
    """Return device_params based on model_id, with optional parametrization override."""

    # If parametrized (indirect=True), use that value as base
    params = getattr(request, "param", {}).copy()

    # If no parametrization, use model-specific params
    if not params:
        params = targets_setup[model_id]["device_params"]
    # If parametrized, merge with model-specific (parametrized takes precedence)
    else:
        model_defaults = targets_setup[model_id]["device_params"]
        params = {**model_defaults, **params}  # Parametrized overrides model defaults

    return params


@pytest.fixture
def is_video(model_id):
    return targets_setup[model_id].get("is_video", False)
