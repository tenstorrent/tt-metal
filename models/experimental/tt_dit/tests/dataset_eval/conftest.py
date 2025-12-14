# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import json
from ...pipelines.flux1.pipeline_flux1 import Flux1Pipeline
from ...pipelines.motif.pipeline_motif import MotifPipeline
from ...pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import StableDiffusion3Pipeline
from ...pipelines.mochi.pipeline_mochi import MochiPipeline
from ...pipelines.wan.pipeline_wan import WanPipeline
import os

targets_setup = json.load(open(os.path.join(os.path.dirname(__file__), "eval_targets.json")))


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
        default=5,
        help="Number of prompts to process (default: 5)",
    )

    # model_id_choices = ["flux1.dev", "flux1.schnell", "sd35.large", "motif.image.6b.preview"]
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
def num_inference_steps(request):
    return (
        request.config.getoption("--num-inference-steps")
        or targets_setup[request.config.getoption("--model-id")]["num_inference_steps"]
    )


@pytest.fixture
def dit_pipeline(model_id):
    pipeline_map = {
        "flux.1-dev": Flux1Pipeline,
        "flux.1-schnell": Flux1Pipeline,
        "stable-diffusion-3.5-large": StableDiffusion3Pipeline,
        "motif-image-6b-preview": MotifPipeline,
        "mochi-1-preview": MochiPipeline,
        "wan2.2": WanPipeline,
    }
    return pipeline_map[model_id]


@pytest.fixture
def model_metadata(model_id):
    return targets_setup[model_id]
