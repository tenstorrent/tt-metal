# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os

os.environ["TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT"] = "15000"

import gc
import json
from pathlib import Path

import pytest
import torch
from diffusers import DiffusionPipeline

import ttnn
from models.demos.stable_diffusion_xl_base.lora.tt_lora_weights_manager import TtLoRAWeightsManager
from models.demos.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations1024x1024
from models.demos.stable_diffusion_xl_base.tt.tt_unet import TtUNet2DConditionModel
from models.perf.device_perf_utils import run_model_device_perf_test

SDXL_MODEL_INDEX = {
    "_class_name": "StableDiffusionXLPipeline",
    "_diffusers_version": "0.18.2",
    "scheduler": ["diffusers", "EulerDiscreteScheduler"],
    "text_encoder": ["transformers", "CLIPTextModel"],
    "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
    "tokenizer": ["transformers", "CLIPTokenizer"],
    "tokenizer_2": ["transformers", "CLIPTokenizer"],
    "unet": ["diffusers", "UNet2DConditionModel"],
    "vae": ["diffusers", "AutoencoderKL"],
}

SDXL_COMPONENTS = ["unet", "vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "scheduler"]


def _get_diffusers_pipeline(model_location_generator, is_ci_env, is_ci_v2_env):
    if is_ci_v2_env:
        pipeline_dir = None
        for component in SDXL_COMPONENTS:
            loc = model_location_generator(
                f"stable-diffusion-xl-base-1.0/{component}",
                download_if_ci_v2=True,
                ci_v2_timeout_in_s=1800,
            )
            if pipeline_dir is None:
                pipeline_dir = Path(loc).parent

        model_index_path = pipeline_dir / "model_index.json"
        if not model_index_path.exists():
            model_index_path.write_text(json.dumps(SDXL_MODEL_INDEX, indent=2))

        model_location = pipeline_dir
    else:
        model_location = "stabilityai/stable-diffusion-xl-base-1.0"

    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0" if not is_ci_v2_env else model_location,
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_env or is_ci_v2_env,
    )

    return pipeline


def test_lora_fuse(
    device,
    model_location_generator,
    is_ci_env,
    is_ci_v2_env,
    lora_path,
):
    pipeline_for_tt = _get_diffusers_pipeline(model_location_generator, is_ci_env, is_ci_v2_env)
    pipeline_for_tt.unet.eval()
    state_dict = pipeline_for_tt.unet.state_dict()

    lora_mgr = TtLoRAWeightsManager(device, pipeline_for_tt)
    model_config = ModelOptimisations1024x1024()
    tt_unet = TtUNet2DConditionModel(
        device,
        state_dict,
        "unet",
        model_config=model_config,
        debug_mode=False,
        lora_weights_manager=lora_mgr,
    )

    lora_mgr.load_lora_weights(lora_path)
    ttnn.ReadDeviceProfiler(device)
    lora_mgr.fuse_lora(lora_scale=1.0)
    ttnn.ReadDeviceProfiler(device)

    del pipeline_for_tt, tt_unet, lora_mgr
    gc.collect()


@pytest.mark.parametrize(
    "command, expected_device_perf_ns_per_iteration, subdir, model_name, num_iterations, batch_size, margin, comments",
    [
        (
            "pytest models/demos/stable_diffusion_xl_base/lora/tests/test_lora_perf.py::test_lora_fuse",
            166_329_976,
            "sdxl_lora_fuse",
            "sdxl_lora_fuse",
            1,
            1,
            0.015,
            "",
        ),
    ],
    ids=["test_lora_fuse"],
)
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.timeout(7200)
def test_lora_perf_device(
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
