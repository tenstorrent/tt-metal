# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os

os.environ["TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT"] = "15000"

import gc
import pytest
import torch
import ttnn

from diffusers import DiffusionPipeline

from models.experimental.stable_diffusion_xl_base.lora.tt_lora_weights_manager import TtLoRAWeightsManager
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations1024x1024
from models.experimental.stable_diffusion_xl_base.tt.tt_unet import TtUNet2DConditionModel
from models.perf.device_perf_utils import run_model_device_perf_test


def _get_diffusers_pipeline(model_location_generator, is_ci_env, is_ci_v2_env):
    model_location = model_location_generator(
        "stable-diffusion-xl-base-1.0", download_if_ci_v2=True, ci_v2_timeout_in_s=1800
    )
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0" if not is_ci_v2_env else model_location,
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_env or is_ci_v2_env,
    )

    return pipeline


@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
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
            "pytest models/experimental/stable_diffusion_xl_base/lora/tests/test_lora_perf.py::test_lora_fuse",
            102_565_757,
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
