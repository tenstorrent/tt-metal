# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
from loguru import logger

from models.demos.stable_diffusion_xl_base.conftest import get_device_name
from models.demos.stable_diffusion_xl_base.demo.demo_lora import run_demo_inference
from models.demos.stable_diffusion_xl_base.tests.test_common import (
    SDXL_FABRIC_CONFIG,
    SDXL_L1_SMALL_SIZE,
    SDXL_TRACE_REGION_SIZE,
    prepare_device,
)
from models.demos.stable_diffusion_xl_base.utils.accuracy_utils import (
    accuracy_assert,
    calculate_accuracy_metrics,
    check_clip_scores,
    create_report_json,
    save_report_json,
    sdxl_get_prompts,
)


@pytest.mark.parametrize(
    "image_resolution",
    [
        (1024, 1024),
        (512, 512),
    ],
    ids=["1024x1024", "512x512"],
)
# Note: The 'fabric_config' parameter is only required when running with cfg_parallel enabled,
# as the all_gather_async operation used in this mode depends on fabric being set.
@pytest.mark.parametrize(
    "device_params, use_cfg_parallel",
    [
        (
            {
                "l1_small_size": SDXL_L1_SMALL_SIZE,
                "trace_region_size": SDXL_TRACE_REGION_SIZE,
                "fabric_config": SDXL_FABRIC_CONFIG,
            },
            True,
        ),
        (
            {
                "l1_small_size": SDXL_L1_SMALL_SIZE,
                "trace_region_size": SDXL_TRACE_REGION_SIZE,
            },
            False,
        ),
    ],
    indirect=["device_params"],
    ids=["use_cfg_parallel", "no_cfg_parallel"],
)
@pytest.mark.parametrize(
    "negative_prompt",
    (("normal quality, low quality, worst quality, low res, blurry, nsfw, nude"),),
)
@pytest.mark.parametrize(
    "num_inference_steps",
    ((20),),
)
@pytest.mark.parametrize(
    "guidance_scale",
    ((8.0),),
)
@pytest.mark.parametrize(
    "vae_on_device",
    [
        (True),
        (False),
    ],
    ids=("device_vae", "host_vae"),
)
@pytest.mark.parametrize(
    "encoders_on_device",
    [
        (True),
        (False),
    ],
    ids=("device_encoders", "host_encoders"),
)
@pytest.mark.parametrize(
    "capture_trace",
    [
        (True),
        (False),
    ],
    ids=("with_trace", "no_trace"),
)
@pytest.mark.parametrize(
    "prompt_2, negative_prompt_2, crop_coords_top_left, guidance_rescale, timesteps, sigmas",
    [
        (None, None, (0, 0), 0.0, None, None),
    ],
    ids=["default_additional_parameters"],
)
@pytest.mark.parametrize("captions_path", ["models/demos/stable_diffusion_xl_base/coco_data/captions.tsv"])
@pytest.mark.parametrize("coco_statistics_path", ["models/demos/stable_diffusion_xl_base/coco_data/val2014.npz"])
def test_accuracy_sdxl_lora(
    validate_fabric_compatibility,
    mesh_device,
    is_ci_env,
    image_resolution,
    num_inference_steps,
    vae_on_device,
    capture_trace,
    encoders_on_device,
    captions_path,
    coco_statistics_path,
    evaluation_range,
    guidance_scale,
    negative_prompt,
    use_cfg_parallel,
    prompt_2,
    negative_prompt_2,
    crop_coords_top_left,
    guidance_rescale,
    timesteps,
    sigmas,
    lora_path,
):
    if image_resolution == (512, 512):
        pytest.skip("Accuracy target not available for 512x512 image resolution")

    prepare_device(mesh_device, use_cfg_parallel)

    start_from, num_prompts = evaluation_range

    prompts = sdxl_get_prompts(
        captions_path,
        start_from,
        num_prompts,
    )

    prompts_suffix = "ColoringBook: "  # Default Lora (ColoringBookRedmond-V2) works best with this suffix
    prompts = [prompts_suffix + prompt for prompt in prompts]

    logger.info(f"Start inference from prompt index: {start_from} to {start_from + num_prompts}")

    images = run_demo_inference(
        mesh_device,
        is_ci_env,
        image_resolution,
        prompts,
        negative_prompt,
        num_inference_steps,
        vae_on_device,
        encoders_on_device,
        evaluation_range,
        capture_trace,
        guidance_scale,
        use_cfg_parallel,
        fixed_seed_for_batch=True,
        prompt_2=prompt_2,
        negative_prompt_2=negative_prompt_2,
        crop_coords_top_left=crop_coords_top_left,
        guidance_rescale=guidance_rescale,
        timesteps=timesteps,
        sigmas=sigmas,
        lora_path=lora_path,
    )

    skip_check_and_save = os.getenv("TT_SDXL_SKIP_CHECK_AND_SAVE", "0") == "1"
    if skip_check_and_save:
        logger.info("Skipping accuracy check and saving results as per environment variable.")
        return

    accuracy_metrics = calculate_accuracy_metrics(images, prompts, coco_statistics_path)

    model_name = "sdxl-lora" + ("-tp" if use_cfg_parallel else "")
    metadata = {
        "model_name": model_name,
        "device": get_device_name(),
        "device_vae": vae_on_device,
        "capture_trace": capture_trace,
        "encoders_on_device": encoders_on_device,
        "use_cfg_parallel": use_cfg_parallel,
        "negative_prompt": negative_prompt,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "start_from": start_from,
        "num_prompts": num_prompts,
    }

    report_json = create_report_json(metadata, accuracy_metrics)

    save_report_json(report_json, metadata)
    print(json.dumps(report_json, indent=4))

    check_clip_scores(model_name, evaluation_range, prompts, accuracy_metrics["clip_scores"])
    accuracy_assert(metadata, accuracy_metrics)
