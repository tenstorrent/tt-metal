# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.experimental.stable_diffusion_xl_base.demo.demo_base_and_refiner import test_demo_base_and_refiner
import os
from loguru import logger
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    SDXL_L1_SMALL_SIZE,
    SDXL_BASE_REFINER_TRACE_REGION_SIZE,
    SDXL_FABRIC_CONFIG,
)
import json
from models.experimental.stable_diffusion_xl_base.conftest import get_device_name
from models.experimental.stable_diffusion_xl_base.utils.accuracy_utils import (
    sdxl_get_prompts,
    calculate_accuracy_metrics,
    create_report_json,
    save_report_json,
    check_clip_scores,
)

test_demo_base_and_refiner.__test__ = False


@pytest.mark.parametrize(
    "device_params, use_cfg_parallel",
    [
        (
            {
                "l1_small_size": SDXL_L1_SMALL_SIZE,
                "trace_region_size": SDXL_BASE_REFINER_TRACE_REGION_SIZE,
                "fabric_config": SDXL_FABRIC_CONFIG,
            },
            True,
        ),
        (
            {
                "l1_small_size": SDXL_L1_SMALL_SIZE,
                "trace_region_size": SDXL_BASE_REFINER_TRACE_REGION_SIZE,
            },
            False,
        ),
    ],
    indirect=["device_params"],
    ids=["use_cfg_parallel", "no_cfg_parallel"],
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
    "negative_prompt",
    (("normal quality, low quality, worst quality, low res, blurry, nsfw, nude"),),
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
    "capture_trace",
    [
        (True),
        (False),
    ],
    ids=("with_trace", "no_trace"),
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
    "use_refiner",
    [
        (True),
        (False),
    ],
    ids=("with_refiner", "no_refiner"),
)
@pytest.mark.parametrize(
    "denoising_split",
    [
        (0.8),
    ],
)
@pytest.mark.parametrize(
    "refiner_strength, refiner_aesthetic_score, refiner_negative_aesthetic_score",
    [
        (0.3, 6.0, 2.5),
    ],
    ids=["default_refiner_params"],
)
@pytest.mark.parametrize("captions_path", ["models/experimental/stable_diffusion_xl_base/coco_data/captions.tsv"])
@pytest.mark.parametrize("coco_statistics_path", ["models/experimental/stable_diffusion_xl_base/coco_data/val2014.npz"])
def test_accuracy_sdxl(
    validate_fabric_compatibility,
    mesh_device,
    is_ci_env,
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
    use_refiner,
    denoising_split,
    refiner_strength,
    refiner_aesthetic_score,
    refiner_negative_aesthetic_score,
):
    start_from, num_prompts = evaluation_range

    prompts = sdxl_get_prompts(
        captions_path,
        start_from,
        num_prompts,
    )

    logger.info(f"Start inference from prompt index: {start_from} to {start_from + num_prompts}")

    images = test_demo_base_and_refiner(
        validate_fabric_compatibility,
        mesh_device,
        is_ci_env,
        prompts,
        negative_prompt,
        num_inference_steps,
        vae_on_device,
        encoders_on_device,
        capture_trace,
        evaluation_range,
        guidance_scale,
        use_cfg_parallel=use_cfg_parallel,
        fixed_seed_for_batch=True,
        prompt_2=None,
        negative_prompt_2=None,
        crop_coords_top_left=(0, 0),
        guidance_rescale=0.0,
        timesteps=None,
        sigmas=None,
        refiner_strength=refiner_strength,
        refiner_aesthetic_score=refiner_aesthetic_score,
        refiner_negative_aesthetic_score=refiner_negative_aesthetic_score,
        use_refiner=use_refiner,
        denoising_split=denoising_split,
    )

    skip_check_and_save = os.getenv("TT_SDXL_SKIP_CHECK_AND_SAVE", "0") == "1"
    if skip_check_and_save:
        logger.info("Skipping accuracy check and saving results as per environment variable.")
        return

    accuracy_metrics = calculate_accuracy_metrics(images, prompts, coco_statistics_path)

    model_name = ("sdxl-base-refiner" if use_refiner else "sdxl") + ("-tp" if use_cfg_parallel else "")
    metadata = {
        "model_name": model_name,
        "device": get_device_name(),
        "device_vae": vae_on_device,
        "capture_trace": capture_trace,
        "encoders_on_device": encoders_on_device,
        "use_cfg_parallel": use_cfg_parallel,
        "use_refiner": use_refiner,
        "negative_prompt": negative_prompt,
        "guidance_scale": guidance_scale,
        "denoising_split": denoising_split,
        "refiner_strength": refiner_strength,
        "refiner_aesthetic_score": refiner_aesthetic_score,
        "refiner_negative_aesthetic_score": refiner_negative_aesthetic_score,
        "num_inference_steps": num_inference_steps,
        "start_from": start_from,
        "num_prompts": num_prompts,
    }

    report_json = create_report_json(metadata, accuracy_metrics)

    save_report_json(report_json, metadata)
    print(json.dumps(report_json, indent=4))

    check_clip_scores(model_name, evaluation_range, prompts, accuracy_metrics["clip_scores"])
