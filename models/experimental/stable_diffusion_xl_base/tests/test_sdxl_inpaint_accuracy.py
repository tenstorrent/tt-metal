# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.experimental.stable_diffusion_xl_base.demo.demo_inpainting import test_demo
from datasets import load_dataset
from loguru import logger
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    SDXL_L1_SMALL_SIZE,
    SDXL_TRACE_REGION_SIZE,
    SDXL_FABRIC_CONFIG,
)
import json
from models.experimental.stable_diffusion_xl_base.conftest import get_device_name
from models.experimental.stable_diffusion_xl_base.utils.accuracy_utils import (
    calculate_accuracy_metrics,
    create_report_json,
    save_report_json,
    check_clip_scores,
)

test_demo.__test__ = False
MAX_N_SAMPLES, MIN_N_SAMPLES = 1260, 2


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
    "num_inference_steps",
    ((20),),
)
@pytest.mark.parametrize(
    "guidance_scale",
    ((8.0),),
)
@pytest.mark.parametrize(
    "strength",
    ((0.99),),
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
@pytest.mark.parametrize("coco_statistics_path", ["models/experimental/stable_diffusion_xl_base/coco_data/val2014.npz"])
def test_accuracy_sdxl_inpaint(
    validate_fabric_compatibility,
    mesh_device,
    is_ci_env,
    num_inference_steps,
    vae_on_device,
    capture_trace,
    encoders_on_device,
    coco_statistics_path,
    evaluation_range,
    guidance_scale,
    negative_prompt,
    use_cfg_parallel,
    strength,
):
    start_from, num_prompts = evaluation_range
    input_images, input_masks, prompts = get_dataset_for_inpainting_accuracy(num_prompts)

    logger.info(f"Start inference from prompt index: {start_from} to {start_from + num_prompts}")

    images = test_demo(
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
        strength,
        use_cfg_parallel=use_cfg_parallel,
        fixed_seed_for_batch=True,
        prompt_2=None,
        negative_prompt_2=None,
        crop_coords_top_left=(0, 0),
        guidance_rescale=0.0,
        timesteps=None,
        sigmas=None,
        input_images=input_images,
        input_masks=input_masks,
    )

    accuracy_metrics = calculate_accuracy_metrics(images, prompts, coco_statistics_path)

    model_name = "sdxl-inpaint-tp" if use_cfg_parallel else "sdxl-inpaint"
    metadata = {
        "model_name": model_name,
        "device": get_device_name(),
        "device_vae": vae_on_device,
        "capture_trace": capture_trace,
        "encoders_on_device": encoders_on_device,
        "use_cfg_parallel": use_cfg_parallel,
        "negative_prompt": negative_prompt,
        "guidance_scale": guidance_scale,
        "strength": strength,
        "num_inference_steps": num_inference_steps,
        "start_from": start_from,
        "num_prompts": num_prompts,
    }

    report_json = create_report_json(metadata, accuracy_metrics)

    save_report_json(report_json, metadata)
    print(json.dumps(report_json, indent=4))

    check_clip_scores(model_name, evaluation_range, prompts, accuracy_metrics["clip_scores"])


def get_dataset_for_inpainting_accuracy(n_prompts: int):
    logger.info(f"Requested {n_prompts} prompts for inpainting accuracy evaluation...")
    if n_prompts > MAX_N_SAMPLES or n_prompts < MIN_N_SAMPLES:
        logger.warning(f"Requested number of prompts {n_prompts} is out of bounds [{MIN_N_SAMPLES}, {MAX_N_SAMPLES}]")
        if n_prompts > MAX_N_SAMPLES:
            n_prompts = MAX_N_SAMPLES
        else:
            n_prompts = MIN_N_SAMPLES
        logger.warning(f"Setting number of prompts to {n_prompts}")

    logger.info("Loading InpaintCOCO dataset for inpainting accuracy evaluation...")
    dataset = load_dataset("phiyodr/InpaintCOCO")

    input_images, input_masks, input_captions = [], [], []
    for index, item in enumerate(dataset["test"]):  # 'test' is only existing split in phiyodr/InpaintCOCO dataset
        if index >= n_prompts:
            break
        input_images.append(item["coco_image"])
        input_masks.append(item["mask"])
        input_captions.append(item["coco_caption"])
    return input_images, input_masks, input_captions
