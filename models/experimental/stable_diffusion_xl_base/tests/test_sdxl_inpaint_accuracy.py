# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.experimental.stable_diffusion_xl_base.demo.demo_inpainting import test_demo
from models.experimental.stable_diffusion_xl_base.utils.clip_encoder import CLIPEncoder
import os
from datasets import load_dataset
from loguru import logger
import statistics
from models.experimental.stable_diffusion_xl_base.utils.fid_score import calculate_fid_score
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    SDXL_L1_SMALL_SIZE,
    SDXL_TRACE_REGION_SIZE,
    SDXL_FABRIC_CONFIG,
)
import json
from models.common.utility_functions import profiler
from models.experimental.stable_diffusion_xl_base.conftest import get_device_name
from models.experimental.stable_diffusion_xl_base.utils.clip_fid_ranges import (
    accuracy_check_clip,
    accuracy_check_fid,
    get_appr_delta_metric,
    get_model_targets,
)

test_demo.__test__ = False
OUT_ROOT, RESULTS_FILE_NAME = "test_reports", "sdxl_test_results.json"
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

    clip = CLIPEncoder()

    clip_scores = []

    for idx, image in enumerate(images):
        clip_scores.append(100 * clip.get_clip_score(prompts[idx], image).item())

    average_clip_score = sum(clip_scores) / len(clip_scores)

    deviation_clip_score = "N/A"
    fid_score = "N/A"

    if num_prompts >= 2:
        deviation_clip_score = statistics.stdev(clip_scores)
        fid_score = calculate_fid_score(images, coco_statistics_path)
    else:
        logger.info("FID score is not calculated for less than 2 prompts.")

    print(f"FID score: {fid_score}")
    print(f"Average CLIP Score: {average_clip_score}")
    print(f"Standard Deviation of CLIP Scores: {deviation_clip_score}")

    avg_gen_end_to_end = profiler.get("end_to_end_generation")

    model_name = "sdxl-inpaint-tp" if use_cfg_parallel else "sdxl-inpaint"
    targets = get_model_targets(model_name)

    data = {
        "model": model_name,
        "metadata": {
            "model_name": model_name,
            "device": get_device_name(),
            "device_vae": vae_on_device,
            "capture_trace": capture_trace,
            "encoders_on_device": encoders_on_device,
            "num_inference_steps": num_inference_steps,
            "start_from": start_from,
            "num_prompts": num_prompts,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "use_cfg_parallel": use_cfg_parallel,
        },
        "benchmarks_summary": [
            {
                "model": model_name,
                "device": get_device_name(),
                "avg_gen_time": avg_gen_end_to_end,
                "target_checks": {
                    "functional": {
                        "avg_gen_time": targets["perf"]["functional"],
                        "avg_gen_time_check": 2 if targets["perf"]["functional"] >= avg_gen_end_to_end else 3,
                    },
                    "complete": {
                        "avg_gen_time": targets["perf"]["complete"],
                        "avg_gen_time_check": 2 if targets["perf"]["complete"] >= avg_gen_end_to_end else 3,
                    },
                    "target": {
                        "avg_gen_time": targets["perf"]["target"],
                        "avg_gen_time_check": 2 if targets["perf"]["target"] >= avg_gen_end_to_end else 3,
                    },
                },
                "average_denoising_time": profiler.get("denoising_loop"),
                "average_vae_time": profiler.get("vae_decode"),
                "min_gen_time": min(profiler.times["end_to_end_generation"]),
                "max_gen_time": max(profiler.times["end_to_end_generation"]),
                "average_encoding_time": profiler.get("encode_prompts"),
            }
        ],
        "evals": [
            {
                "model": model_name,
                "device": get_device_name(),
                "average_clip": average_clip_score,
                "deviation_clip": deviation_clip_score,
                "delta_clip": get_appr_delta_metric(model_name, average_clip_score, num_prompts, score_type="clip"),
                "fid_score": fid_score,
                "delta_fid": get_appr_delta_metric(model_name, fid_score, num_prompts, score_type="fid"),
                "accuracy_check": min(
                    accuracy_check_fid(model_name, fid_score, num_prompts, mode="approx"),
                    accuracy_check_clip(model_name, average_clip_score, num_prompts, mode="approx"),
                ),
            }
        ],
    }

    os.makedirs(OUT_ROOT, exist_ok=True)
    trace_flag = "with_trace" if capture_trace else "no_trace"
    vae_flag = "device_vae" if vae_on_device else "host_vae"
    encoders_flag = "device_encoders" if encoders_on_device else "host_encoders"
    use_cfg_parallel_flag = "cfg_parallel" if use_cfg_parallel else "no_cfg_parallel"
    new_file_name = (
        f"sdxl_test_results_{trace_flag}_{vae_flag}_{encoders_flag}_{use_cfg_parallel_flag}_{num_prompts}.json"
    )
    with open(f"{OUT_ROOT}/{new_file_name}", "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"Test results saved to {OUT_ROOT}/{new_file_name}")

    with open(
        f"{OUT_ROOT}/{RESULTS_FILE_NAME}", "w"
    ) as f:  # this is for CI and test_sdxl_accuracy_with_reset.py compatibility
        json.dump(data, f, indent=4)

    logger.info(f"Test results saved to {OUT_ROOT}/{RESULTS_FILE_NAME}")
    print(json.dumps(data, indent=4))


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
