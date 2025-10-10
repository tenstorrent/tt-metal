# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.demo.demo import test_demo
from models.experimental.stable_diffusion_xl_base.utils.clip_encoder import CLIPEncoder
from loguru import logger
import statistics
from models.experimental.stable_diffusion_xl_base.utils.fid_score import calculate_fid_score
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    SDXL_L1_SMALL_SIZE,
    SDXL_TRACE_REGION_SIZE,
    SDXL_FABRIC_CONFIG,
)
from models.common.utility_functions import profiler
from models.experimental.stable_diffusion_xl_base.conftest import get_device_name
from models.experimental.stable_diffusion_xl_base.utils.clip_fid_ranges import (
    accuracy_check_clip,
    accuracy_check_fid,
    get_appr_delta_metric,
    targets,
)
from models.experimental.stable_diffusion_xl_base.utils.accuracy_helper import (
    sdxl_get_prompts,
    check_clip_scores,
    save_json_results,
)

test_demo.__test__ = False


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
):
    start_from, num_prompts = evaluation_range

    prompts = sdxl_get_prompts(
        captions_path,
        start_from,
        num_prompts,
    )

    logger.info(f"Start inference from prompt index: {start_from} to {start_from + num_prompts}")

    error_detected = False
    try:
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
            use_cfg_parallel=use_cfg_parallel,
            fixed_seed_for_batch=True,
        )
        ttnn.synchronize_device(mesh_device)
    except Exception as error_msg:
        error_msg = str(error_msg)
        error_detected = True

    data = {
        "model": "sdxl",
        "metadata": {
            "model_name": "sdxl",
            "device": get_device_name(),
            "device_vae": vae_on_device,
            "capture_trace": capture_trace,
            "encoders_on_device": encoders_on_device,
            "num_inference_steps": num_inference_steps,
            "start_from": start_from,
            "num_prompts": num_prompts,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
        },
        "benchmarks_summary": [
            {"model": "sdxl", "device": get_device_name(), "stability_check": 3 if error_detected else 2}
        ],
    }

    if error_detected:
        save_json_results(data, capture_trace, vae_on_device, encoders_on_device, use_cfg_parallel, num_inference_steps)
        logger.warning(f"Error detected during inference")
        raise RuntimeError(error_msg)

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

    if use_cfg_parallel:
        for key in ["functional", "complete", "target"]:
            targets["perf"][key] /= 2

    avg_gen_end_to_end = profiler.get("end_to_end_generation")
    data["benchmarks_summary"][0].update(
        {
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
    )
    data["evals"] = [
        {
            "model": "sdxl",
            "device": get_device_name(),
            "average_clip": average_clip_score,
            "deviation_clip": deviation_clip_score,
            "approx_clip_accuracy_check": accuracy_check_clip(average_clip_score, num_prompts, mode="approx"),
            "average_clip_accuracy_check": accuracy_check_clip(average_clip_score, num_prompts, mode="valid"),
            "delta_clip": get_appr_delta_metric(average_clip_score, num_prompts, score_type="clip"),
            "fid_score": fid_score,
            "approx_fid_accuracy_check": accuracy_check_fid(fid_score, num_prompts, mode="approx"),
            "fid_score_accuracy_check": accuracy_check_fid(fid_score, num_prompts, mode="valid"),
            "delta_fid": get_appr_delta_metric(fid_score, num_prompts, score_type="fid"),
            "accuracy_check_approx": min(
                accuracy_check_fid(fid_score, num_prompts, mode="approx"),
                accuracy_check_clip(average_clip_score, num_prompts, mode="approx"),
            ),
            "accuracy_check_delta": min(
                accuracy_check_fid(fid_score, num_prompts, mode="delta"),
                accuracy_check_clip(average_clip_score, num_prompts, mode="delta"),
            ),
            "accuracy_check_valid": min(
                accuracy_check_fid(fid_score, num_prompts, mode="valid"),
                accuracy_check_clip(average_clip_score, num_prompts, mode="valid"),
            ),
        }
    ]

    save_json_results(data, capture_trace, vae_on_device, encoders_on_device, use_cfg_parallel, num_inference_steps)

    check_clip_scores(start_from, num_prompts, prompts, clip_scores)
