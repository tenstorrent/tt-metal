# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.experimental.stable_diffusion_xl_base.demo.demo_img2img import test_demo
from datasets import load_dataset
from loguru import logger
import statistics
import json
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    SDXL_L1_SMALL_SIZE,
    SDXL_TRACE_REGION_SIZE,
    SDXL_FABRIC_CONFIG,
)
import torch.nn.functional as F
from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)
from models.experimental.stable_diffusion_xl_base.conftest import get_device_name
from models.experimental.stable_diffusion_xl_base.utils.accuracy_utils import get_benchmark_summary, save_report_json
from models.experimental.stable_diffusion_xl_base.utils.clip_fid_ranges import accuracy_check_clip

test_demo.__test__ = False
MAX_N_SAMPLES, MIN_N_SAMPLES = 10000, 2


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
    "fixed_seed_for_batch",
    (True,),
)
@pytest.mark.parametrize(
    "negative_prompt",
    ((None),),
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
    "strength",
    ((0.7),),
)
@pytest.mark.parametrize(
    "prompt_2, negative_prompt_2, crop_coords_top_left, guidance_rescale, timesteps, sigmas",
    [
        (None, None, (0, 0), 0.0, None, None),
    ],
    ids=["default_additional_parameters"],
)
def test_accuracy_sdxl_img2img(
    validate_fabric_compatibility,
    mesh_device,
    is_ci_env,
    negative_prompt,
    num_inference_steps,
    vae_on_device,
    encoders_on_device,
    capture_trace,
    evaluation_range,
    guidance_scale,
    use_cfg_parallel,
    fixed_seed_for_batch,
    strength,
    prompt_2,
    negative_prompt_2,
    crop_coords_top_left,
    guidance_rescale,
    timesteps,
    sigmas,
):
    start_from, num_prompts = evaluation_range

    assert (
        num_prompts >= MIN_N_SAMPLES and num_prompts <= MAX_N_SAMPLES
    ), f"Number of prompts must be between {MIN_N_SAMPLES} and {MAX_N_SAMPLES}"
    dataset = load_dataset(
        "imthanhlv/instructpix2pix-clip-filtered-10k",
        split=f"train[:{num_prompts}]",
    )

    images = test_demo(
        validate_fabric_compatibility,
        mesh_device,
        is_ci_env,
        dataset["edit_prompt"],
        dataset["original_image"],
        negative_prompt,
        num_inference_steps,
        vae_on_device,
        encoders_on_device,
        capture_trace,
        evaluation_range,
        guidance_scale,
        use_cfg_parallel,
        fixed_seed_for_batch,
        strength,
        prompt_2,
        negative_prompt_2,
        crop_coords_top_left,
        guidance_rescale,
        timesteps,
        sigmas,
    )

    clip_models = load_clip_models()
    scores = []
    for i in range(num_prompts):
        score = compute_directional_similarity(
            dataset["original_image"][i],
            images[i],
            dataset["original_prompt"][i],
            dataset["edited_prompt"][i],
            clip_models,
        )
        scores.append(score.item())

    average_clip_score = sum(scores) / len(scores)
    deviation_clip_score = statistics.stdev(scores)
    logger.info(f"Average directional similarity: {average_clip_score}")

    model_name = "sdxl-img2img-tp" if use_cfg_parallel else "sdxl-img2img"
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
        "num_prompts": num_prompts,
    }

    report_json = {
        "model": model_name,
        "metadata": metadata,
        "benchmarks_summary": get_benchmark_summary(metadata),
        "evals": [
            {
                "model": model_name,
                "device": metadata["device"],
                "average_clip": average_clip_score,
                "deviation_clip": deviation_clip_score,
                "accuracy_check": accuracy_check_clip(model_name, average_clip_score, num_prompts, mode="approx"),
            }
        ],
    }

    save_report_json(report_json, metadata)
    print(json.dumps(report_json, indent=4))


def load_clip_models():
    clip_id = "openai/clip-vit-large-patch14"
    return {
        "tokenizer": CLIPTokenizer.from_pretrained(clip_id),
        "text_encoder": CLIPTextModelWithProjection.from_pretrained(clip_id),
        "image_processor": CLIPImageProcessor.from_pretrained(clip_id),
        "image_encoder": CLIPVisionModelWithProjection.from_pretrained(clip_id),
    }


def compute_directional_similarity(image_one, image_two, caption_one, caption_two, clip_models):
    img1 = clip_models["image_processor"](image_one, return_tensors="pt")["pixel_values"]
    img2 = clip_models["image_processor"](image_two, return_tensors="pt")["pixel_values"]

    img_feat_one = clip_models["image_encoder"](pixel_values=img1).image_embeds
    img_feat_two = clip_models["image_encoder"](pixel_values=img2).image_embeds

    img_feat_one = img_feat_one / img_feat_one.norm(dim=1, keepdim=True)
    img_feat_two = img_feat_two / img_feat_two.norm(dim=1, keepdim=True)

    tok1 = clip_models["tokenizer"](
        caption_one, max_length=77, padding="max_length", truncation=True, return_tensors="pt"
    )
    tok2 = clip_models["tokenizer"](
        caption_two, max_length=77, padding="max_length", truncation=True, return_tensors="pt"
    )

    text_feat_one = clip_models["text_encoder"](input_ids=tok1.input_ids).text_embeds
    text_feat_two = clip_models["text_encoder"](input_ids=tok2.input_ids).text_embeds

    text_feat_one = text_feat_one / text_feat_one.norm(dim=1, keepdim=True)
    text_feat_two = text_feat_two / text_feat_two.norm(dim=1, keepdim=True)

    sim_direction = F.cosine_similarity(img_feat_two - img_feat_one, text_feat_two - text_feat_one)
    return sim_direction
