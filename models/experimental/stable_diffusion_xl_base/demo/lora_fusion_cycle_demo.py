# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import ttnn
import torch
from diffusers import DiffusionPipeline
from loguru import logger
from transformers import CLIPTextModelWithProjection, CLIPTextModel
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    SDXL_L1_SMALL_SIZE,
    SDXL_TRACE_REGION_SIZE,
    MAX_SEQUENCE_LENGTH,
    TEXT_ENCODER_2_PROJECTION_DIM,
    CONCATENATED_TEXT_EMBEDINGS_SIZE,
    prepare_device,
)
import os
from conftest import is_galaxy

from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_pipeline import TtSDXLPipeline, TtSDXLPipelineConfig


LORA_PATH = "lora_weights/ColoringBookRedmond-ColoringBook-ColoringBookAF.safetensors"

PROMPTS = [
    ("An astronaut riding a green horse", "A Coloring Book of an astronaut riding a green horse"),
    ("A city at night with people walking around.", "A Coloring Book of a city at night with people walking around."),
    ("An apple", "A Coloring Book of an apple"),
]


def _run_forward(
    tt_sdxl, pipeline, prompt, negative_prompt, prompt_2, negative_prompt_2, fixed_seed_for_batch, timesteps, sigmas
):
    """Encode one prompt, run one forward pass, return the generated image (batch size 1)."""
    all_prompt_embeds_torch, torch_add_text_embeds = tt_sdxl.encode_prompts(
        [prompt], negative_prompt, [prompt_2] if prompt_2 else None, [negative_prompt_2] if negative_prompt_2 else None
    )
    tt_latents, tt_prompt_embeds, tt_add_text_embeds = tt_sdxl.generate_input_tensors(
        all_prompt_embeds_torch,
        torch_add_text_embeds,
        start_latent_seed=0,
        fixed_seed_for_batch=fixed_seed_for_batch,
        timesteps=timesteps,
        sigmas=sigmas,
    )
    tt_sdxl.prepare_input_tensors([tt_latents, tt_prompt_embeds[0], tt_add_text_embeds[0]])
    imgs = tt_sdxl.generate_images()
    img = imgs[0].unsqueeze(0)
    return pipeline.image_processor.postprocess(img, output_type="pil")[0]


@torch.no_grad()
def run_demo_inference(
    ttnn_device,
    is_ci_env,
    prompts,
    lora_prompts,
    negative_prompt,
    num_inference_steps,
    vae_on_device,
    encoders_on_device,
    evaluation_range,
    capture_trace,
    guidance_scale,
    fixed_seed_for_batch,
    prompt_2=None,
    negative_prompt_2=None,
    crop_coords_top_left=(0, 0),
    guidance_rescale=0.0,
    timesteps=None,
    sigmas=None,
):
    assert 0.0 <= guidance_rescale <= 1.0, f"guidance_rescale must be in [0.0, 1.0], got {guidance_rescale}"
    assert not (timesteps is not None and sigmas is not None), "Cannot pass both timesteps and sigmas. Choose one."

    if isinstance(prompts, str):
        prompts = [prompts]
    if isinstance(lora_prompts, str):
        lora_prompts = [lora_prompts]
    assert len(prompts) == len(lora_prompts), "prompts and lora_prompts must have the same length"

    start_from, _ = evaluation_range

    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_env,
    )
    assert isinstance(pipeline.text_encoder, CLIPTextModel), "pipeline.text_encoder is not a CLIPTextModel"
    assert isinstance(
        pipeline.text_encoder_2, CLIPTextModelWithProjection
    ), "pipeline.text_encoder_2 is not a CLIPTextModelWithProjection"

    tt_sdxl = TtSDXLPipeline(
        ttnn_device=ttnn_device,
        torch_pipeline=pipeline,
        pipeline_config=TtSDXLPipelineConfig(
            capture_trace=capture_trace,
            vae_on_device=vae_on_device,
            encoders_on_device=encoders_on_device,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            is_galaxy=is_galaxy(),
            use_cfg_parallel=False,
            crop_coords_top_left=crop_coords_top_left,
            guidance_rescale=guidance_rescale,
        ),
    )

    if encoders_on_device:
        tt_sdxl.compile_text_encoding()

    tt_sdxl.generate_input_tensors(
        all_prompt_embeds_torch=torch.randn(1, 2, MAX_SEQUENCE_LENGTH, CONCATENATED_TEXT_EMBEDINGS_SIZE),
        torch_add_text_embeds=torch.randn(1, 2, TEXT_ENCODER_2_PROJECTION_DIM),
        timesteps=timesteps,
        sigmas=sigmas,
    )
    tt_sdxl.compile_image_processing()

    if not is_ci_env and not os.path.exists("output"):
        os.mkdir("output")

    skip_saving = os.getenv("TT_SDXL_SKIP_CHECK_AND_SAVE", "0") == "1"
    all_images = []

    for i, (prompt, lora_prompt) in enumerate(zip(prompts, lora_prompts)):
        idx = start_from + i + 1
        logger.info(f"Prompt {i + 1}/{len(prompts)}: {prompt[:50]}...")

        # 1. Forward without LoRA
        img_base = _run_forward(
            tt_sdxl,
            pipeline,
            prompt,
            negative_prompt,
            prompt_2,
            negative_prompt_2,
            fixed_seed_for_batch,
            timesteps,
            sigmas,
        )
        all_images.append(img_base)
        if not is_ci_env and not skip_saving:
            img_base.save(f"output/output_base_{idx}.png")
            logger.info(f"Saved output/output_base_{idx}.png")

        # 2. Load LoRA, fuse, forward with LoRA
        tt_sdxl.load_lora_weights(LORA_PATH)
        tt_sdxl.fuse_lora()
        img_lora = _run_forward(
            tt_sdxl,
            pipeline,
            lora_prompt,
            negative_prompt,
            prompt_2,
            negative_prompt_2,
            fixed_seed_for_batch,
            timesteps,
            sigmas,
        )
        all_images.append(img_lora)
        if not is_ci_env and not skip_saving:
            img_lora.save(f"output/output_lora_{idx}.png")
            logger.info(f"Saved output/output_lora_{idx}.png")

        # 3. Unload LoRA, forward again (same as first)
        tt_sdxl.rollback_base_weights()
        img_rollback = _run_forward(
            tt_sdxl,
            pipeline,
            prompt,
            negative_prompt,
            prompt_2,
            negative_prompt_2,
            fixed_seed_for_batch,
            timesteps,
            sigmas,
        )
        all_images.append(img_rollback)
        if not is_ci_env and not skip_saving:
            img_rollback.save(f"output/output_rollback_{idx}.png")
            logger.info(f"Saved output/output_rollback_{idx}.png")

    ttnn.synchronize_device(ttnn_device)
    return all_images


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": SDXL_L1_SMALL_SIZE, "trace_region_size": SDXL_TRACE_REGION_SIZE}],
    indirect=["device_params"],
)
def test_demo(validate_fabric_compatibility, mesh_device, is_ci_env, evaluation_range):
    """Set pipeline once, then loop over prompts: base → load/fuse lora → lora → unload → base."""
    prepare_device(mesh_device, use_cfg_parallel=False)
    prompts, lora_prompts = zip(*PROMPTS)
    return run_demo_inference(
        mesh_device,
        is_ci_env,
        list(prompts),
        list(lora_prompts),
        negative_prompt="disturbing",
        num_inference_steps=50,
        vae_on_device=True,
        encoders_on_device=True,
        evaluation_range=evaluation_range,
        capture_trace=False,
        guidance_scale=5.0,
        fixed_seed_for_batch=False,
        prompt_2=None,
        negative_prompt_2=None,
        crop_coords_top_left=(0, 0),
        guidance_rescale=0.0,
        timesteps=None,
        sigmas=None,
    )
