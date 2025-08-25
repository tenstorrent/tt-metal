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
    SDXL_FABRIC_CONFIG,
)
import os
from models.common.utility_functions import profiler
from conftest import is_galaxy

from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_pipeline import TtSDXLPipeline, TtSDXLPipelineConfig


@torch.no_grad()
def run_demo_inference(
    ttnn_device,
    is_ci_env,
    prompts,
    negative_prompts,
    num_inference_steps,
    vae_on_device,
    encoders_on_device,
    evaluation_range,
    capture_trace,
    guidance_scale,
    use_tp,
):
    batch_size = list(ttnn_device.shape)[1] if use_tp else ttnn_device.get_num_devices()

    start_from, _ = evaluation_range
    torch.manual_seed(0)

    if isinstance(prompts, str):
        prompts = [prompts]

    needed_padding = (batch_size - len(prompts) % batch_size) % batch_size
    if isinstance(negative_prompts, list):
        assert len(negative_prompts) == len(prompts), "prompts and negative_prompt lists must be the same length"

    prompts = prompts + [""] * needed_padding
    if isinstance(negative_prompts, list):
        negative_prompts = negative_prompts + [""] * needed_padding

    # 1. Load components
    profiler.start("diffusion_pipeline_from_pretrained")
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    profiler.end("diffusion_pipeline_from_pretrained")

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
            use_tp=use_tp,
        ),
    )

    if encoders_on_device:
        tt_sdxl.compile_text_encoding()
    (
        all_prompt_embeds_torch,  # tuple len = prompts/batch_size of tensors of shape (2, 77, 2048)
        torch_add_text_embeds,  # tuple len = prompts/batch_size of tensors of shape (2, 77, 2048)
    ) = tt_sdxl.encode_prompts(prompts, negative_prompts)

    # prompt_embeds_torch is tuple of len = prompts/batch_size of tensors of shape (batch_size, 77, 2048)

    # prompt_embeds_torch = torch.stack(prompt_embeds_torch, dim=0) # not sure if this is right, should be (prompts/batch_size, 2, 77, 2048)
    # negative_prompt_embeds_torch = torch.stack(negative_prompt_embeds_torch, dim=0) # which should be split into 2 tensors of shape (prompts/batch_size, 1, 77, 2048)
    # pooled_prompt_embeds_torch = torch.stack(pooled_prompt_embeds_torch, dim=0)
    # negative_pooled_prompt_embeds_torch = torch.stack(negative_pooled_prompt_embeds_torch, dim=0)

    tt_latents, tt_prompt_embeds, tt_add_text_embeds = tt_sdxl.generate_input_tensors(
        all_prompt_embeds_torch,
        torch_add_text_embeds,
    )

    tt_sdxl.prepare_input_tensors(
        [
            tt_latents,
            tt_prompt_embeds[0],
            tt_add_text_embeds[0],
        ]
    )
    tt_sdxl.compile_image_processing()

    logger.info("=" * 80)
    for key, data in profiler.times.items():
        logger.info(f"{key}: {data[-1]:.2f} seconds")
    logger.info("=" * 80)

    profiler.clear()

    if not is_ci_env and not os.path.exists("output"):
        os.mkdir("output")

    images = []
    logger.info("Starting ttnn inference...")
    for iter in range(len(prompts) // batch_size):
        logger.info(
            f"Running inference for prompts {iter * batch_size + 1}-{iter * batch_size + batch_size}/{len(prompts)}"
        )

        tt_sdxl.prepare_input_tensors(
            [
                tt_latents,
                tt_prompt_embeds[iter],
                tt_add_text_embeds[iter],
            ]
        )
        imgs = tt_sdxl.generate_images()

        logger.info(
            f"Prepare input tensors for {batch_size} prompts completed in {profiler.times['prepare_input_tensors'][-1]:.2f} seconds"
        )
        logger.info(f"Image gen for {batch_size} prompts completed in {profiler.times['image_gen'][-1]:.2f} seconds")
        logger.info(
            f"Denoising loop for {batch_size} promts completed in {profiler.times['denoising_loop'][-1]:.2f} seconds"
        )
        logger.info(
            f"{'On device VAE' if vae_on_device else 'Host VAE'} decoding completed in {profiler.times['vae_decode'][-1]:.2f} seconds"
        )
        logger.info(f"Output tensor read completed in {profiler.times['read_output_tensor'][-1]:.2f} seconds")

        for idx, img in enumerate(imgs):
            if iter == len(prompts) // batch_size - 1 and idx >= batch_size - needed_padding:
                break
            img = img.unsqueeze(0)
            img = pipeline.image_processor.postprocess(img, output_type="pil")[0]
            images.append(img)
            if is_ci_env:
                logger.info(f"Image {len(images)}/{len(prompts) // batch_size} generated successfully")
            else:
                img.save(f"output/output{len(images) + start_from}.png")
                logger.info(f"Image saved to output/output{len(images) + start_from}.png")

    return images


def prepare_device(mesh_device, use_tp):
    if use_tp:
        assert mesh_device.get_num_devices() % 2 == 0, "Mesh device must have even number of devices"
        mesh_device.reshape(ttnn.MeshShape(2, mesh_device.get_num_devices() // 2))


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": SDXL_L1_SMALL_SIZE,
            "trace_region_size": SDXL_TRACE_REGION_SIZE,
            "fabric_config": SDXL_FABRIC_CONFIG,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "prompt",
    (["An astronaut riding a green horse", "A dog in a red hat"],),
)
@pytest.mark.parametrize(
    "negative_prompt",
    ((None),),
)
@pytest.mark.parametrize(
    "num_inference_steps",
    ((50),),
)
@pytest.mark.parametrize(
    "guidance_scale",
    ((5.0),),
)
@pytest.mark.parametrize(
    "vae_on_device",
    [
        (True),
    ],
)
@pytest.mark.parametrize(
    "encoders_on_device",
    [
        (False),
    ],
)
@pytest.mark.parametrize(
    "capture_trace",
    [
        (True),
    ],
)
@pytest.mark.parametrize(
    "use_tp",
    [
        (True),
    ],
)
def test_demo(
    mesh_device,
    is_ci_env,
    prompt,
    negative_prompt,
    num_inference_steps,
    vae_on_device,
    encoders_on_device,
    capture_trace,
    evaluation_range,
    guidance_scale,
    use_tp,
):
    prepare_device(mesh_device, use_tp)
    return run_demo_inference(
        mesh_device,
        is_ci_env,
        prompt,
        negative_prompt,
        num_inference_steps,
        vae_on_device,
        encoders_on_device,
        evaluation_range,
        capture_trace,
        guidance_scale,
        use_tp,
    )
