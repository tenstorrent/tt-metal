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

MAX_SEQUENCE_LENGTH = 77
TEXT_ENCODER_2_PROJECTION_DIM = 1280
CONCATENATED_TEXT_EMBEDINGS_SIZE = 2048  # text_encoder_1_hidden_size + text_encoder_2_hidden_size (768 + 1280)


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
    use_cfg_parallel,
    fixed_seed_for_batch,
):
    batch_size = list(ttnn_device.shape)[1] if use_cfg_parallel else ttnn_device.get_num_devices()

    start_from, _ = evaluation_range

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
            use_cfg_parallel=use_cfg_parallel,
        ),
    )

    if encoders_on_device:
        tt_sdxl.compile_text_encoding()

    tt_latents, tt_prompt_embeds, tt_add_text_embeds = tt_sdxl.generate_input_tensors(
        all_prompt_embeds_torch=torch.randn(batch_size, 2, MAX_SEQUENCE_LENGTH, CONCATENATED_TEXT_EMBEDINGS_SIZE),
        torch_add_text_embeds=torch.randn(batch_size, 2, TEXT_ENCODER_2_PROJECTION_DIM),
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
        profiler.start("end_to_end_generation")
        logger.info(
            f"Running inference for prompts {iter * batch_size + 1}-{iter * batch_size + batch_size}/{len(prompts)}"
        )

        prompts_batch = prompts[iter * batch_size : (iter + 1) * batch_size]
        negative_prompts_batch = (
            negative_prompts[iter * batch_size : (iter + 1) * batch_size]
            if isinstance(negative_prompts, list)
            else negative_prompts
        )

        (
            all_prompt_embeds_torch,
            torch_add_text_embeds,
        ) = tt_sdxl.encode_prompts(prompts_batch, negative_prompts_batch)

        tt_latents, tt_prompt_embeds, tt_add_text_embeds = tt_sdxl.generate_input_tensors(
            all_prompt_embeds_torch,
            torch_add_text_embeds,
            start_latent_seed=0,
            fixed_seed_for_batch=fixed_seed_for_batch,
        )

        tt_sdxl.prepare_input_tensors(
            [
                tt_latents,
                tt_prompt_embeds[0],
                tt_add_text_embeds[0],
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

        profiler.end("end_to_end_generation")

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


def prepare_device(mesh_device, use_cfg_parallel):
    if use_cfg_parallel:
        assert mesh_device.get_num_devices() % 2 == 0, "Mesh device must have even number of devices"
        mesh_device.reshape(ttnn.MeshShape(2, mesh_device.get_num_devices() // 2))


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
    "fixed_seed_for_batch",
    (False,),
)
@pytest.mark.parametrize(
    "prompt",
    (("An astronaut riding a green horse"),),
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
def test_demo(
    validate_fabric_compatibility,
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
    use_cfg_parallel,
    fixed_seed_for_batch,
):
    prepare_device(mesh_device, use_cfg_parallel)
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
        use_cfg_parallel,
        fixed_seed_for_batch,
    )
