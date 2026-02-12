# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
from loguru import logger
from transformers import CLIPTextModelWithProjection
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    SDXL_L1_SMALL_SIZE,
    SDXL_TRACE_REGION_SIZE,
    SDXL_FABRIC_CONFIG,
    MAX_SEQUENCE_LENGTH,
    TEXT_ENCODER_2_PROJECTION_DIM,
    CONCATENATED_TEXT_EMBEDINGS_SIZE_REFINER,
    determinate_min_batch_size,
    prepare_device,
)
import os
from models.common.utility_functions import profiler
from conftest import is_galaxy

from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_img2img_pipeline import (
    TtSDXLImg2ImgPipeline,
    TtSDXLImg2ImgPipelineConfig,
)


@torch.no_grad()
def run_demo_inference(
    ttnn_device,
    is_ci_env,
    prompts,
    images,
    negative_prompts,
    num_inference_steps,
    vae_on_device,
    encoders_on_device,
    evaluation_range,
    capture_trace,
    guidance_scale,
    use_cfg_parallel,
    fixed_seed_for_batch,
    strength,
    prompt_2=None,
    negative_prompt_2=None,
    crop_coords_top_left=(0, 0),
    guidance_rescale=0.0,
    timesteps=None,
    sigmas=None,
):
    batch_size = determinate_min_batch_size(ttnn_device, use_cfg_parallel)

    start_from, _ = evaluation_range

    assert 0.0 <= guidance_rescale <= 1.0, f"guidance_rescale must be in [0.0, 1.0], got {guidance_rescale}"

    assert not (timesteps is not None and sigmas is not None), "Cannot pass both timesteps and sigmas. Choose one."

    if isinstance(prompts, str):
        prompts = [prompts]

    if prompt_2 is not None and isinstance(prompt_2, str):
        prompt_2 = [prompt_2]

    needed_padding = (batch_size - len(prompts) % batch_size) % batch_size
    if isinstance(negative_prompts, list):
        assert len(negative_prompts) == len(prompts), "prompts and negative_prompt lists must be the same length"

    prompts = prompts + [""] * needed_padding
    if prompt_2 is not None:
        prompt_2 = prompt_2 + [""] * needed_padding
    if isinstance(negative_prompts, list):
        negative_prompts = negative_prompts + [""] * needed_padding

    # 1. Load components
    profiler.start("diffusion_pipeline_from_pretrained")
    pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_env,
    ).to("cpu")
    profiler.end("diffusion_pipeline_from_pretrained")

    assert isinstance(
        pipeline.text_encoder_2, CLIPTextModelWithProjection
    ), "pipeline.text_encoder_2 is not a CLIPTextModelWithProjection"

    tt_sdxl = TtSDXLImg2ImgPipeline(
        ttnn_device=ttnn_device,
        torch_pipeline=pipeline,
        pipeline_config=TtSDXLImg2ImgPipelineConfig(
            capture_trace=capture_trace,
            vae_on_device=vae_on_device,
            encoders_on_device=encoders_on_device,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            is_galaxy=is_galaxy(),
            use_cfg_parallel=use_cfg_parallel,
            crop_coords_top_left=crop_coords_top_left,
            guidance_rescale=guidance_rescale,
        ),
    )

    if encoders_on_device:
        tt_sdxl.compile_text_encoding()

    images = images + [images[0]] * needed_padding
    images = [
        tt_sdxl.torch_pipeline.image_processor.preprocess(
            image, height=1024, width=1024, crops_coords=None, resize_mode="default"
        ).to(dtype=torch.float32)
        for image in images
    ]

    images = torch.cat(images, dim=0)  # [batch_size, 3, 1024, 1024]

    tt_latents, tt_prompt_embeds, tt_add_text_embeds = tt_sdxl.generate_input_tensors(
        torch_image=torch.randn(batch_size, 3, 1024, 1024),
        all_prompt_embeds_torch=torch.randn(
            batch_size, 2, MAX_SEQUENCE_LENGTH, CONCATENATED_TEXT_EMBEDINGS_SIZE_REFINER
        ),
        torch_add_text_embeds=torch.randn(batch_size, 2, TEXT_ENCODER_2_PROJECTION_DIM),
        timesteps=timesteps,
        sigmas=sigmas,
    )

    tt_sdxl.compile_image_processing()

    logger.info("=" * 80)
    for key, data in profiler.times.items():
        logger.info(f"{key}: {data[-1]:.2f} seconds")
    logger.info("=" * 80)

    profiler.clear()

    if not is_ci_env and not os.path.exists("output"):
        os.mkdir("output")

    out_images = []
    logger.info("Starting ttnn inference...")
    for iter in range(len(prompts) // batch_size):
        logger.info(
            f"Running inference for prompts {iter * batch_size + 1}-{iter * batch_size + batch_size}/{len(prompts)}"
        )

        prompts_batch = prompts[iter * batch_size : (iter + 1) * batch_size]
        negative_prompts_batch = (
            negative_prompts[iter * batch_size : (iter + 1) * batch_size]
            if isinstance(negative_prompts, list)
            else negative_prompts
        )

        prompts_2_batch = (
            prompt_2[iter * batch_size : (iter + 1) * batch_size] if isinstance(prompt_2, list) else prompt_2
        )
        negative_prompts_2_batch = (
            negative_prompt_2[iter * batch_size : (iter + 1) * batch_size]
            if isinstance(negative_prompt_2, list)
            else negative_prompt_2
        )

        profiler.start("end_to_end_generation")
        (
            all_prompt_embeds_torch,
            torch_add_text_embeds,
        ) = tt_sdxl.encode_prompts(prompts_batch, negative_prompts_batch, prompts_2_batch, negative_prompts_2_batch)

        tt_latents, tt_prompt_embeds, tt_add_text_embeds = tt_sdxl.generate_input_tensors(
            torch_image=images[iter * batch_size : (iter + 1) * batch_size],
            all_prompt_embeds_torch=all_prompt_embeds_torch,
            torch_add_text_embeds=torch_add_text_embeds,
            start_latent_seed=0,
            fixed_seed_for_batch=fixed_seed_for_batch,
            timesteps=timesteps,
            sigmas=sigmas,
        )

        tt_sdxl.prepare_input_tensors(
            [
                tt_latents,
                tt_prompt_embeds[0],
                tt_add_text_embeds[0],
            ]
        )

        imgs = tt_sdxl.generate_images()

        profiler.end("end_to_end_generation")

        logger.info(
            f"Prepare input tensors for {batch_size} prompts completed in {profiler.times['prepare_input_tensors'][-1]:.2f} seconds"
        )
        logger.info(
            f"Image gen for {batch_size} prompts completed in {profiler.times['end_to_end_generation'][-1]:.2f} seconds"
        )
        logger.info(
            f"Denoising loop for {batch_size} prompts completed in {profiler.times['denoising_loop'][-1]:.2f} seconds"
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
            out_images.append(img)
            if is_ci_env:
                logger.info(f"Image {len(out_images)}/{len(prompts) // batch_size} generated successfully")
            else:
                img.save(f"output/output{len(out_images) + start_from}_tt_img2img.png")
                logger.info(f"Image saved to output/output{len(out_images) + start_from}_tt_img2img.png")

    return out_images


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
    (("An astronaut riding a red dragon in space, cinematic lighting"),),
)
@pytest.mark.parametrize(
    "images_or_path",
    (("models/experimental/stable_diffusion_xl_base/reference/output/sdxl_output.jpg"),),
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
@pytest.mark.parametrize(
    "strength",
    ((0.3),),
)
@pytest.mark.parametrize(
    "prompt_2, negative_prompt_2, crop_coords_top_left, guidance_rescale, timesteps, sigmas",
    [
        (None, None, (0, 0), 0.0, None, None),
    ],
    ids=["default_additional_parameters"],
)
def test_demo(
    validate_fabric_compatibility,
    mesh_device,
    is_ci_env,
    prompt,
    images_or_path,
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
    if isinstance(images_or_path, str):
        images = [Image.open(images_or_path).convert("RGB")]
    else:
        images = images_or_path if isinstance(images_or_path, list) else [images_or_path]

    prepare_device(mesh_device, use_cfg_parallel)
    return run_demo_inference(
        mesh_device,
        is_ci_env,
        prompt,
        images,
        negative_prompt,
        num_inference_steps,
        vae_on_device,
        encoders_on_device,
        evaluation_range,
        capture_trace,
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
