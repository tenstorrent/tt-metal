# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import ttnn
import torch
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from loguru import logger
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    SDXL_L1_SMALL_SIZE,
    SDXL_BASE_REFINER_TRACE_REGION_SIZE,
    SDXL_FABRIC_CONFIG,
)
import os
from models.common.utility_functions import profiler
from conftest import is_galaxy

from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_combined_pipeline import (
    TtSDXLCombinedPipeline,
    TtSDXLCombinedPipelineConfig,
)


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
    use_refiner,
    denoising_split,
    refiner_strength,
    refiner_aesthetic_score,
    refiner_negative_aesthetic_score,
    prompt_2=None,
    negative_prompt_2=None,
    crop_coords_top_left=(0, 0),
    guidance_rescale=0.0,
    timesteps=None,
    sigmas=None,
):
    batch_size = list(ttnn_device.shape)[1] if use_cfg_parallel else ttnn_device.get_num_devices()

    start_from, _ = evaluation_range

    # Convert single prompts to lists
    if isinstance(prompts, str):
        prompts = [prompts]

    if prompt_2 is not None and isinstance(prompt_2, str):
        prompt_2 = [prompt_2]

    if isinstance(negative_prompts, list):
        assert len(negative_prompts) == len(prompts), "prompts and negative_prompt lists must be the same length"

    # 1. Load base and refiner torch pipelines
    profiler.start("diffusion_pipeline_from_pretrained")
    base_pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_env,
    )

    refiner_pipeline = None
    if use_refiner:
        refiner_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float32,
            use_safetensors=True,
            local_files_only=is_ci_env,
            text_encoder_2=base_pipeline.text_encoder_2,
            vae=base_pipeline.vae,
        )
    profiler.end("diffusion_pipeline_from_pretrained")

    # 2. Create unified config and combined pipeline
    config = TtSDXLCombinedPipelineConfig(
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        is_galaxy=is_galaxy(),
        use_cfg_parallel=use_cfg_parallel,
        vae_on_device=vae_on_device,
        encoders_on_device=encoders_on_device,
        capture_trace=capture_trace,
        crop_coords_top_left=crop_coords_top_left,
        guidance_rescale=guidance_rescale,
        use_refiner=use_refiner,
        denoising_split=denoising_split,
        strength=refiner_strength,
        aesthetic_score=refiner_aesthetic_score,
        negative_aesthetic_score=refiner_negative_aesthetic_score,
    )

    tt_sdxl_combined = TtSDXLCombinedPipeline(
        ttnn_device=ttnn_device,
        torch_base_pipeline=base_pipeline,
        torch_refiner_pipeline=refiner_pipeline,
        config=config,
    )

    logger.info("=" * 80)
    for key, data in profiler.times.items():
        logger.info(f"{key}: {data[-1]:.2f} seconds")
    logger.info("=" * 80)

    profiler.clear()

    if not is_ci_env and not os.path.exists("output"):
        os.mkdir("output")

    images = []
    logger.info("Starting ttnn inference...")

    for iter in range(len(prompts) // batch_size + (1 if len(prompts) % batch_size != 0 else 0)):
        logger.info(
            f"Running inference for prompts {iter * batch_size + 1}-{min((iter + 1) * batch_size, len(prompts))}/{len(prompts)}"
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

        # Combined pipeline will pad the batch internally if needed
        imgs = tt_sdxl_combined.generate(
            prompts=prompts_batch,
            negative_prompts=negative_prompts_batch,
            prompt_2=prompts_2_batch,
            negative_prompt_2=negative_prompts_2_batch,
            start_latent_seed=0,
            fixed_seed_for_batch=fixed_seed_for_batch,
            timesteps=timesteps,
            sigmas=sigmas,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            crop_coords_top_left=crop_coords_top_left,
            guidance_rescale=guidance_rescale,
        )

        profiler.end("end_to_end_generation")
        if iter == 0:
            profiler.times["end_to_end_generation"][0] -= profiler.times["auto_compile_if_needed"][0]
            for key in profiler.times.keys():
                profiler.times[key] = [profiler.times[key][-1]]

        logger.info(
            f"Combined generation for batch {iter + 1} completed in {profiler.times['end_to_end_generation'][-1]:.2f} seconds"
        )

        for idx, img in enumerate(imgs):
            if iter * batch_size + idx >= len(prompts):
                break

            img = img.unsqueeze(0)
            pil_img = base_pipeline.image_processor.postprocess(img, output_type="pil")[0]
            images.append(pil_img)

            if is_ci_env:
                logger.info(f"Image {len(images)}/{len(prompts)} generated successfully")
            else:
                output_path = f"output/output{len(images) - 1 + start_from}.png"
                pil_img.save(output_path)
                logger.info(f"Image saved to {output_path}")

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
@pytest.mark.parametrize(
    "use_refiner",
    [
        (True),
    ],
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
@pytest.mark.parametrize(
    "prompt_2, negative_prompt_2, crop_coords_top_left, guidance_rescale, timesteps, sigmas",
    [
        (None, None, (0, 0), 0.0, None, None),
    ],
    ids=["default_additional_parameters"],
)
def test_demo_base_and_refiner(
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
    use_refiner,
    denoising_split,
    refiner_strength,
    refiner_aesthetic_score,
    refiner_negative_aesthetic_score,
    prompt_2,
    negative_prompt_2,
    crop_coords_top_left,
    guidance_rescale,
    timesteps,
    sigmas,
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
        use_refiner,
        denoising_split,
        refiner_strength,
        refiner_aesthetic_score,
        refiner_negative_aesthetic_score,
        prompt_2,
        negative_prompt_2,
        crop_coords_top_left,
        guidance_rescale,
        timesteps,
        sigmas,
    )
