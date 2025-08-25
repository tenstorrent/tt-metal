# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import pytest
import ttnn
from loguru import logger

from ...pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import (
    StableDiffusion3Pipeline,
    TimingCollector,
)
from ...parallel.config import DiTParallelConfig, ParallelFactor


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(
    "model_name, image_w, image_h, guidance_scale, num_inference_steps",
    [
        ("large", 1024, 1024, 3.5, 28),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, cfg, sp, tp, topology, num_links",
    [
        [(2, 4), (2, 1), (2, 0), (2, 1), ttnn.Topology.Linear, 1],
        [(2, 4), (2, 0), (1, 0), (4, 1), ttnn.Topology.Linear, 1],
        [(4, 8), (2, 1), (4, 0), (4, 1), ttnn.Topology.Linear, 4],
    ],
    ids=[
        "t3k_cfg2_sp2_tp2",
        "t3k_cfg2_sp1_tp4",
        "tg_cfg2_sp4_tp4",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 25000000}],
    indirect=True,
)
@pytest.mark.parametrize("use_cache", [True, False], ids=["yes_use_cache", "no_use_cache"])
@pytest.mark.parametrize("traced", [True, False], ids=["yes_traced", "no_traced"])
def test_sd35_pipeline(
    *,
    mesh_device: ttnn.MeshDevice,
    model_name,
    image_w,
    image_h,
    guidance_scale,
    num_inference_steps,
    cfg,
    sp,
    tp,
    topology,
    num_links,
    no_prompt,
    model_location_generator,
    traced,
    use_cache,
) -> None:
    """Test the new SD3.5 pipeline implementation."""
    cfg_factor, cfg_axis = cfg
    sp_factor, sp_axis = sp
    tp_factor, tp_axis = tp

    # Create parallel configuration for the new pipeline
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=cfg_factor, mesh_axis=cfg_axis),
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )

    # Determine guidance condition
    if guidance_scale > 1 and cfg_factor == 1:
        guidance_cond = 2
    else:
        guidance_cond = 1

    # Enable T5 based on device configuration
    # T5 is disabled if mesh needs reshaping for CLIP encoder
    submesh_shape = list(mesh_device.shape)
    submesh_shape[cfg_axis] //= cfg_factor
    enable_t5_text_encoder = submesh_shape[1] == 4  # T5 only works if submesh doesn't need reshaping

    logger.info(f"Mesh device shape: {mesh_device.shape}")
    logger.info(f"Submesh shape: {submesh_shape}")
    logger.info(f"Parallel config: {parallel_config}")
    logger.info(f"T5 enabled: {enable_t5_text_encoder}")

    # Create timing collector
    timing_collector = TimingCollector()

    # Create pipeline
    pipeline = StableDiffusion3Pipeline(
        checkpoint_name=f"stabilityai/stable-diffusion-3.5-{model_name}",
        mesh_device=mesh_device,
        enable_t5_text_encoder=enable_t5_text_encoder,
        guidance_cond=guidance_cond,
        parallel_config=parallel_config,
        num_links=num_links,
        height=image_h,
        width=image_w,
        model_location_generator=model_location_generator,
        use_cache=use_cache,
    )

    # Set timing collector
    pipeline.timing_collector = timing_collector

    # Prepare pipeline
    pipeline.prepare(
        batch_size=1,
        width=image_w,
        height=image_h,
        guidance_scale=guidance_scale,
        prompt_sequence_length=333,
        spatial_sequence_length=4096,
    )

    # Define test prompt
    prompt = (
        "An epic, high-definition cinematic shot of a rustic snowy cabin glowing "
        "warmly at dusk, nestled in a serene winter landscape. Surrounded by gentle "
        "snow-covered pines and delicate falling snowflakes - captured in a rich, "
        "atmospheric, wide-angle scene with deep cinematic depth and warmth."
    )

    if no_prompt:
        # Run single generation
        negative_prompt = ""
        images = pipeline(
            prompt_1=[prompt],
            prompt_2=[prompt],
            prompt_3=[prompt],
            negative_prompt_1=[negative_prompt],
            negative_prompt_2=[negative_prompt],
            negative_prompt_3=[negative_prompt],
            num_inference_steps=num_inference_steps,
            seed=0,
            traced=traced,
        )

        # Save image
        output_filename = f"sd35_new_{image_w}_{image_h}.png"
        images[0].save(output_filename)
        logger.info(f"Image saved as {output_filename}")

        # Print timing information
        timing_data = timing_collector.get_timing_data()
        logger.info(f"CLIP encoding time: {timing_data.clip_encoding_time:.2f}s")
        logger.info(f"T5 encoding time: {timing_data.t5_encoding_time:.2f}s")
        logger.info(f"Total encoding time: {timing_data.total_encoding_time:.2f}s")
        logger.info(f"VAE decoding time: {timing_data.vae_decoding_time:.2f}s")
        logger.info(f"Total pipeline time: {timing_data.total_time:.2f}s")
        if timing_data.denoising_step_times:
            avg_step_time = sum(timing_data.denoising_step_times) / len(timing_data.denoising_step_times)
            logger.info(f"Average denoising step time: {avg_step_time:.2f}s")

    else:
        # Interactive demo
        for i in itertools.count():
            new_prompt = input("Enter the input prompt, or q to exit: ")
            if new_prompt:
                prompt = new_prompt
            if prompt[0] == "q":
                break

            negative_prompt = ""

            images = pipeline(
                prompt_1=[prompt],
                prompt_2=[prompt],
                prompt_3=[prompt],
                negative_prompt_1=[negative_prompt],
                negative_prompt_2=[negative_prompt],
                negative_prompt_3=[negative_prompt],
                num_inference_steps=num_inference_steps,
                seed=0,
                traced=traced,
            )

            output_filename = f"sd35_new_{image_w}_{image_h}_{i}.png"
            images[0].save(output_filename)
            logger.info(f"Image saved as {output_filename}")

    # Synchronize all devices
    for submesh_device in pipeline.submesh_devices:
        ttnn.synchronize_device(submesh_device)

    logger.info("SD3.5 pipeline test completed successfully!")
