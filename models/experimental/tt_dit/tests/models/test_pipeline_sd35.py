# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import pytest
import ttnn
from loguru import logger

from ...pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import (
    create_pipeline,
    TimingCollector,
)


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
        "2x4cfg1sp0tp1",
        "2x4cfg0sp0tp1",
        "4x8cfg1sp0tp1",
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
    is_ci_env,
    monkeypatch,
) -> None:
    """Test the new SD3.5 pipeline implementation."""

    model_version = f"stabilityai/stable-diffusion-3.5-{model_name}"

    # Setup CI environment
    if is_ci_env:
        if use_cache:
            monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")
        else:
            pytest.skip("Skipping. No use cache is implicitly tested with the configured non persistent cache path.")
        if traced:
            pytest.skip("Skipping traced test in CI environment. Use Performance test for detailed timing analysis.")

    # Create timing collector
    timing_collector = TimingCollector()

    # Create pipeline
    pipeline = create_pipeline(
        mesh_device=mesh_device,
        batch_size=1,
        image_w=image_w,
        image_h=image_h,
        guidance_scale=guidance_scale,
        prompt_sequence_length=333,
        spatial_sequence_length=4096,
        max_t5_sequence_length=256,
        cfg_config=cfg,
        sp_config=sp,
        tp_config=tp,
        num_links=num_links,
        model_checkpoint_path=model_location_generator(model_version, model_subdir="StableDiffusion_35_Large"),
        use_cache=use_cache,
    )

    # Set timing collector
    pipeline.timing_collector = timing_collector

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
