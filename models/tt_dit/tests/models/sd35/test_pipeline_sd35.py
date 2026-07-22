# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import pytest
from loguru import logger

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler

from ....parallel.config import DiTParallelConfig
from ....pipelines.events import profiler_event_callback
from ....pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import (
    StableDiffusion3Pipeline,
    StableDiffusion3PipelineConfig,
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
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 50000000}],
    indirect=True,
)
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
    is_ci_env,
) -> None:
    """Test the new SD3.5 pipeline implementation."""

    model_version = f"stabilityai/stable-diffusion-3.5-{model_name}"

    # Setup CI environment
    if is_ci_env and traced:
        pytest.skip("Skipping traced test in CI environment. Use Performance test for detailed timing analysis.")

    pipeline = StableDiffusion3Pipeline(
        device=mesh_device,
        config=StableDiffusion3PipelineConfig.default(
            mesh_shape=mesh_device.shape,
            dit_parallel_config=DiTParallelConfig.from_tuples(cfg=cfg, sp=sp, tp=tp),
            topology=topology,
            num_links=num_links,
            width=image_w,
            height=image_h,
            checkpoint_name=model_location_generator(model_version, model_subdir="StableDiffusion_35_Large"),
        ),
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
        benchmark_profiler = BenchmarkProfiler()
        with benchmark_profiler("run", iteration=0):
            images = pipeline(
                prompts=[prompt],
                negative_prompts=[negative_prompt],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                traced=traced,
                vae_traced=False,
                encoder_traced=False,
                on_event=profiler_event_callback(benchmark_profiler, 0),
            )

        # Save image

        output_filename = f"sd35_new_{image_w}_{image_h}"
        if traced:
            output_filename += "_traced"
        output_filename += ".png"

        images[0].save(output_filename)
        logger.info(f"Image saved as {output_filename}")

        # Print timing information
        logger.info(f"CLIP encoding time: {benchmark_profiler.get_duration('clip_encoding', 0):.2f}s")
        logger.info(f"T5 encoding time: {benchmark_profiler.get_duration('t5_encoding', 0):.2f}s")
        logger.info(f"Total encoding time: {benchmark_profiler.get_duration('encoder', 0):.2f}s")
        logger.info(f"VAE decoding time: {benchmark_profiler.get_duration('vae', 0):.2f}s")
        logger.info(f"Total pipeline time: {benchmark_profiler.get_duration('total', 0):.2f}s")
        avg_step_time = benchmark_profiler.get_duration("denoising", 0) / num_inference_steps
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
                prompts=[prompt],
                negative_prompts=[negative_prompt],
                num_inference_steps=num_inference_steps,
                traced=traced,
                vae_traced=False,
                encoder_traced=False,
            )

            output_filename = f"sd35_new_{image_w}_{image_h}_{i:02}"
            if traced:
                output_filename += "_traced"
            output_filename += ".png"

            images[0].save(output_filename)
            logger.info(f"Image saved as {output_filename}")

    # Synchronize all devices
    for submesh_device in pipeline.submesh_devices:
        ttnn.synchronize_device(submesh_device)

    logger.info("SD3.5 pipeline test completed successfully!")
