# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import time
import pytest
from loguru import logger
import ttnn
from ..tt.fun_pipeline import TtStableDiffusion3Pipeline
from ..tt.utils import create_parallel_configs


@pytest.mark.parametrize(
    "model_name, image_w, image_h, guidance_scale, num_inference_steps",
    [
        ("large", 1024, 1024, 3.5, 20),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, cfg, sp, tp, topology, num_links",
    [
        [(2, 4), (2, 1), (2, 0), (2, 1), ttnn.Topology.Linear, 1],
        [(4, 8), (2, 1), (4, 0), (4, 1), ttnn.Topology.Linear, 4],
    ],
    ids=[
        "t3k_cfg2_sp2_tp2",
        "tg_cfg2_sp4_tp4",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 25000000}],
    indirect=True,
)
@pytest.mark.parametrize("test_duration_seconds", [24 * 60 * 60, 10 * 60], ids=["24h", "10m"])
def test_sd35_stability(
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
    test_duration_seconds,
    model_location_generator,
    is_ci_env,
    galaxy_type,
) -> None:
    """Performance test for SD35 pipeline with detailed timing analysis."""

    if galaxy_type == "4U":
        pytest.skip("4U is not supported for this test")

    cfg_factor = cfg[0]

    parallel_manager, encoder_parallel_manager, vae_parallel_manager, enable_t5_text_encoder = create_parallel_configs(
        mesh_device, cfg, sp, tp, topology, num_links
    )

    if guidance_scale > 1 and cfg_factor == 1:
        guidance_cond = 2
    else:
        guidance_cond = 1
    # Create pipeline
    pipeline = TtStableDiffusion3Pipeline(
        checkpoint_name=f"stabilityai/stable-diffusion-3.5-{model_name}",
        mesh_device=mesh_device,
        enable_t5_text_encoder=enable_t5_text_encoder,
        guidance_cond=guidance_cond,
        parallel_manager=parallel_manager,
        encoder_parallel_manager=encoder_parallel_manager,
        vae_parallel_manager=vae_parallel_manager,
        height=image_h,
        width=image_w,
        model_location_generator=model_location_generator,
        quiet=True,
    )

    pipeline.prepare(
        batch_size=1,
        width=image_w,
        height=image_h,
        guidance_scale=guidance_scale,
        prompt_sequence_length=333,
        spatial_sequence_length=4096,
    )

    prompt = """A neon-lit alley in a sprawling cyberpunk metropols at night, rain-slick streets reflecting glowing holograms, dense atmosphere, flying cars in the sky, people in high-tech streetwear — ultra-detailed, cinematic lighting, 4K"""
    negative_prompt = ""

    start = time.time()
    iter = 0
    num_logs = 100
    log_interval = test_duration_seconds / num_logs
    last_log_time = start

    while True:
        # Run pipeline
        images = pipeline(
            prompt_1=[prompt],
            prompt_2=[prompt],
            prompt_3=[prompt],
            negative_prompt_1=[negative_prompt],
            negative_prompt_2=[negative_prompt],
            negative_prompt_3=[negative_prompt],
            num_inference_steps=num_inference_steps,
            seed=0,
            traced=True,
        )

        if time.time() - start > test_duration_seconds:
            break

        if time.time() - last_log_time > log_interval:
            logger.info(f"At {time.time() - start} seconds, iteration {iter} completed")
            last_log_time = time.time()

        iter += 1

    logger.info(f"Test completed after {iter} iterations and {time.time() - start} seconds")

    for submesh_device in parallel_manager.submesh_devices:
        ttnn.synchronize_device(submesh_device)
