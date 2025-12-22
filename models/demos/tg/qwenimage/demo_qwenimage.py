# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
QwenImage Demo for Galaxy (4x8 mesh configuration)

This demo runs the QwenImage image generation pipeline on a Galaxy (32 device) mesh configuration.
It generates high-quality images from text prompts using the Qwen-Image diffusion model.

Usage:
    pytest models/demos/tg/qwenimage/demo_qwenimage.py -k "demo"
"""

import pytest
from loguru import logger

import ttnn
from models.experimental.tt_dit.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from models.experimental.tt_dit.pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import (
    TimingCollector,
)

# Default prompts for demo
DEFAULT_PROMPTS = [
    'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light '
    'beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the '
    'poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition',
    'Tokyo neon alley at night, rain-slick pavement, cinematic cyberpunk lighting; include glowing sign text "深夜営業" in bold neon above a doorway; moody reflections, shallow depth of field.',
    'Steamy ramen shop entrance at dusk; fabric noren curtain gently swaying; print "しょうゆラーメン" across the curtain in thick brush-style kana; warm lantern light, photorealistic.',
    'Minimalist tea poster, cream background, elegant layout; vertical calligraphy "抹茶" centered in sumi ink; small red hanko-style seal "本格" in the corner; high-resolution graphic design.',
    'Hardcover fantasy novel cover, textured paper, gold foil; title text "物語のはじまり" centered; author line "山本ひかり" below; tasteful serif typography, dramatic vignette illustration.',
]


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 38000000}],
    indirect=True,
)
@pytest.mark.parametrize(("width", "height", "num_inference_steps"), [(1024, 1024, 50)])
@pytest.mark.parametrize(
    "mesh_device, cfg, sp, tp, encoder_tp, vae_tp, topology, num_links",
    [
        [(4, 8), (2, 1), (4, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 4],
    ],
    ids=["galaxy_4x8cfg2sp4tp4"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "use_torch_text_encoder",
    [
        pytest.param(False, id="encoder_device"),
    ],
)
@pytest.mark.parametrize(
    "traced",
    [
        pytest.param(True, id="traced"),
    ],
)
def test_qwenimage_demo_galaxy(
    *,
    mesh_device: ttnn.MeshDevice,
    width: int,
    height: int,
    num_inference_steps: int,
    cfg: tuple[int, int],
    sp: tuple[int, int],
    tp: tuple[int, int],
    encoder_tp: tuple[int, int],
    vae_tp: tuple[int, int],
    topology: ttnn.Topology,
    num_links: int,
    use_torch_text_encoder: bool,
    traced: bool,
    is_ci_env: bool,
    galaxy_type: str,
) -> None:
    """
    Demo test for QwenImage on Galaxy (4x8 mesh, 32 devices).

    This test demonstrates the full QwenImage pipeline on Galaxy including:
    - Text encoding (CLIP and T5)
    - Denoising with the DiT transformer
    - VAE decoding to generate final images
    """
    logger.info(f"Starting QwenImage demo on Galaxy ({galaxy_type}, 4x8 mesh)")
    logger.info(f"Configuration: width={width}, height={height}, steps={num_inference_steps}")
    logger.info(f"Mesh shape: {mesh_device.shape}")

    # Skip on unsupported Galaxy configurations if needed
    if galaxy_type not in ["4U", "6U"]:
        pytest.skip(f"Unsupported galaxy type: {galaxy_type}")

    # Create the pipeline
    pipeline = QwenImagePipeline.create_pipeline(
        mesh_device=mesh_device,
        dit_cfg=cfg,
        dit_sp=sp,
        dit_tp=tp,
        encoder_tp=encoder_tp,
        vae_tp=vae_tp,
        use_torch_text_encoder=use_torch_text_encoder,
        use_torch_vae_decoder=False,
        num_links=num_links,
        topology=topology,
        width=width,
        height=height,
        is_fsdp=True,  # Enable FSDP to avoid model load/unload cycle
    )
    pipeline.timing_collector = TimingCollector()

    # Select prompts based on environment
    if is_ci_env:
        # In CI, run only one prompt to save time
        prompts_to_run = DEFAULT_PROMPTS[:1]
    else:
        prompts_to_run = DEFAULT_PROMPTS

    cfg_factor, cfg_axis = cfg
    sp_factor, sp_axis = sp
    tp_factor, tp_axis = tp
    mesh_test_id = f"galaxy_{mesh_device.shape[0]}x{mesh_device.shape[1]}_cfg{cfg_factor}sp{sp_factor}tp{tp_factor}"

    for i, prompt in enumerate(prompts_to_run):
        logger.info(f"Generating image {i+1}/{len(prompts_to_run)}")
        logger.info(f"Prompt: {prompt[:100]}...")

        images = pipeline(
            prompts=[prompt],
            negative_prompts=[None],
            num_inference_steps=num_inference_steps,
            cfg_scale=4.0,
            seed=i,
            traced=traced,
        )

        output_filename = f"qwenimage_{mesh_test_id}_{width}x{height}_{i}.png"
        images[0].save(output_filename)
        logger.info(f"Image saved as {output_filename}")

        # Log timing information
        timing_data = pipeline.timing_collector.get_timing_data()
        logger.info(f"CLIP encoding time: {timing_data.clip_encoding_time:.2f}s")
        logger.info(f"T5 encoding time: {timing_data.t5_encoding_time:.2f}s")
        logger.info(f"Total encoding time: {timing_data.total_encoding_time:.2f}s")
        logger.info(f"VAE decoding time: {timing_data.vae_decoding_time:.2f}s")
        logger.info(f"Total pipeline time: {timing_data.total_time:.2f}s")
        if timing_data.denoising_step_times:
            avg_step_time = sum(timing_data.denoising_step_times) / len(timing_data.denoising_step_times)
            logger.info(f"Average denoising step time: {avg_step_time:.2f}s")

    # Synchronize all devices
    ttnn.synchronize_device(mesh_device)

    logger.info("QwenImage Galaxy demo completed successfully!")


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 38000000}],
    indirect=True,
)
@pytest.mark.parametrize(("width", "height", "num_inference_steps"), [(1024, 1024, 50)])
@pytest.mark.parametrize(
    "mesh_device, cfg, sp, tp, encoder_tp, vae_tp, topology, num_links",
    [
        [(4, 8), (2, 1), (4, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 4],
    ],
    ids=["galaxy_4x8cfg2sp4tp4"],
    indirect=["mesh_device"],
)
def test_qwenimage_demo_galaxy_quick(
    *,
    mesh_device: ttnn.MeshDevice,
    width: int,
    height: int,
    num_inference_steps: int,
    cfg: tuple[int, int],
    sp: tuple[int, int],
    tp: tuple[int, int],
    encoder_tp: tuple[int, int],
    vae_tp: tuple[int, int],
    topology: ttnn.Topology,
    num_links: int,
    is_ci_env: bool,
    galaxy_type: str,
) -> None:
    """
    Quick demo test for QwenImage on Galaxy with reduced inference steps.

    This is a faster version of the demo for quick verification and CI testing.
    """
    logger.info(f"Starting quick QwenImage demo on Galaxy ({galaxy_type}, 4x8 mesh)")

    # Skip on unsupported Galaxy configurations if needed
    if galaxy_type not in ["4U", "6U"]:
        pytest.skip(f"Unsupported galaxy type: {galaxy_type}")

    # Create the pipeline
    pipeline = QwenImagePipeline.create_pipeline(
        mesh_device=mesh_device,
        dit_cfg=cfg,
        dit_sp=sp,
        dit_tp=tp,
        encoder_tp=encoder_tp,
        vae_tp=vae_tp,
        use_torch_text_encoder=False,
        use_torch_vae_decoder=False,
        num_links=num_links,
        topology=topology,
        width=width,
        height=height,
        is_fsdp=True,
    )
    pipeline.timing_collector = TimingCollector()

    # Use a simple prompt for quick testing
    prompt = DEFAULT_PROMPTS[0]

    logger.info(f"Generating image with {num_inference_steps} steps...")

    images = pipeline(
        prompts=[prompt],
        negative_prompts=[None],
        num_inference_steps=num_inference_steps,
        cfg_scale=4.0,
        seed=0,
        traced=True,
    )

    output_filename = f"qwenimage_galaxy_quick_{width}x{height}.png"
    images[0].save(output_filename)
    logger.info(f"Image saved as {output_filename}")

    # Log timing
    timing_data = pipeline.timing_collector.get_timing_data()
    logger.info(f"Total pipeline time: {timing_data.total_time:.2f}s")

    ttnn.synchronize_device(mesh_device)
    logger.info("QwenImage Galaxy quick demo completed!")


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 40000000}],
    indirect=True,
)
@pytest.mark.parametrize(("width", "height", "num_inference_steps"), [(1024, 1024, 50)])
@pytest.mark.parametrize(
    "mesh_device, cfg, sp, tp, encoder_tp, vae_tp, topology, num_links",
    [
        [(4, 8), (2, 1), (4, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 4],
    ],
    ids=["galaxy_4x8cfg2sp4tp4"],
    indirect=["mesh_device"],
)
def test_qwenimage_demo_galaxy_full(
    *,
    mesh_device: ttnn.MeshDevice,
    width: int,
    height: int,
    num_inference_steps: int,
    cfg: tuple[int, int],
    sp: tuple[int, int],
    tp: tuple[int, int],
    encoder_tp: tuple[int, int],
    vae_tp: tuple[int, int],
    topology: ttnn.Topology,
    num_links: int,
    is_ci_env: bool,
    galaxy_type: str,
) -> None:
    """
    Full demo test for QwenImage on Galaxy running all default prompts.

    This demo runs a comprehensive set of prompts to showcase the model's capabilities.
    """
    logger.info(f"Starting full QwenImage demo on Galaxy ({galaxy_type}, 4x8 mesh)")

    # Skip on unsupported Galaxy configurations or in CI to save time
    if galaxy_type not in ["4U", "6U"]:
        pytest.skip(f"Unsupported galaxy type: {galaxy_type}")

    if is_ci_env:
        pytest.skip("Skipping full demo in CI environment to save time")

    # Create the pipeline
    pipeline = QwenImagePipeline.create_pipeline(
        mesh_device=mesh_device,
        dit_cfg=cfg,
        dit_sp=sp,
        dit_tp=tp,
        encoder_tp=encoder_tp,
        vae_tp=vae_tp,
        use_torch_text_encoder=False,
        use_torch_vae_decoder=False,
        num_links=num_links,
        topology=topology,
        width=width,
        height=height,
        is_fsdp=True,
    )
    pipeline.timing_collector = TimingCollector()

    logger.info(f"Running {len(DEFAULT_PROMPTS)} prompts...")

    total_time = 0.0
    for i, prompt in enumerate(DEFAULT_PROMPTS):
        logger.info(f"Generating image {i+1}/{len(DEFAULT_PROMPTS)}")
        logger.info(f"Prompt: {prompt[:80]}...")

        images = pipeline(
            prompts=[prompt],
            negative_prompts=[None],
            num_inference_steps=num_inference_steps,
            cfg_scale=4.0,
            seed=i,
            traced=True,
        )

        output_filename = f"qwenimage_galaxy_full_{width}x{height}_{i}.png"
        images[0].save(output_filename)

        timing_data = pipeline.timing_collector.get_timing_data()
        total_time += timing_data.total_time
        logger.info(f"Image {i+1} completed in {timing_data.total_time:.2f}s")

    avg_time = total_time / len(DEFAULT_PROMPTS)
    logger.info(f"Full demo completed! Average time per image: {avg_time:.2f}s")

    ttnn.synchronize_device(mesh_device)
    logger.info("QwenImage Galaxy full demo completed!")
