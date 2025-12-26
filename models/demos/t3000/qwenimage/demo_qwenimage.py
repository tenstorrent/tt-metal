# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
QwenImage Demo for T3K (2x4 mesh configuration)

This demo runs the QwenImage image generation pipeline on a T3K (8 device) mesh configuration.
It generates high-quality images from text prompts using the Qwen-Image diffusion model.

Usage:
    pytest models/demos/t3000/qwenimage/demo_qwenimage.py -k "demo"
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
        [(2, 4), (1, 0), (2, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 1],
    ],
    ids=["t3k_2x4sp2tp4"],
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
def test_qwenimage_demo_t3k(
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
) -> None:
    """
    Demo test for QwenImage on T3K (2x4 mesh, 8 devices).

    This test demonstrates the full QwenImage pipeline including:
    - Text encoding (CLIP and T5)
    - Denoising with the DiT transformer
    - VAE decoding to generate final images
    """
    logger.info("Starting QwenImage demo on T3K (2x4 mesh)")
    logger.info(f"Configuration: width={width}, height={height}, steps={num_inference_steps}")
    logger.info(f"Mesh shape: {mesh_device.shape}")

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
    mesh_test_id = f"t3k_{mesh_device.shape[0]}x{mesh_device.shape[1]}_sp{sp_factor}tp{tp_factor}"

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

    logger.info("QwenImage T3K demo completed successfully!")


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 38000000}],
    indirect=True,
)
@pytest.mark.parametrize(("width", "height", "num_inference_steps"), [(1024, 1024, 50)])
@pytest.mark.parametrize(
    "mesh_device, cfg, sp, tp, encoder_tp, vae_tp, topology, num_links",
    [
        [(2, 4), (1, 0), (2, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 1],
    ],
    ids=["t3k_2x4sp2tp4"],
    indirect=["mesh_device"],
)
def test_qwenimage_demo_t3k_quick(
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
) -> None:
    """
    Quick demo test for QwenImage on T3K with reduced inference steps.

    This is a faster version of the demo for quick verification and CI testing.
    """
    logger.info("Starting quick QwenImage demo on T3K (2x4 mesh)")

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

    output_filename = f"qwenimage_t3k_quick_{width}x{height}.png"
    images[0].save(output_filename)
    logger.info(f"Image saved as {output_filename}")

    # Log timing
    timing_data = pipeline.timing_collector.get_timing_data()
    logger.info(f"Total pipeline time: {timing_data.total_time:.2f}s")

    ttnn.synchronize_device(mesh_device)
    logger.info("QwenImage T3K quick demo completed!")
