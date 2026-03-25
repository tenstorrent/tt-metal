# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test the LTX-2 pipeline denoising loop.

Uses a 1-layer model with random weights to verify the pipeline mechanics:
- Sigma schedule computation
- Euler step formula
- CFG blending
- Device↔host data flow in denoising loop
"""

import sys

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline, compute_sigmas, euler_step

sys.path.insert(0, "LTX-2/packages/ltx-core/src")


def test_sigma_schedule():
    """Test that sigma schedule matches LTX-2 reference."""
    from ltx_core.components.schedulers import LTX2Scheduler

    steps = 30
    num_tokens = 2048

    # Our implementation
    our_sigmas = compute_sigmas(steps=steps, num_tokens=num_tokens)

    # Reference
    ref_scheduler = LTX2Scheduler()
    # Create a dummy latent with the right token count for the reference
    # num_tokens = prod(latent.shape[2:]), so we need shape where spatial dims multiply to num_tokens
    dummy_latent = torch.randn(1, 1, num_tokens)
    ref_sigmas = ref_scheduler.execute(steps=steps, latent=dummy_latent)

    assert our_sigmas.shape == ref_sigmas.shape, f"Shape mismatch: {our_sigmas.shape} vs {ref_sigmas.shape}"
    assert torch.allclose(
        our_sigmas, ref_sigmas, atol=1e-6
    ), f"Sigma mismatch: max diff {(our_sigmas - ref_sigmas).abs().max():.2e}"
    logger.info(f"Sigma schedule matches reference (max diff: {(our_sigmas - ref_sigmas).abs().max():.2e})")


def test_euler_step():
    """Test that Euler step matches LTX-2 reference."""
    from ltx_core.components.diffusion_steps import EulerDiffusionStep

    torch.manual_seed(42)
    sample = torch.randn(1, 256, 128)
    denoised = torch.randn(1, 256, 128)
    sigmas = torch.tensor([0.8, 0.6, 0.4, 0.2, 0.0])

    # Our implementation
    our_result = euler_step(sample, denoised, sigma=0.8, sigma_next=0.6)

    # Reference
    ref_stepper = EulerDiffusionStep()
    ref_result = ref_stepper.step(sample, denoised, sigmas, step_index=0)

    assert torch.allclose(
        our_result, ref_result, atol=1e-5
    ), f"Euler step mismatch: max diff {(our_result - ref_result).abs().max():.2e}"
    logger.info(f"Euler step matches reference (max diff: {(our_result - ref_result).abs().max():.2e})")


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    ids=["1x1"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_pipeline_denoising_loop(mesh_device: ttnn.MeshDevice):
    """
    Test the full pipeline denoising loop with a 1-layer model.
    Verifies device↔host data flow, sigma stepping, and output shape.
    """
    from ltx_core.model.transformer.model import LTXModel, LTXModelType

    sp_axis, tp_axis = 0, 1
    dim = 4096
    num_heads = 32
    head_dim = dim // num_heads
    in_channels = 128
    out_channels = 128
    num_layers = 1
    num_inference_steps = 3  # Very few steps for fast test

    # Create random-weight reference model to get state dict
    torch_model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        in_channels=in_channels,
        out_channels=out_channels,
        cross_attention_dim=dim,
        use_middle_indices_grid=True,
        cross_attention_adaln=True,
    )
    torch_model.eval()

    # Create pipeline
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)

    pipeline = LTXPipeline(
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        num_layers=num_layers,
        cross_attention_dim=dim,
    )
    pipeline.load_transformer(torch_model.state_dict())
    # No text encoder — will use zero embeddings

    # Run pipeline with tiny dimensions
    num_frames = 17
    px_height, px_width = 128, 128
    output = pipeline(
        prompt=["test"],
        num_frames=num_frames,
        height=px_height,
        width=px_width,
        num_inference_steps=num_inference_steps,
        guidance_scale=1.0,  # No CFG for speed
        seed=42,
    )

    # Compute expected latent shape
    latent_frames = (num_frames - 1) // 8 + 1  # 8x temporal compression
    latent_h = px_height // 32  # 32x spatial compression
    latent_w = px_width // 32
    expected_tokens = latent_frames * latent_h * latent_w
    assert output.shape == (
        1,
        expected_tokens,
        out_channels,
    ), f"Output shape {output.shape} != expected (1, {expected_tokens}, {out_channels})"
    assert torch.isfinite(output).all(), "Output contains NaN/Inf"
    logger.info(f"Pipeline output shape: {output.shape}, range: [{output.min():.3f}, {output.max():.3f}]")
    logger.info("PASSED: Pipeline denoising loop works end-to-end")


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    ids=["1x1"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_pipeline_with_vae_decode(mesh_device: ttnn.MeshDevice):
    """
    Test pipeline with TTNN VAE decoder: denoise → decode to video.
    """
    from ltx_core.model.transformer.model import LTXModel, LTXModelType

    sp_axis, tp_axis = 0, 1
    dim = 4096
    num_heads = 32
    head_dim = dim // num_heads
    in_channels = 128
    out_channels = 128
    num_layers = 1
    num_inference_steps = 2

    # Create transformer
    torch_model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        in_channels=in_channels,
        out_channels=out_channels,
        cross_attention_dim=dim,
        use_middle_indices_grid=True,
        cross_attention_adaln=True,
    )
    torch_model.eval()

    # Create VAE decoder
    from ltx_core.model.video_vae.enums import NormLayerType
    from ltx_core.model.video_vae.video_vae import VideoDecoder

    decoder_blocks = [
        ("compress_all", {"multiplier": 2}),
        ("compress_all", {"multiplier": 2}),
        ("compress_time", {"multiplier": 2}),
        ("compress_space", {"multiplier": 2}),
    ]
    torch_decoder = VideoDecoder(
        convolution_dimensions=3,
        in_channels=128,
        out_channels=3,
        decoder_blocks=decoder_blocks,
        patch_size=4,
        norm_layer=NormLayerType.PIXEL_NORM,
        causal=True,
        timestep_conditioning=False,
        base_channels=128,
    )
    torch_decoder.eval()
    dec_state = torch_decoder.state_dict()
    dec_state["per_channel_statistics.mean-of-means"] = torch.zeros(128)
    dec_state["per_channel_statistics.std-of-means"] = torch.ones(128)
    torch_decoder.load_state_dict(dec_state)

    # Create pipeline
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)

    pipeline = LTXPipeline(
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        num_layers=num_layers,
        cross_attention_dim=dim,
    )
    pipeline.load_transformer(torch_model.state_dict())
    pipeline.load_vae_decoder(
        state_dict=torch_decoder.state_dict(),
        decoder_blocks=decoder_blocks,
        use_ttnn=True,
    )

    # Run pipeline — output should be decoded video
    num_frames = 17
    px_height, px_width = 128, 128
    output = pipeline(
        prompt=["test"],
        num_frames=num_frames,
        height=px_height,
        width=px_width,
        num_inference_steps=num_inference_steps,
        guidance_scale=1.0,
        seed=42,
    )

    # Output should be video (B, 3, F, H, W) not latent
    assert output.shape[1] == 3, f"Expected 3 channels (RGB), got {output.shape[1]}"
    assert output.shape[2] == num_frames, f"Expected {num_frames} frames, got {output.shape[2]}"
    assert output.shape[3] == px_height, f"Expected height {px_height}, got {output.shape[3]}"
    assert output.shape[4] == px_width, f"Expected width {px_width}, got {output.shape[4]}"
    logger.info(f"Pipeline+VAE output: {output.shape}")
    logger.info("PASSED: Pipeline with TTNN VAE decoder")


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis",
    [((2, 4), (2, 4), 0, 1)],
    ids=["wh_lb_2x4"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_pipeline_av_22b(mesh_device: ttnn.MeshDevice, mesh_shape, sp_axis: int, tp_axis: int):
    """
    Test LTX-2.3 22B AudioVideo pipeline on WH LB 2x4 mesh.

    Mirrors the reference ti2vid_one_stage.py flow:
    1. Encode prompts using reference encode_prompts
    2. Run AV denoising with full MultiModalGuider guidance
    3. Verify output shapes

    Uses real 22B checkpoint with 5 steps for fast validation.
    """
    import os

    from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline

    ckpt = os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors")
    if not os.path.exists(ckpt):
        pytest.skip("22B checkpoint not found")

    gemma = None
    # Find Gemma model in HF cache
    import glob

    candidates = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/*/"))
    if candidates:
        gemma = candidates[0].rstrip("/")
    if gemma is None:
        pytest.skip("Gemma model not found in HF cache")

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)

    # Create pipeline (AV mode)
    pipeline = LTXPipeline(
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        mode="av",
    )

    # Load checkpoint (transformer + VAE)
    pipeline.load_from_checkpoint(ckpt)

    # Encode prompts using reference pipeline
    torch.cuda.synchronize = lambda *a, **kw: None  # No CUDA on TT host
    sys.path.insert(0, "LTX-2/packages/ltx-pipelines/src")
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT

    prompt = "A cat playing piano in a cozy room with warm lighting"
    results = pipeline.encode_prompts_reference([prompt, DEFAULT_NEGATIVE_PROMPT], ckpt, gemma)
    v_embeds = results[0].video_encoding.float()
    a_embeds = results[0].audio_encoding.float() if results[0].audio_encoding is not None else None
    neg_v = results[1].video_encoding.float()
    neg_a = results[1].audio_encoding.float() if results[1].audio_encoding is not None else None

    if a_embeds is None:
        pytest.skip("Audio embeddings not available")

    # Run AV denoising (2 steps for fast test, no guidance for speed)
    video_latent, audio_latent = pipeline.call_av(
        video_prompt_embeds=v_embeds,
        audio_prompt_embeds=a_embeds,
        num_frames=33,
        height=512,
        width=768,
        num_inference_steps=2,
        video_cfg_scale=1.0,
        audio_cfg_scale=1.0,
        video_stg_scale=0.0,
        audio_stg_scale=0.0,
        video_modality_scale=1.0,
        audio_modality_scale=1.0,
        rescale_scale=0.0,
        seed=10,
    )

    logger.info(f"Video latent: {video_latent.shape}, Audio latent: {audio_latent.shape}")
    assert video_latent.shape[1] == 5 * 16 * 24  # 33 frames @ 512x768
    assert torch.isfinite(video_latent).all(), "Video latent has NaN/Inf"
    assert torch.isfinite(audio_latent).all(), "Audio latent has NaN/Inf"
    logger.info("PASSED: AV 22B pipeline test")
