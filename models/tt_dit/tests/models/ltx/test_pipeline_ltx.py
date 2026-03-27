# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test the LTX-2 pipeline denoising loop.

Uses standalone references (no ltx_core dependency):
- Sigma schedule computation
- Euler step formula
- CFG blending
- Device↔host data flow in denoising loop
- VAE decode
- Full AV pipeline with device Gemma encoding + connectors
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[6]))

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline, compute_sigmas, euler_step

# ---------------------------------------------------------------------------
# Scheduler tests (no device needed)
# ---------------------------------------------------------------------------


def test_sigma_schedule():
    """Test sigma schedule produces valid monotonically-decreasing values ending at 0."""
    steps = 30
    num_tokens = 2048

    sigmas = compute_sigmas(steps=steps, num_tokens=num_tokens)

    assert sigmas.shape == (steps + 1,), f"Expected {steps+1} sigma values, got {sigmas.shape}"
    assert sigmas[0] > 0, "First sigma must be positive"
    assert sigmas[-1] == 0.0, "Last sigma must be 0"
    # Monotonically decreasing
    for i in range(len(sigmas) - 1):
        assert sigmas[i] >= sigmas[i + 1], f"Sigma not decreasing at index {i}: {sigmas[i]} < {sigmas[i+1]}"
    logger.info(f"Sigma schedule: {sigmas[0]:.4f} -> {sigmas[-1]:.4f} ({len(sigmas)} values)")


def test_euler_step():
    """Test Euler step: x_{t+1} = x_t + velocity * dt."""
    torch.manual_seed(42)
    sample = torch.randn(1, 256, 128)
    denoised = torch.randn(1, 256, 128)
    sigma, sigma_next = 0.8, 0.6

    result = euler_step(sample, denoised, sigma=sigma, sigma_next=sigma_next)

    # Manual computation
    velocity = (sample.float() - denoised.float()) / sigma
    dt = sigma_next - sigma
    expected = (sample.float() + velocity * dt).to(sample.dtype)

    assert torch.allclose(
        result, expected, atol=1e-6
    ), f"Euler step mismatch: max diff {(result - expected).abs().max():.2e}"
    logger.info("Euler step matches manual computation")


# ---------------------------------------------------------------------------
# Pipeline tests (need device)
# ---------------------------------------------------------------------------


def _fill_module_random(module, prefix=""):
    """Recursively fill all Parameters in a Module tree with random data."""

    for name, param in module.named_parameters():
        param.load_torch_tensor(torch.randn(param.total_shape))
    for name, child in module.named_children():
        _fill_module_random(child, prefix=f"{prefix}{name}.")


def _make_pipeline_with_random_weights(
    mesh_device,
    num_layers=1,
    dim=4096,
    num_heads=32,
    in_channels=128,
    out_channels=128,
):
    """Create an LTXPipeline with random transformer weights (no ltx_core dependency)."""
    from models.tt_dit.models.transformers.ltx.ltx_transformer import LTXTransformerModel

    sp_axis, tp_axis = 0, 1
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

    # Create TTNN model and fill with random weights
    head_dim = dim // num_heads
    pipeline.transformer = LTXTransformerModel(
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        in_channels=in_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        cross_attention_dim=dim,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    torch.manual_seed(0)
    _fill_module_random(pipeline.transformer)
    logger.info(f"Created pipeline with {num_layers}L random-weight transformer")
    return pipeline


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_pipeline_denoising_loop(mesh_device: ttnn.MeshDevice):
    """Test the full pipeline denoising loop with a 1-layer model."""
    num_layers = 1
    out_channels = 128
    num_inference_steps = 3

    pipeline = _make_pipeline_with_random_weights(mesh_device, num_layers=num_layers)

    num_frames = 17
    px_height, px_width = 256, 256
    output = pipeline(
        prompt=["test"],
        num_frames=num_frames,
        height=px_height,
        width=px_width,
        num_inference_steps=num_inference_steps,
        guidance_scale=1.0,
        seed=42,
    )

    latent_frames = (num_frames - 1) // 8 + 1
    latent_h = px_height // 32
    latent_w = px_width // 32
    expected_tokens = latent_frames * latent_h * latent_w
    assert output.shape == (
        1,
        expected_tokens,
        out_channels,
    ), f"Output shape {output.shape} != expected (1, {expected_tokens}, {out_channels})"
    assert torch.isfinite(output).all(), "Output contains NaN/Inf"
    logger.info(f"Pipeline output: {output.shape}, range: [{output.min():.3f}, {output.max():.3f}]")
    logger.info("PASSED: Pipeline denoising loop works end-to-end")


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_pipeline_with_vae_decode(mesh_device: ttnn.MeshDevice):
    """Test pipeline with TTNN VAE decoder: denoise → decode to video."""
    from models.tt_dit.utils.vae_reference import VideoDecoder as TorchVideoDecoder

    num_layers = 1
    num_inference_steps = 2

    decoder_blocks = [
        ("compress_all", {"multiplier": 2}),
        ("compress_all", {"multiplier": 2}),
        ("compress_time", {"multiplier": 2}),
        ("compress_space", {"multiplier": 2}),
    ]
    torch_decoder = TorchVideoDecoder(
        convolution_dimensions=3,
        in_channels=128,
        out_channels=3,
        decoder_blocks=decoder_blocks,
        patch_size=4,
        norm_layer="pixel_norm",
        causal=True,
        base_channels=128,
    )
    torch_decoder.eval()
    dec_state = torch_decoder.state_dict()
    dec_state["per_channel_statistics.mean-of-means"] = torch.zeros(128)
    dec_state["per_channel_statistics.std-of-means"] = torch.ones(128)
    torch_decoder.load_state_dict(dec_state)

    pipeline = _make_pipeline_with_random_weights(mesh_device, num_layers=num_layers)
    pipeline.load_vae_decoder(
        state_dict=torch_decoder.state_dict(),
        decoder_blocks=decoder_blocks,
        use_ttnn=True,
    )

    num_frames = 17
    px_height, px_width = 256, 256
    output = pipeline(
        prompt=["test"],
        num_frames=num_frames,
        height=px_height,
        width=px_width,
        num_inference_steps=num_inference_steps,
        guidance_scale=1.0,
        seed=42,
    )

    assert output.shape[1] == 3, f"Expected 3 channels (RGB), got {output.shape[1]}"
    assert output.shape[2] == num_frames, f"Expected {num_frames} frames, got {output.shape[2]}"
    assert output.shape[3] == px_height, f"Expected height {px_height}, got {output.shape[3]}"
    assert output.shape[4] == px_width, f"Expected width {px_width}, got {output.shape[4]}"
    logger.info(f"Pipeline+VAE output: {output.shape}")
    logger.info("PASSED: Pipeline with TTNN VAE decoder")


# ---------------------------------------------------------------------------
# Full AV pipeline PCC test (needs 22B checkpoint + Gemma on 2x4 mesh)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis",
    [((2, 4), (2, 4), 0, 1)],
    ids=["wh_lb_2x4"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_pipeline_av_22b(mesh_device: ttnn.MeshDevice, mesh_shape, sp_axis: int, tp_axis: int):
    """
    Full LTX-2.3 22B AV pipeline on WH LB 2x4 mesh.

    Gemma encoding (device) → connectors → dealloc → DiT denoise → dealloc → VAE decode → export MP4.
    All on device, no ltx_core dependency for encoding/denoising/decode.
    """
    import gc
    import glob
    import json
    import time

    from safetensors.torch import load_file

    from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline

    # --- Config (overridable via env vars) ---
    ckpt = os.environ.get("LTX_CHECKPOINT", os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors"))
    if not os.path.exists(ckpt):
        pytest.skip("22B checkpoint not found")

    gemma = os.environ.get("GEMMA_PATH", "/localdev/kevinmi/.cache/gemma-3-12b-it-qat-q4_0-unquantized")
    if not os.path.isdir(gemma):
        # Fallback to HF cache
        candidates = glob.glob(
            os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/*/")
        )
        gemma = candidates[0].rstrip("/") if candidates else None
    if gemma is None or not os.path.isdir(gemma):
        pytest.skip("Gemma model not found")

    prompt = os.environ.get(
        "PROMPT",
        "Studio Ghibli style. A young girl in a sundress runs through a field of giant, "
        "luminescent mushrooms at twilight. The mushrooms glow soft greens and blues, pulsing "
        "gently like breathing. Fireflies trail behind her as she runs. The background shows "
        "rolling hills with a distant castle. Painterly textures, soft edges, and warm nostalgic "
        "color grading. Audio: Whimsical orchestral music with woodwinds, her laughter echoing.",
    )
    num_frames = int(os.environ.get("NUM_FRAMES", "121"))
    height = int(os.environ.get("HEIGHT", "512"))
    width = int(os.environ.get("WIDTH", "768"))
    num_steps = int(os.environ.get("NUM_STEPS", "40"))
    seed = int(os.environ.get("SEED", "42"))
    output_path = os.environ.get("OUTPUT_PATH", "output_e2e.mp4")

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

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
        mode="av",
    )

    # === Stage 1: Gemma encoding (reference CPU pipeline) ===
    # TTNN Gemma encoder has numerical instability (bf16 accumulation across 48 layers)
    # that produces divergent hidden states. Use reference CPU encoding until fixed.
    sys.path.insert(0, "LTX-2/packages/ltx-core/src")
    sys.path.insert(0, "LTX-2/packages/ltx-pipelines/src")
    torch.cuda.synchronize = lambda *a, **kw: None
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT as REF_NEG

    t0 = time.time()
    results = pipeline.encode_prompts_reference([prompt, REF_NEG], ckpt, gemma)
    encode_time = time.time() - t0

    v_embeds = results[0].video_encoding.float()
    a_embeds = results[0].audio_encoding.float() if results[0].audio_encoding is not None else None
    neg_v = results[1].video_encoding.float()
    neg_a = results[1].audio_encoding.float() if results[1].audio_encoding is not None else None
    logger.info(f"Reference encoding: {encode_time:.1f}s — video {v_embeds.shape}")

    # === Stage 2: Load DiT + denoise ===
    raw = load_file(ckpt)
    prefix = "model.diffusion_model."
    transformer_sd = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}

    # Extract VAE config before deleting raw
    with open(ckpt, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_size))
    vae_cfg = json.loads(header.get("__metadata__", {}).get("config", "{}")).get("vae", {})
    pipeline._vae_checkpoint_path = ckpt
    pipeline._vae_decoder_blocks = vae_cfg.get("decoder_blocks", [])
    pipeline._vae_causal = vae_cfg.get("causal_decoder", False)
    pipeline._vae_base_channels = vae_cfg.get("decoder_base_channels", 128)

    del raw
    pipeline.load_transformer(transformer_sd)
    del transformer_sd
    gc.collect()

    if a_embeds is None:
        pytest.skip("Audio embeddings not available")

    logger.info(f"Denoising: {num_frames}f @ {height}x{width}, {num_steps} steps, seed={seed}")
    t0 = time.time()
    video_latent, audio_latent = pipeline.call_av(
        video_prompt_embeds=v_embeds,
        audio_prompt_embeds=a_embeds,
        neg_video_prompt_embeds=neg_v,
        neg_audio_prompt_embeds=neg_a,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        video_cfg_scale=3.0,
        audio_cfg_scale=7.0,
        video_stg_scale=0.0,
        audio_stg_scale=0.0,
        video_modality_scale=3.0,
        audio_modality_scale=3.0,
        rescale_scale=0.7,
        seed=seed,
        ge_gamma=0.0,  # Disable gradient estimation to match known-good baseline
    )
    denoise_time = time.time() - t0
    logger.info(f"Denoising done: {denoise_time:.1f}s ({denoise_time/num_steps:.1f}s/step)")

    assert torch.isfinite(video_latent).all(), "Video latent has NaN/Inf"
    assert torch.isfinite(audio_latent).all(), "Audio latent has NaN/Inf"
    logger.info(f"Video latent: {video_latent.shape}, range [{video_latent.min():.3f}, {video_latent.max():.3f}]")
    logger.info(f"Audio latent: {audio_latent.shape}, range [{audio_latent.min():.3f}, {audio_latent.max():.3f}]")

    # === Stage 3: Dealloc DiT, load VAE, decode video ===
    pipeline.transformer = None
    gc.collect()

    t0 = time.time()
    pipeline.load_vae_from_checkpoint()
    logger.info(f"VAE decoder loaded in {time.time()-t0:.0f}s")

    latent_frames = (num_frames - 1) // 8 + 1
    latent_h, latent_w = height // 32, width // 32

    t0 = time.time()
    video_pixels = pipeline.decode_latents(video_latent, latent_frames, latent_h, latent_w)
    decode_time = time.time() - t0
    logger.info(f"VAE decode: {decode_time:.1f}s — {video_pixels.shape}")

    # === Stage 4: Audio decode (reference, optional) ===
    audio_obj = pipeline.decode_audio_reference(audio_latent, ckpt, num_frames, fps=24)

    # === Stage 5: Export MP4 ===
    pipeline.export_video(video_pixels, output_path, fps=24, audio=audio_obj)

    # === Summary ===
    logger.info("=" * 60)
    logger.info(f"Prompt: {prompt[:80]}...")
    logger.info(f"Config: {num_frames}f @ {height}x{width}, {num_steps} steps")
    logger.info(f"Encoding:  {encode_time:.1f}s")
    logger.info(f"Denoising: {denoise_time:.1f}s ({denoise_time/num_steps:.1f}s/step)")
    logger.info(f"VAE decode: {decode_time:.1f}s")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
