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
    from models.tt_dit.models.transformers.ltx.transformer_ltx import LTXTransformerModel

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


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((1, 1), {})],
    ids=["1x1"],
    indirect=["mesh_device", "device_params"],
)
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


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((1, 1), {})],
    ids=["1x1"],
    indirect=["mesh_device", "device_params"],
)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
