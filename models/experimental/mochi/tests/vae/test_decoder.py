import torch
import pytest
from genmo.mochi_preview.vae.models import Decoder, decode_latents
from safetensors.torch import load_file


@torch.no_grad()
def test_decoder():
    # Initialize decoder with same parameters as in pipelines.py
    decoder = Decoder(
        out_channels=3,
        base_channels=128,
        channel_multipliers=[1, 2, 4, 6],
        temporal_expansions=[1, 2, 3],
        spatial_expansions=[2, 2, 2],
        num_res_blocks=[3, 3, 4, 6, 3],
        latent_dim=12,
        has_attention=[False, False, False, False, False],
        output_norm=False,
        nonlinearity="silu",
        output_nonlinearity="silu",
        causal=True,
    )
    print(decoder)

    # Create sample input
    batch_size = 1
    in_channels = 12
    latent_t = 28  # ((num_frames=163 - 1) // TEMPORAL_DOWNSAMPLE=6) + 1
    latent_h = 60  # height=480 // SPATIAL_DOWNSAMPLE=8
    latent_w = 106  # width=848 // SPATIAL_DOWNSAMPLE=8

    z = torch.randn(
        (batch_size, in_channels, latent_t, latent_h, latent_w),
        # device="meta",
    )

    # Run inference
    decoder.eval()
    # decoder = decoder.to(device="meta")

    # output = decode_latents(decoder, z)
    traced_model = torch.jit.trace(decoder, z, check_trace=False)
    traced_model.save("vae_decoder.pt")

    # Verify output shape
    # Output should be [B, C=3, T, H, W] where:
    # T = (latent_t - 1) * 4 (based on docstring in Decoder.forward)
    # H = latent_h * 16 (based on docstring)
    # W = latent_w * 16 (based on docstring)
    # expected_t = (latent_t - 1) * 4
    # expected_h = latent_h * 16
    # expected_w = latent_w * 16

    # assert output.shape == (batch_size, 3, expected_t, expected_h, expected_w), \
    #     f"Expected shape {(batch_size, 3, expected_t, expected_h, expected_w)}, got {output.shape}"

    # # Verify output range (should be scaled to [-1, 1] per docstring)
    # assert output.min() >= -1.0 and output.max() <= 1.0, \
    #     f"Output should be in range [-1, 1], got [{output.min()}, {output.max()}]"
