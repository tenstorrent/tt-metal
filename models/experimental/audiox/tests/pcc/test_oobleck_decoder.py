# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from math import prod

import pytest
import torch

from models.experimental.audiox.reference.oobleck import (
    DecoderBlock,
    EncoderBlock,
    OobleckEncoder,
    OobleckDecoder,
    ResidualUnit,
    SnakeBeta,
)


def test_snake_beta_at_zero_init_is_identity():
    # alpha=beta=0 -> exp -> 1; sin(x)^2 / (1 + eps) ~= sin(x)^2.
    act = SnakeBeta(in_features=4).eval()
    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        y = act(x)
    expected = x + torch.sin(x).pow(2) / (1.0 + act.eps)
    assert torch.allclose(y, expected, atol=1e-5)


def test_residual_unit_preserves_shape():
    block = ResidualUnit(channels=16, dilation=3).eval()
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        y = block(x)
    assert y.shape == x.shape


def test_decoder_block_upsamples_by_stride():
    block = DecoderBlock(in_channels=8, out_channels=4, stride=4).eval()
    x = torch.randn(1, 8, 12)
    with torch.no_grad():
        y = block(x)
    assert y.shape == (1, 4, 12 * 4)


def test_encoder_block_downsamples_by_stride():
    block = EncoderBlock(in_channels=4, out_channels=8, stride=4).eval()
    x = torch.randn(1, 4, 48)
    with torch.no_grad():
        y = block(x)
    assert y.shape == (1, 8, 12)


@pytest.mark.parametrize(
    "latent_dim, channels, c_mults, strides, t_latent",
    [
        # Tiny shape — same architecture, smaller channels, for a fast smoke test.
        (4, 8, (1, 2, 4), (2, 2, 2), 8),
        # AudioX HF config (channels reduced to 16 to keep the test light).
        (64, 16, (1, 2, 4, 8, 16), (2, 4, 4, 8, 8), 4),
    ],
)
def test_oobleck_decoder_shape(latent_dim, channels, c_mults, strides, t_latent):
    decoder = OobleckDecoder(
        out_channels=2,
        channels=channels,
        latent_dim=latent_dim,
        c_mults=c_mults,
        strides=strides,
    ).eval()

    x = torch.randn(1, latent_dim, t_latent)
    with torch.no_grad():
        y = decoder(x)

    assert y.shape == (1, 2, t_latent * prod(strides))


@pytest.mark.parametrize(
    "in_channels, latent_dim, channels, c_mults, strides, t_audio",
    [
        (2, 4, 8, (1, 2, 4), (2, 2, 2), 64),
        (2, 64, 16, (1, 2, 4, 8, 16), (2, 4, 4, 8, 8), 2048),
    ],
)
def test_oobleck_encoder_shape(in_channels, latent_dim, channels, c_mults, strides, t_audio):
    encoder = OobleckEncoder(
        in_channels=in_channels,
        channels=channels,
        latent_dim=latent_dim,
        c_mults=c_mults,
        strides=strides,
    ).eval()

    x = torch.randn(1, in_channels, t_audio)
    with torch.no_grad():
        y = encoder(x)

    assert y.shape == (1, latent_dim, t_audio // prod(strides))


def test_oobleck_decoder_param_names_match_upstream_convention():
    # Upstream uses dac.WNConv1d (= weight_norm(Conv1d)) so the checkpoint
    # ships separate weight_g/weight_v tensors. Our reference must produce
    # the same names so the pretrained loader can drop in directly.
    decoder = OobleckDecoder(out_channels=2, channels=8, latent_dim=4, c_mults=(1, 2), strides=(2, 2))
    names = set(dict(decoder.named_parameters()).keys())
    assert "in_conv.weight_g" in names
    assert "in_conv.weight_v" in names
    assert "out_conv.weight_g" in names
    assert "blocks.0.upsample.weight_g" in names
    assert "blocks.0.res1.conv1.weight_g" in names
    assert "blocks.0.res1.act1.alpha" in names


def test_oobleck_encoder_param_names_match_upstream_convention():
    encoder = OobleckEncoder(in_channels=2, channels=8, latent_dim=4, c_mults=(1, 2), strides=(2, 2))
    names = set(dict(encoder.named_parameters()).keys())
    assert "in_conv.weight_g" in names
    assert "out_conv.weight_g" in names
    assert "blocks.0.downsample.weight_g" in names
    assert "blocks.0.res1.conv1.weight_g" in names
    assert "blocks.0.res1.act1.alpha" in names
