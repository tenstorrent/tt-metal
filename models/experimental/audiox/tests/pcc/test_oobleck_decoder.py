# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from math import prod

import pytest
import torch

from models.experimental.audiox.tt import oobleck as tt_oobleck
from models.experimental.audiox.reference.oobleck import (
    DecoderBlock,
    EncoderBlock,
    OobleckEncoder,
    OobleckDecoder,
    ResidualUnit,
    SnakeBeta,
)
from models.experimental.audiox.tt.decoder_policy import (
    decoder_transpose_input_chunk_size,
    should_stream_decoder_block,
    should_use_long_transpose_profile,
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


def test_should_stream_decoder_block_only_for_really_long_low_channel_tail():
    assert should_stream_decoder_block(input_length=221184, stride=2, out_channels=128) is False
    assert should_stream_decoder_block(input_length=165376, stride=4, out_channels=128) is True
    assert should_stream_decoder_block(input_length=27584, stride=4, out_channels=256) is True
    assert should_stream_decoder_block(input_length=20000, stride=4, out_channels=256) is False
    assert should_stream_decoder_block(input_length=165376, stride=4, out_channels=512) is False


def test_long_transpose_profile_and_chunking():
    assert should_use_long_transpose_profile(input_length=27584, stride=4, out_channels=256) is True
    assert should_use_long_transpose_profile(input_length=41472, stride=4, out_channels=256) is True
    assert should_use_long_transpose_profile(input_length=41472, stride=4, out_channels=128) is True
    assert should_use_long_transpose_profile(input_length=165376, stride=4, out_channels=128) is True
    assert (
        decoder_transpose_input_chunk_size(input_length=27584, stride=4, out_channels=256, default_chunk_size=32768)
        == 32768
    )
    assert (
        decoder_transpose_input_chunk_size(input_length=41472, stride=4, out_channels=256, default_chunk_size=32768)
        == 32768
    )
    assert (
        decoder_transpose_input_chunk_size(input_length=41472, stride=4, out_channels=128, default_chunk_size=32768)
        == 32768
    )


def test_targeted_mid_decoder_chunking_with_raised_long_threshold(monkeypatch):
    monkeypatch.setenv("AUDIOX_TT_CONV_TRANSPOSE_LONG_THRESHOLD", "131072")
    assert should_use_long_transpose_profile(input_length=27584, stride=4, out_channels=256) is True
    assert (
        decoder_transpose_input_chunk_size(input_length=27584, stride=4, out_channels=256, default_chunk_size=32768)
        == 8192
    )


def test_chunked_transpose_keeps_original_long_profile_length(monkeypatch):
    class FakeTensor:
        def __init__(self, length):
            self.shape = (1, length, 1, 256)

        def memory_config(self):
            return "fake-memory"

    calls = []

    def fake_slice_time(_x, start, end):
        return FakeTensor(end - start)

    def fake_conv_transpose1d_impl(
        x,
        weight_height,
        weight_width,
        bias,
        device,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        batch_size,
        input_length,
        profile_input_length=None,
        label="",
        prepared_cache=None,
    ):
        calls.append((input_length, profile_input_length, label))
        return FakeTensor(x.shape[1] * stride), x.shape[1] * stride

    monkeypatch.setattr(tt_oobleck, "_slice_time", fake_slice_time)
    monkeypatch.setattr(tt_oobleck, "_concat_time", lambda chunks, memory_config=None: chunks[0])
    monkeypatch.setattr(tt_oobleck, "_conv_transpose1d_impl", fake_conv_transpose1d_impl)

    out, out_length = tt_oobleck._conv_transpose1d(
        FakeTensor(41472),
        weight_height=None,
        weight_width=None,
        bias=None,
        device=None,
        in_channels=256,
        out_channels=256,
        kernel_size=8,
        stride=4,
        padding=2,
        batch_size=1,
        input_length=41472,
        label="test",
    )

    assert out is not None
    assert out_length == 41472 * 4
    assert calls
    assert all(profile_input_length == 41472 for _, profile_input_length, _ in calls)


def test_stream_upsample_keeps_original_long_profile_length(monkeypatch):
    class FakeTensor:
        def __init__(self, length, channels=128):
            self.shape = (1, length, 1, channels)

        def memory_config(self):
            return "fake-memory"

    calls = []

    def fake_slice_time(_x, start, end):
        return FakeTensor(end - start)

    def fake_conv_transpose1d_impl(
        x,
        weight_height,
        weight_width,
        bias,
        device,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        batch_size,
        input_length,
        profile_input_length=None,
        label="",
        prepared_cache=None,
    ):
        calls.append((input_length, profile_input_length, label))
        return FakeTensor(x.shape[1] * stride, out_channels), x.shape[1] * stride

    monkeypatch.setattr(tt_oobleck, "_slice_time", fake_slice_time)
    monkeypatch.setattr(tt_oobleck, "_gather_chunk_with_halo", lambda chunks, index, halo: (chunks[index], 0))
    monkeypatch.setattr(tt_oobleck, "_conv_transpose1d_impl", fake_conv_transpose1d_impl)
    monkeypatch.setattr(tt_oobleck.ttnn, "deallocate", lambda tensor, force=True: None)

    block = tt_oobleck.TtDecoderBlock.__new__(tt_oobleck.TtDecoderBlock)
    block.up_w = None
    block.up_w_width = None
    block.up_b = None
    block.up_cache = {}
    block.mesh_device = None
    block.in_channels = 256
    block.out_channels = 128
    block.kernel_size = 4
    block.stride = 2
    block.padding = 1

    outputs = block._stream_upsample(FakeTensor(300000, 256))

    assert outputs
    assert calls
    assert all(profile_input_length == 300000 for _, profile_input_length, _ in calls)
    assert block.last_stream_profile["input_length"] == 300000
    assert block.last_stream_profile["base_chunk_count"] == 10
    assert block.last_stream_profile["base_chunk_length"] == 32768
    assert block.last_stream_profile["upsampled_chunk_count"] == 10


def test_streamed_chunk_chain_keeps_original_long_profile_length(monkeypatch):
    class FakeTensor:
        def __init__(self, length, channels=128):
            self.shape = (1, length, 1, channels)

        def memory_config(self):
            return "fake-memory"

    calls = []

    def fake_slice_time(_x, start, end):
        return FakeTensor(end - start)

    def fake_conv_transpose1d_impl(
        x,
        weight_height,
        weight_width,
        bias,
        device,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        batch_size,
        input_length,
        profile_input_length=None,
        label="",
        prepared_cache=None,
    ):
        calls.append((input_length, profile_input_length, label))
        return FakeTensor(x.shape[1] * stride, out_channels), x.shape[1] * stride

    monkeypatch.setattr(tt_oobleck, "_slice_time", fake_slice_time)
    monkeypatch.setattr(tt_oobleck, "_gather_chunk_with_halo", lambda chunks, index, halo: (chunks[index], 0))
    monkeypatch.setattr(tt_oobleck, "_conv_transpose1d_impl", fake_conv_transpose1d_impl)
    monkeypatch.setattr(tt_oobleck, "_split_stream_chunks", lambda chunks, max_chunk_length: chunks)
    monkeypatch.setattr(tt_oobleck.ttnn, "to_layout", lambda tensor, layout: tensor)
    monkeypatch.setattr(tt_oobleck.ttnn, "deallocate", lambda tensor, force=True: None)

    block = tt_oobleck.TtDecoderBlock.__new__(tt_oobleck.TtDecoderBlock)
    block.up_w = None
    block.up_w_width = None
    block.up_b = None
    block.up_cache = {}
    block.mesh_device = None
    block.in_channels = 256
    block.out_channels = 128
    block.kernel_size = 4
    block.stride = 2
    block.padding = 1
    block.act = lambda x: x
    block.res1 = type("FakeRes", (), {"stream": staticmethod(lambda chunks: chunks)})()
    block.res2 = type("FakeRes", (), {"stream": staticmethod(lambda chunks: chunks)})()
    block.res3 = type("FakeRes", (), {"stream": staticmethod(lambda chunks: chunks)})()
    block.last_stream_profile = {}

    outputs = block._stream([FakeTensor(120000, 256), FakeTensor(180000, 256)])

    assert outputs
    assert calls
    assert all(profile_input_length == 300000 for _, profile_input_length, _ in calls)
    assert block.last_stream_profile["input_length"] == 300000
    assert block.last_stream_profile["base_chunk_count"] == 2
    assert block.last_stream_profile["base_chunk_length"] is None
    assert block.last_stream_profile["upsampled_chunk_count_before_split"] == 2
    assert block.last_stream_profile["residual_chunk_count"] == 2
    assert block.last_stream_profile["residual_chunk_length"] == 16384
