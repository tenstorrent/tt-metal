# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Loader tests build the upstream-style Sequential layout in PyTorch, dump
its state_dict, run our remap, and check every key lands. We don't pull in
the upstream package itself — replicating the structure with vanilla
``nn.Sequential`` is enough to exercise the index math."""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from models.experimental.audiox.reference.dit import DiffusionTransformer
from models.experimental.audiox.reference.oobleck import OobleckDecoder, OobleckEncoder
from models.experimental.audiox.utils.loader import (
    _resolve_oobleck_encoder_block_key,
    _resolve_oobleck_decoder_block_key,
    load_into,
    remap_conditioner_state_dict,
    remap_dit_state_dict,
    remap_oobleck_decoder_state_dict,
    remap_oobleck_encoder_state_dict,
)


def _wn_conv(in_c, out_c, k, **kw):
    return weight_norm(nn.Conv1d(in_c, out_c, k, **kw))


def _wn_convT(in_c, out_c, k, stride):
    return weight_norm(nn.ConvTranspose1d(in_c, out_c, k, stride=stride, padding=stride // 2))


class _UpstreamSnake(nn.Module):
    """Stand-in for upstream's snake activation — same param names (``alpha``,
    ``beta``) as our SnakeBeta, packed flat (no inner Sequential)."""

    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))


class _UpstreamResidualUnit(nn.Module):
    """Sequential[snake, conv, snake, conv]."""

    def __init__(self, channels, dilation):
        super().__init__()
        pad = (dilation * 6) // 2
        self.layers = nn.Sequential(
            _UpstreamSnake(channels),
            _wn_conv(channels, channels, 7, dilation=dilation, padding=pad),
            _UpstreamSnake(channels),
            _wn_conv(channels, channels, 1),
        )


class _UpstreamDecoderBlock(nn.Module):
    """Sequential[snake, upsample, res1, res2, res3]."""

    def __init__(self, in_c, out_c, stride):
        super().__init__()
        self.layers = nn.Sequential(
            _UpstreamSnake(in_c),
            _wn_convT(in_c, out_c, 2 * stride, stride=stride),
            _UpstreamResidualUnit(out_c, dilation=1),
            _UpstreamResidualUnit(out_c, dilation=3),
            _UpstreamResidualUnit(out_c, dilation=9),
        )


class _UpstreamOobleckDecoder(nn.Module):
    """Mirror of upstream's OobleckDecoder Sequential layout."""

    def __init__(self, latent_dim, channels, c_mults, strides, out_channels=2):
        super().__init__()
        c_mults_ext = [1] + list(c_mults)
        depth = len(c_mults_ext)
        layers = [_wn_conv(latent_dim, c_mults_ext[-1] * channels, 7, padding=3)]
        for i in range(depth - 1, 0, -1):
            layers.append(
                _UpstreamDecoderBlock(
                    c_mults_ext[i] * channels,
                    c_mults_ext[i - 1] * channels,
                    strides[i - 1],
                )
            )
        layers.append(_UpstreamSnake(c_mults_ext[0] * channels))
        layers.append(_wn_conv(c_mults_ext[0] * channels, out_channels, 7, padding=3, bias=False))
        layers.append(nn.Identity())  # tanh slot, no params
        self.layers = nn.Sequential(*layers)


class _UpstreamEncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()
        self.layers = nn.Sequential(
            _UpstreamResidualUnit(in_c, dilation=1),
            _UpstreamResidualUnit(in_c, dilation=3),
            _UpstreamResidualUnit(in_c, dilation=9),
            _UpstreamSnake(in_c),
            _wn_conv(in_c, out_c, 2 * stride, stride=stride, padding=(stride + 1) // 2),
        )


class _UpstreamOobleckEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim, channels, c_mults, strides):
        super().__init__()
        c_mults_ext = [1] + list(c_mults)
        depth = len(c_mults_ext)
        layers = [_wn_conv(in_channels, c_mults_ext[0] * channels, 7, padding=3)]
        for i in range(depth - 1):
            layers.append(
                _UpstreamEncoderBlock(
                    c_mults_ext[i] * channels,
                    c_mults_ext[i + 1] * channels,
                    strides[i],
                )
            )
        layers.append(_UpstreamSnake(c_mults_ext[-1] * channels))
        layers.append(_wn_conv(c_mults_ext[-1] * channels, latent_dim, 3, padding=1))
        self.layers = nn.Sequential(*layers)


def test_resolve_oobleck_block_key_in_conv():
    assert _resolve_oobleck_decoder_block_key("layers.0.weight_g", n_blocks=3) == "in_conv.weight_g"
    assert _resolve_oobleck_decoder_block_key("layers.0.bias", n_blocks=3) == "in_conv.bias"
    assert _resolve_oobleck_encoder_block_key("layers.0.weight_g", n_blocks=3) == "in_conv.weight_g"
    assert _resolve_oobleck_encoder_block_key("layers.0.bias", n_blocks=3) == "in_conv.bias"


def test_resolve_oobleck_block_key_decoder_block():
    # n_blocks=3 -> blocks at upstream layers.{1,2,3}; out_act at 4, out_conv at 5.
    assert _resolve_oobleck_decoder_block_key("layers.1.layers.0.alpha", n_blocks=3) == "blocks.0.act.alpha"
    assert _resolve_oobleck_decoder_block_key("layers.1.layers.1.weight_g", n_blocks=3) == "blocks.0.upsample.weight_g"
    # Inside res1: layers.<i>.layers.<2>.layers.<k>
    assert (
        _resolve_oobleck_decoder_block_key("layers.2.layers.2.layers.1.weight_g", n_blocks=3)
        == "blocks.1.res1.conv1.weight_g"
    )
    assert (
        _resolve_oobleck_decoder_block_key("layers.3.layers.4.layers.0.alpha", n_blocks=3) == "blocks.2.res3.act1.alpha"
    )
    assert (
        _resolve_oobleck_encoder_block_key("layers.1.layers.0.layers.1.weight_g", n_blocks=3)
        == "blocks.0.res1.conv1.weight_g"
    )
    assert _resolve_oobleck_encoder_block_key("layers.2.layers.3.alpha", n_blocks=3) == "blocks.1.act.alpha"
    assert (
        _resolve_oobleck_encoder_block_key("layers.3.layers.4.weight_g", n_blocks=3) == "blocks.2.downsample.weight_g"
    )


def test_resolve_oobleck_block_key_tail():
    assert _resolve_oobleck_decoder_block_key("layers.4.alpha", n_blocks=3) == "out_act.alpha"
    assert _resolve_oobleck_decoder_block_key("layers.5.weight_g", n_blocks=3) == "out_conv.weight_g"
    assert _resolve_oobleck_encoder_block_key("layers.4.alpha", n_blocks=3) == "out_act.alpha"
    assert _resolve_oobleck_encoder_block_key("layers.5.weight_g", n_blocks=3) == "out_conv.weight_g"
    # Tanh / Identity tail has no params, but a stray key would be dropped.
    assert _resolve_oobleck_decoder_block_key("layers.6.something", n_blocks=3) is None
    assert _resolve_oobleck_encoder_block_key("layers.6.something", n_blocks=3) is None


def test_oobleck_decoder_remap_round_trip_loads_into_our_decoder():
    """End-to-end: build upstream-style decoder, dump state_dict, remap, load
    into our OobleckDecoder, expect 0 missing and 0 unexpected keys."""
    c_mults = (1, 2, 4)
    strides = (2, 2, 2)
    latent_dim = 4
    channels = 8

    upstream = _UpstreamOobleckDecoder(latent_dim=latent_dim, channels=channels, c_mults=c_mults, strides=strides)
    ours = OobleckDecoder(out_channels=2, channels=channels, latent_dim=latent_dim, c_mults=c_mults, strides=strides)

    remapped = remap_oobleck_decoder_state_dict(upstream.state_dict(), prefix="", n_blocks=len(c_mults))
    missing, unexpected = load_into(ours, remapped, label="oobleck")
    assert missing == [] and unexpected == []


def test_oobleck_encoder_remap_round_trip_loads_into_our_encoder():
    c_mults = (1, 2, 4)
    strides = (2, 2, 2)
    latent_dim = 4
    channels = 8

    upstream = _UpstreamOobleckEncoder(
        in_channels=2,
        latent_dim=latent_dim,
        channels=channels,
        c_mults=c_mults,
        strides=strides,
    )
    ours = OobleckEncoder(
        in_channels=2,
        channels=channels,
        latent_dim=latent_dim,
        c_mults=c_mults,
        strides=strides,
    )

    remapped = remap_oobleck_encoder_state_dict(upstream.state_dict(), prefix="", n_blocks=len(c_mults))
    missing, unexpected = load_into(ours, remapped, label="oobleck_encoder")
    assert missing == [] and unexpected == []


def test_dit_remap_strips_prefix_and_loads():
    """DiT keys already match upstream — only the two-level wrapper prefix needs stripping."""
    dit = DiffusionTransformer(io_channels=16, embed_dim=64, depth=2, num_heads=4, cond_token_dim=16)
    # Upstream wraps the DiT under model.model.<...> (ConditionedDiffusionModelWrapper -> DiTWrapper -> DiffusionTransformer).
    sd = {f"model.model.{k}": v for k, v in dit.state_dict().items()}

    remapped = remap_dit_state_dict(sd)
    fresh = DiffusionTransformer(io_channels=16, embed_dim=64, depth=2, num_heads=4, cond_token_dim=16)
    missing, unexpected = load_into(fresh, remapped, label="dit")
    assert missing == [] and unexpected == []


def test_conditioner_remap_isolates_one_id():
    """Multi-conditioner state_dict slice: keep only the matching id."""
    sd = {
        "conditioner.conditioners.text_prompt.proj_out.weight": torch.zeros(8, 4),
        "conditioner.conditioners.text_prompt.proj_out.bias": torch.zeros(8),
        "conditioner.conditioners.video_prompt.empty_visual_feat": torch.zeros(1, 128, 768),
        "conditioner.conditioners.audio_prompt.empty_audio_feat": torch.zeros(1, 215, 768),
    }
    text = remap_conditioner_state_dict(sd, conditioner_id="text_prompt")
    assert set(text.keys()) == {"proj_out.weight", "proj_out.bias"}

    video = remap_conditioner_state_dict(sd, conditioner_id="video_prompt")
    assert set(video.keys()) == {"empty_visual_feat"}
