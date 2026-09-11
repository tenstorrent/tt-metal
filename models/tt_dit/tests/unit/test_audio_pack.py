# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks of the time-packed weight construction in ``layers/audio_pack.py``: the dense packed conv on
``(T/k, k*C)`` rows must reproduce the unpacked op (exactly for zero-padded convs; away from the sequence ends for
the replicate-padded resamplers) for the vocoder's late-band shapes."""

import math

import pytest
import torch
import torch.nn.functional as F

from models.tt_dit.layers.audio_pack import (
    conv1d_same,
    downsample2x_ref,
    kaiser_taps,
    packed_weight,
    upsample2x_ref,
)

BANDS = [(8, 4), (16, 2)]  # (channels, pack) -> 32-wide packed rows


def _pack(x_1CT, k):
    """``(1, C, T) -> (1, k*C, T/k)`` slot-major (a row-major reshape of ``(T, C)``)."""
    c, t = x_1CT.shape[1], x_1CT.shape[2]
    return x_1CT[0].transpose(0, 1).reshape(t // k, k * c).transpose(0, 1).unsqueeze(0)


def _unpack(y_1PT, k, c):
    p, t = y_1PT.shape[1], y_1PT.shape[2]
    return y_1PT[0].transpose(0, 1).reshape(t * k, c).transpose(0, 1).unsqueeze(0)


def _psnr(ref, test):
    mse = torch.mean((ref - test) ** 2).item()
    return float("inf") if mse == 0 else 20 * math.log10(ref.abs().max().item()) - 10 * math.log10(mse)


@pytest.mark.parametrize(("channels", "pack"), BANDS)
@pytest.mark.parametrize("kernel", [3, 7, 11])
@pytest.mark.parametrize("dilation", [1, 3, 5])
def test_packed_conv_matches_conv1d(channels, pack, kernel, dilation):
    torch.manual_seed(0)
    w = torch.randn(channels, channels, kernel)
    t = 64 * pack
    x = torch.randn(1, channels, t)
    ref = conv1d_same(w, dilation)(x.double()).float()
    wp = packed_weight(
        conv1d_same(w, dilation),
        c_in=channels,
        c_out=channels,
        k_in=pack,
        k_out=pack,
        support=(kernel - 1) * dilation + 1,
    )
    assert wp.shape[0] == pack * channels and wp.shape[1] == pack * channels and wp.shape[-1] % 2 == 1
    out = _unpack(F.conv1d(_pack(x, pack), wp, padding=wp.shape[-1] // 2), pack, channels)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), (out - ref).abs().max()


@pytest.mark.parametrize(("channels", "pack"), BANDS)
def test_packed_upsample_matches_interior(channels, pack):
    torch.manual_seed(1)
    taps = kaiser_taps()
    t = 64 * pack
    x = torch.randn(1, channels, t)
    ref = upsample2x_ref(taps, channels)(x.double()).float()
    wp = packed_weight(
        upsample2x_ref(taps, channels), c_in=channels, c_out=channels, k_in=pack, k_out=2 * pack, support=26
    )
    out = _unpack(F.conv1d(_pack(x, pack), wp, padding=wp.shape[-1] // 2), 2 * pack, channels)
    assert out.shape == ref.shape
    edge = 2 * pack * (wp.shape[-1] // 2 + 1)  # samples the zero-vs-replicate end padding can reach
    assert torch.allclose(out[..., edge:-edge], ref[..., edge:-edge], atol=1e-4, rtol=1e-4)
    assert _psnr(ref, out) > 40, "end padding difference dominates the whole clip"


@pytest.mark.parametrize(("channels", "pack"), BANDS)
def test_packed_downsample_matches_interior(channels, pack):
    torch.manual_seed(2)
    taps = kaiser_taps()
    k_in = 2 * pack
    t = 64 * k_in
    x = torch.randn(1, channels, t)
    ref = downsample2x_ref(taps, channels)(x.double()).float()
    wp = packed_weight(
        downsample2x_ref(taps, channels), c_in=channels, c_out=channels, k_in=k_in, k_out=pack, support=26
    )
    out = _unpack(F.conv1d(_pack(x, k_in), wp, padding=wp.shape[-1] // 2), pack, channels)
    assert out.shape == ref.shape
    edge = pack * (wp.shape[-1] // 2 + 1)
    assert torch.allclose(out[..., edge:-edge], ref[..., edge:-edge], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(("channels", "pack"), BANDS)
def test_packed_kernel_widths(channels, pack):
    """The packed kernels stay short: this is what keeps the dense form's FLOPs at today's padded-conv level."""
    widths = {}
    for kernel, dilation in [(3, 1), (7, 1), (11, 1), (3, 3), (7, 3), (11, 3), (3, 5), (7, 5), (11, 5)]:
        wp = packed_weight(
            conv1d_same(torch.ones(channels, channels, kernel), dilation),
            c_in=channels,
            c_out=channels,
            k_in=pack,
            k_out=pack,
            support=(kernel - 1) * dilation + 1,
        )
        widths[(kernel, dilation)] = wp.shape[-1]
        assert wp.shape[-1] == 2 * math.ceil((kernel - 1) * dilation / 2 / pack) + 1
    up = packed_weight(
        upsample2x_ref(kaiser_taps(), channels), c_in=channels, c_out=channels, k_in=pack, k_out=2 * pack, support=26
    )
    down = packed_weight(
        downsample2x_ref(kaiser_taps(), channels), c_in=channels, c_out=channels, k_in=2 * pack, k_out=pack, support=26
    )
    print(f"C={channels} k={pack}: conv taps {widths}, up {up.shape}, down {down.shape}")
