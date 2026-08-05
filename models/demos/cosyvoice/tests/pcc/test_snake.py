# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Snake activation, host and device.

Snake is exhaustively checkable against a one-line torch reference, which makes it
the cheapest place to catch a broadcast or dtype mistake before the same mistake
shows up buried inside a ResBlock.
"""
from __future__ import annotations

import pytest
import torch

from models.demos.cosyvoice.tt.common import pcc
from models.demos.cosyvoice.tt.hifigan.snake import TtSnake

GATE = 0.999

needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)


def _reference_snake(x, alpha, logscale=False):
    """cosyvoice.transformer.activation.Snake.forward, transcribed from source.

    Deliberately written out rather than imported: the CosyVoice package lives in a
    different venv, so importing it here would make a tt-metal test depend on the
    reference environment being installed.
    """
    a = alpha.unsqueeze(0).unsqueeze(-1)
    if logscale:
        a = torch.exp(a)
    return x + (1.0 / (a + 1e-9)) * torch.pow(torch.sin(x * a), 2)


@pytest.mark.parametrize("logscale", [False, True])
def test_torch_reference_matches_source(logscale):
    """Our torch_reference must equal the reference implementation exactly."""
    torch.manual_seed(0)
    x = torch.randn(2, 64, 128)
    alpha = torch.rand(64) * 2.0
    got = TtSnake.torch_reference(x, alpha, alpha_logscale=logscale)
    want = _reference_snake(x, alpha, logscale)
    assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max()


def test_alpha_of_one_is_the_textbook_form():
    """With alpha == 1, Snake collapses to x + sin^2(x). A named special case is
    worth pinning: it catches a reciprocal applied on the wrong side."""
    torch.manual_seed(1)
    x = torch.randn(1, 8, 32)
    got = TtSnake.torch_reference(x, torch.ones(8))
    want = x + torch.sin(x).pow(2)
    assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max()


def test_alpha_broadcasts_per_channel_not_per_sample():
    """alpha is per-CHANNEL. If it were applied per timestep the result would still
    look plausible, so this asserts the axis explicitly."""
    x = torch.ones(1, 3, 5)
    alpha = torch.tensor([0.5, 1.0, 2.0])
    got = TtSnake.torch_reference(x, alpha)
    for c, a in enumerate(alpha):
        want_c = 1.0 + (1.0 / (a + 1e-9)) * torch.sin(a).pow(2)
        assert torch.allclose(got[0, c], want_c.expand(5), atol=1e-6)


def test_channels_last_and_channels_first_agree():
    """The device path is channels-last while the reference is channels-first.
    Both spellings must produce the same values under transposition -- otherwise
    alpha is broadcasting along the wrong axis, which is a silent wrong answer
    whenever C happens to equal T."""
    torch.manual_seed(5)
    C, T = 8, 16
    x_cf = torch.randn(1, C, T)
    alpha = torch.rand(C) + 0.5
    cf = TtSnake.torch_reference(x_cf, alpha)
    cl = TtSnake.torch_reference(x_cf.permute(0, 2, 1).contiguous(), alpha, channels_last=True)
    assert torch.allclose(cf, cl.permute(0, 2, 1), atol=1e-6), (cf - cl.permute(0, 2, 1)).abs().max()


@needs_l1_small
@pytest.mark.parametrize("channels,length", [(64, 256), (512, 4096)])
def test_device_snake_matches_host(device, channels, length):
    """The composed five-op form on device, channels-last as HiFT carries it.
    512 x 4096 is the shape it actually hits after the second upsample stage."""
    import ttnn

    torch.manual_seed(3)
    x_t = torch.randn(1, length, channels)  # [B, T, C]
    alpha_t = torch.rand(channels) * 1.5 + 0.25  # away from 0, where 1/alpha explodes
    want = TtSnake.torch_reference(x_t, alpha_t, channels_last=True)

    op = TtSnake(device, alpha_t)
    x = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(op(x)).float()

    p = pcc(got, want)
    print(f"\n  snake C={channels} T={length} PCC {p:.10f}  max|d| {(got - want).abs().max():.3e}")
    assert got.shape == want.shape, (got.shape, want.shape)
    assert p >= GATE, p
