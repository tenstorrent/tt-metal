# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Conv1d and ResBlock: the bulk of the HiFT vocoder's dispatch count.

HiFT instantiates 8 ResBlocks, each running 6 conv1d and 6 Snake calls over three
dilations, so these two classes account for ~48 convs and ~48 activations per
vocoder invocation. Both run under ttsim (bfloat16 only), so unlike the NSF source
module they can be validated before silicon.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from models.experimental.cosyvoice.tt.common import pcc
from models.experimental.cosyvoice.tt.hifigan.conv import (
    TtConv1d,
    extract_conv_weights,
    fold_weight_norm,
    prepare_weights_default,
)
from models.experimental.cosyvoice.tt.hifigan.resblock import TtResBlock, get_padding

GATE = 0.99
needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)


# --------------------------------------------------------------------------
# host tier
# --------------------------------------------------------------------------
def test_get_padding_matches_reference():
    """ "Same" padding for every (kernel, dilation) HiFT actually uses."""
    for k in (3, 7, 11):
        for d in (1, 3, 5):
            assert get_padding(k, d) == int((k * d - d) / 2)
            length = 64
            out = TtConv1d.out_length(length, k, 1, get_padding(k, d), d)
            assert out == length, f"k={k} d={d} changed length {length} -> {out}"


def test_weight_norm_folding_reproduces_the_module():
    """Folding w = g*v/||v|| must reproduce what the wrapped module computes.

    Every conv in HiFT is weight_norm-wrapped, so an error here would be a
    uniform, plausible-looking distortion across the whole vocoder rather than an
    obvious failure.
    """
    torch.manual_seed(0)
    conv = torch.nn.Conv1d(8, 16, 3, padding=1)
    try:
        from torch.nn.utils.parametrizations import weight_norm
    except ImportError:
        from torch.nn.utils import weight_norm
    wn = weight_norm(conv)

    w, b = extract_conv_weights(wn)
    x = torch.randn(1, 8, 32)
    with torch.no_grad():
        want = wn(x)
        got = torch.nn.functional.conv1d(x, w, b, padding=1)
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()


def test_fold_weight_norm_matches_torch_norm_semantics():
    """The norm is per-output-channel, over every other axis. Getting the axis
    wrong yields a scaled-but-plausible weight."""
    torch.manual_seed(1)
    v = torch.randn(4, 3, 5)
    g = torch.rand(4, 1, 1) + 0.5
    got = fold_weight_norm(v, g, dim=0)
    for oc in range(4):
        want = g[oc] * v[oc] / v[oc].norm(2)
        assert torch.allclose(got[oc], want, atol=1e-6), (got[oc] - want).abs().max()


def test_vocoder_verifies_prepared_weights_by_default(monkeypatch):
    """The vocoder checks each prepared conv geometry against the op once, on every
    architecture. Blackhole used to skip the check, and there `Conv1d(256 -> 256, k=7)` at
    length 8264 returned `inf`: a 1033-frame mel came out as a railed waveform."""
    monkeypatch.delenv("COSYVOICE_CONV_PREPARE", raising=False)
    assert prepare_weights_default(None) is False
    monkeypatch.setenv("COSYVOICE_CONV_PREPARE", "1")
    assert prepare_weights_default(None) is True


# --------------------------------------------------------------------------
# device tier
# --------------------------------------------------------------------------
@needs_l1_small
@pytest.mark.parametrize("kernel,dilation", [(3, 1), (3, 5), (7, 3), (11, 5)])
def test_device_conv1d_matches_torch(device, kernel, dilation):
    """Every (kernel, dilation) pair HiFT's ResBlocks use, at "same" padding."""
    import ttnn

    torch.manual_seed(0)
    C, L = 32, 64
    pad = get_padding(kernel, dilation)
    w = torch.randn(C, C, kernel) * 0.05
    b = torch.randn(C) * 0.01
    x = torch.randn(1, C, L)
    want = torch.nn.functional.conv1d(x, w, b, padding=pad, dilation=dilation)

    op = TtConv1d(device, w, b, padding=pad, dilation=dilation)
    xt = ttnn.from_torch(x.permute(0, 2, 1), dtype=ttnn.bfloat16)
    out, out_len = op(xt, L)
    got = ttnn.to_torch(out).reshape(1, out_len, C).permute(0, 2, 1).float()

    p = pcc(got, want)
    print(f"\n  conv1d k={kernel} d={dilation} PCC {p:.8f}")
    assert out_len == L, (out_len, L)
    assert p >= GATE, p


@needs_l1_small
@pytest.mark.parametrize("kernel", [3, 7])
def test_device_resblock_matches_torch(device, kernel):
    """The full residual stack, channels-last, against the reference in [B, C, L]."""
    import ttnn

    torch.manual_seed(3)
    C, L, dilations = 32, 64, (1, 3, 5)
    convs1 = [torch.nn.Conv1d(C, C, kernel, padding=get_padding(kernel, d), dilation=d) for d in dilations]
    convs2 = [torch.nn.Conv1d(C, C, kernel, padding=get_padding(kernel, 1)) for _ in dilations]
    for m in convs1 + convs2:
        torch.nn.init.normal_(m.weight, std=0.02)
        torch.nn.init.zeros_(m.bias)
    a1 = [torch.rand(C) + 0.5 for _ in dilations]
    a2 = [torch.rand(C) + 0.5 for _ in dilations]

    x = torch.randn(1, C, L) * 0.5
    with torch.no_grad():
        want = TtResBlock.torch_reference(x, convs1, convs2, a1, a2).permute(0, 2, 1)

    op = TtResBlock(device, C, kernel, dilations, convs1=convs1, convs2=convs2, alphas1=a1, alphas2=a2)
    xt = ttnn.from_torch(x.permute(0, 2, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(op(xt, L)).reshape(1, L, C).float()

    p = pcc(got, want)
    print(f"\n  resblock k={kernel} PCC {p:.8f}  max|d| {(got - want).abs().max():.3e}")
    assert got.shape == want.shape, (got.shape, want.shape)
    assert p >= GATE, p


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}, {"l1_small_size": 524288}], indirect=True)
@pytest.mark.parametrize("channels,kernel,length", [(80, 3, 1895), (256, 7, 8264), (256, 11, 8264)])
def test_device_verified_conv1d_matches_torch(device, channels, kernel, length):
    """A verified conv matches torch where prepared weights do not on Blackhole.

    Prepared weights at (80, k=3, 1895) return values of the right size in the wrong places
    (PCC 0.35). At (256, k=7 or 11, 8264) they return `inf` or values off by 1e4, but only
    with the 512 KB L1_SMALL reservation; with the usual 32 KB that geometry is exact.
    Verification has to catch both kinds, the first only
    by comparing outputs element by element, and fall back to the op's own preparation.
    """
    import ttnn

    g = torch.Generator().manual_seed(length + kernel)
    w = torch.randn(channels, channels, kernel, generator=g) / (channels * kernel) ** 0.5
    b = torch.randn(channels, generator=g) * 0.1
    pad = get_padding(kernel, 1)
    conv = TtConv1d(device, w, b, padding=pad)
    conv._verify = True  # what the vocoder sets on each of its convolutions
    xt = ttnn.from_torch(
        torch.randn(1, length, channels, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    out, n = conv(xt, length)
    got = ttnn.to_torch(out).float().reshape(1, n, channels)
    want = F.conv1d(ttnn.to_torch(xt).float().transpose(1, 2), w, b, padding=pad).transpose(1, 2)

    p = pcc(got, want)
    print(f"\n  Conv1d({channels}, k={kernel}) at {length}, verified: PCC {p:.6f}")
    assert got.shape == want.shape, (got.shape, want.shape)
    assert p >= 0.999, p
