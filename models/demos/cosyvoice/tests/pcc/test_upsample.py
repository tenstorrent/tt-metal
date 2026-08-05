# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ConvTranspose1d via conv_transpose2d at H=1 -- HiFT's upsampling stages.

TTNN has no native 1-D transpose, so both places HiFT needs one (the two
upsample stages, and the iSTFT's overlap-add) route through conv_transpose2d
with a degenerate height. These tests pin the two shape families that matters:

    upsample   k=16, stride=8, padding=4    (upsample_rates [8, 8])
    iSTFT OLA  k=16, stride=4, padding=0    (see test_istft.py)

If the H=1 path turns out to carry large constant overhead, that measurement is
the case for proposing a native ttnn.conv_transpose1d (03_plan.md P3).
"""
from __future__ import annotations

import pytest
import torch

from models.demos.cosyvoice.tt.common import pcc
from models.demos.cosyvoice.tt.hifigan.upsample import TtConvTranspose1d

GATE = 0.99
needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)


def test_out_length_matches_torch():
    """Output length must match conv_transpose1d exactly -- the two upsample
    stages multiply, so an off-by-one here becomes an off-by-64 at the vocoder
    output and shows up as a length mismatch against the mel hop."""
    for L in (8, 16, 64, 282):
        w = torch.zeros(4, 2, 16)
        op_len = TtConvTranspose1d.__dict__["out_length"]
        got = op_len(type("S", (), dict(stride=8, padding=4, dilation=1, kernel_size=16))(), L)
        want = torch.nn.functional.conv_transpose1d(torch.zeros(1, 4, L), w, stride=8, padding=4).shape[-1]
        assert got == want, (L, got, want)


def test_total_upsample_matches_mel_hop():
    """8 * 8 * 4 = 256 must equal the mel hop_size, or the vocoder output length
    stops matching the input frame count."""
    from models.demos.cosyvoice.tt.model_config import DEFAULT

    assert DEFAULT.hift.total_upsample == DEFAULT.mel.hop_size == 256


@needs_l1_small
@pytest.mark.parametrize("in_ch,out_ch,length", [(32, 16, 16), (64, 32, 32)])
def test_device_conv_transpose1d_matches_torch(device, in_ch, out_ch, length):
    """HiFT's real upsample parameters: k=16, stride=8, padding=(16-8)//2=4."""
    import ttnn

    torch.manual_seed(0)
    k, s, p = 16, 8, 4
    w = torch.randn(in_ch, out_ch, k) * 0.05
    b = torch.randn(out_ch) * 0.01
    x = torch.randn(1, in_ch, length)
    want = torch.nn.functional.conv_transpose1d(x, w, b, stride=s, padding=p)

    op = TtConvTranspose1d(device, w, b, stride=s, padding=p)
    xt = ttnn.from_torch(x.permute(0, 2, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out, lo = op(xt, length)
    got = ttnn.to_torch(out).reshape(1, lo, out_ch).permute(0, 2, 1).float()

    p_ = pcc(got, want)
    print(f"\n  convT1d {in_ch}->{out_ch} L={length}->{lo}  PCC {p_:.8f}")
    assert lo == want.shape[-1] == length * s, (lo, want.shape[-1])
    assert p_ >= GATE, p_
