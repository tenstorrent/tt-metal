# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU check of LTXCausalConv3d's folded T pad (LTX_VAE_FOLD_TIME_PAD=1).

The fold passes padding=(time_pad // 2, 0, 0) and padding_mode="replicate" to conv3d instead of
concatenating edge-frame copies. This emulates the conv3d vol2col reader index math (replicate:
clamp every index; zeros: out-of-range -> 0) and compares it to the diffusers LTX-2 non-causal conv.
Runs without a device: python -m pytest --noconftest <this file>
"""

from types import SimpleNamespace

import pytest
import torch


def _vol2col_conv3d(x, weight, bias, padding, padding_mode):
    """Stride-1 mirror of reader_vol2col.cpp: idx = out + k - pad, then clamp or zero-pad."""
    _, C, T, H, W = x.shape
    kT, kH, kW = weight.shape[2:]
    sizes = (T, H, W)
    outs = [s + 2 * p - k + 1 for s, p, k in zip(sizes, padding, (kT, kH, kW))]
    cols = []
    for kt in range(kT):
        for kh in range(kH):
            for kw in range(kW):
                idx, oob = [], []
                for k, p, s, n in zip((kt, kh, kw), padding, sizes, outs):
                    i = torch.arange(n) + k - p
                    oob.append((i < 0) | (i >= s))
                    idx.append(i.clamp(0, s - 1))
                patch = x[:, :, idx[0]][:, :, :, idx[1]][:, :, :, :, idx[2]]
                if padding_mode == "zeros":
                    pad = oob[0][:, None, None] | oob[1][None, :, None] | oob[2][None, None, :]
                    patch = patch.masked_fill(pad, 0.0)
                cols.append(patch)
    col = torch.stack(cols, dim=2)  # kernel offsets in (kt, kh, kw) order, like the weight
    w2 = weight.reshape(weight.shape[0], C, kT * kH * kW)
    return torch.einsum("bckthw,ock->bothw", col, w2) + bias[None, :, None, None, None]


def _ltx2_conv(c_in=8, c_out=16):
    diffusers_ltx2 = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_ltx2")
    torch.manual_seed(0)
    return diffusers_ltx2.LTX2VideoCausalConv3d(c_in, c_out, kernel_size=3, stride=1).eval()


def _halo(x):
    # H/W sharded: the halo exchange zero-pads H/W, so conv3d gets an H/W-padded input and H/W pad 0.
    return torch.nn.functional.pad(x, (1, 1, 1, 1))


@pytest.mark.parametrize("T, H, W", [(3, 6, 5), (9, 8, 8), (1, 4, 4)])
def test_folded_time_pad_matches_ltx_noncausal_conv(T, H, W):
    ref = _ltx2_conv()
    x = torch.randn(1, 8, T, H, W)
    with torch.no_grad():
        expected = ref(x, causal=False)
        got = _vol2col_conv3d(_halo(x), ref.conv.weight, ref.conv.bias, (1, 0, 0), "replicate")
    assert got.shape == expected.shape
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_replicate_with_internal_hw_pad_is_not_ltx():
    """Why the fold requires zero internal H/W pad: replicate also clamps H/W."""
    ref = _ltx2_conv()
    x = torch.randn(1, 8, 5, 6, 6) + 1.0
    with torch.no_grad():
        expected = ref(x, causal=False)
        got = _vol2col_conv3d(x, ref.conv.weight, ref.conv.bias, (1, 1, 1), "replicate")
    assert not torch.allclose(got, expected, atol=1e-3)


def test_zero_time_pad_is_not_ltx():
    """The emulator tells the modes apart: a zero T pad differs on the edge frames only."""
    ref = _ltx2_conv()
    x = torch.randn(1, 8, 5, 6, 6) + 1.0
    with torch.no_grad():
        expected = ref(x, causal=False)
        got = _vol2col_conv3d(_halo(x), ref.conv.weight, ref.conv.bias, (1, 0, 0), "zeros")
    assert not torch.allclose(got[:, :, 0], expected[:, :, 0], atol=1e-3)
    torch.testing.assert_close(got[:, :, 1:-1], expected[:, :, 1:-1], rtol=1e-5, atol=1e-5)


def _conv_state(h_factor, w_factor, use_halo_conv=True, persist_pad=True):
    """The LTXCausalConv3d fields _folds_time_pad reads (k=3, both pads external when sharded)."""
    axis = lambda factor: SimpleNamespace(factor=factor)
    return SimpleNamespace(
        fold_time_pad=True,
        external_padding=(0, 1 if h_factor > 1 else 0, 1 if w_factor > 1 else 0),
        parallel_config=SimpleNamespace(height_parallel=axis(h_factor), width_parallel=axis(w_factor)),
        _use_halo_conv=use_halo_conv,
        _persist_pad=persist_pad,
    )


@pytest.mark.parametrize(
    "state, causal, folds",
    [
        (_conv_state(4, 8), False, True),  # persistent-padded path: plain conv3d
        (_conv_state(4, 8), True, False),  # causal T pad is one-sided: keep the concat
        (_conv_state(4, 8, persist_pad=False), False, False),  # halo-buffer conv3d: zeros only
        (_conv_state(4, 8, use_halo_conv=False, persist_pad=False), False, True),  # padded-buffer conv3d
        (_conv_state(4, 1), False, False),  # only H halo -> halo-buffer conv3d
        (_conv_state(4, 1, use_halo_conv=False), False, True),
    ],
    ids=["persist", "causal", "halo_buffer", "no_halo_conv", "h_only_halo", "h_only_no_halo_conv"],
)
def test_folds_time_pad_only_on_plain_conv3d(state, causal, folds):
    vae_ltx = pytest.importorskip("models.tt_dit.models.vae.vae_ltx")
    assert vae_ltx.LTXCausalConv3d._folds_time_pad(state, causal) is folds


def test_folds_time_pad_off_without_flag():
    vae_ltx = pytest.importorskip("models.tt_dit.models.vae.vae_ltx")
    state = _conv_state(4, 8)
    state.fold_time_pad = False
    assert vae_ltx.LTXCausalConv3d._folds_time_pad(state, False) is False
