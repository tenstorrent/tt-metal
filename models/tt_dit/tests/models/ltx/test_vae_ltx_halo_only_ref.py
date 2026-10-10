# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU check of the LTX_VAE_HALO_ONLY conv input (neighbor_pad_halo + conv3d halo_buffer).

Emulates, per device, the compact [Htop|Hbot|Wleft|Wright] halo buffer (raw neighbor sticks, zeros at
mesh edges, no masking) and conv3d's halo-mode shard gather (reader_vol2col gather_rows_to_shard:
logical-pad mask first, then input / halo / T-pad). The gathered padded volume must equal the input the
default path hands conv3d: the logical H/W pad zeroed, zero spatial pad, then the T pad (replicate,
zeros or the causal concat). Equal conv inputs in the same reduction order give a bit-exact output.
Run: python -m pytest --noconftest <this file>
"""

import pytest
import torch


def _halo_sections(x, r, c, h_dev, w_dev, ph, pw):
    """Halo buffer sections of device (r, c), indexed like the reader: Htop/Hbot [T, ph, w_dev],
    Wleft/Wright [T, h_dev + 2ph, pw] (corners included)."""
    xz = torch.nn.functional.pad(x, (0, 0, pw, pw, ph, ph))  # [T, H+2ph, W+2pw, C], unmasked
    win = xz[:, r * h_dev : r * h_dev + h_dev + 2 * ph, c * w_dev : c * w_dev + w_dev + 2 * pw]
    return {
        "htop": win[:, :ph, pw : pw + w_dev],
        "hbot": win[:, ph + h_dev :, pw : pw + w_dev],
        "wleft": win[:, :, :pw],
        "wright": win[:, :, pw + w_dev :],
    }


def _gather(x, r, c, h_dev, w_dev, pt, ph, pw, mode, h_mask, w_mask):
    """conv3d halo-mode gather of device (r, c): returns its padded [T+2pt, h_dev+2ph, w_dev+2pw, C] input."""
    shard = x[:, r * h_dev : (r + 1) * h_dev, c * w_dev : (c + 1) * w_dev]
    halo = _halo_sections(x, r, c, h_dev, w_dev, ph, pw)
    t_in = x.shape[0]
    h_start, w_start = r * h_dev, c * w_dev  # pad_offset_tensor page 0
    out = torch.empty(t_in + 2 * pt, h_dev + 2 * ph, w_dev + 2 * pw, x.shape[3])
    u32 = 1 << 32
    for t in range(-pt, t_in + pt):
        t_outside = t < 0 or t >= t_in
        tc = min(max(t, 0), t_in - 1)
        for h in range(-ph, h_dev + ph):
            for w in range(-pw, w_dev + pw):
                dst = out[t + pt, h + ph, w + pw]
                # Global coordinate as uint32: h < 0 on the first row wraps and is masked.
                masked = (h_mask != 0 and (h_start + h) % u32 >= h_mask) or (
                    w_mask != 0 and (w_start + w) % u32 >= w_mask
                )
                h_out = h < 0 or h >= h_dev
                w_out = w < 0 or w >= w_dev
                if masked or (mode == "zeros" and t_outside):
                    dst.zero_()
                elif not (h_out or w_out):
                    dst.copy_(shard[tc, h, w])
                elif w_out:
                    sec = halo["wleft"] if w < 0 else halo["wright"]
                    dst.copy_(sec[tc, h + ph, w + pw if w < 0 else w - w_dev])
                elif h < 0:
                    dst.copy_(halo["htop"][tc, h + ph, w])
                else:
                    dst.copy_(halo["hbot"][tc, h - h_dev, w])
    return out


def _golden(x, r, c, h_dev, w_dev, pt, ph, pw, mode, logical_h, logical_w):
    """Default path: logical pad zeroed, zero spatial pad, T pad as conv3d applies it."""
    xm = x.clone()
    if logical_h:
        xm[:, logical_h:] = 0
    if logical_w:
        xm[:, :, logical_w:] = 0
    xz = torch.nn.functional.pad(xm, (0, 0, pw, pw, ph, ph))
    win = xz[:, r * h_dev : r * h_dev + h_dev + 2 * ph, c * w_dev : c * w_dev + w_dev + 2 * pw]
    if pt == 0:
        return win
    if mode == "replicate":
        return torch.cat([win[:1]] * pt + [win] + [win[-1:]] * pt, dim=0)
    zeros = torch.zeros_like(win[:1])
    return torch.cat([zeros] * pt + [win] + [zeros] * pt, dim=0)


def _host_masks(h_dev, w_dev, rh, rw, logical_h, logical_w):
    # Mirrors LTXCausalConv3d.forward's halo path (fold_w_mask is forced on there).
    h_mask = logical_h if logical_h > 0 and h_dev * rh > logical_h else 0
    w_mask = logical_w if logical_w > 0 and w_dev * rw > logical_w else 0
    return h_mask, w_mask


# (rh, rw, T, H_pad, W_pad, logical_h, logical_w, kind)
#   kind: "replicate" = FOLD_TIME_PAD (pt=1), "zeros" = temporal_padding_mode zeros (pt=1),
#         "causal" = first-frame concat done before the conv (pt=0).
CASES = [
    (2, 4, 3, 18, 32, 17, 30, "replicate"),  # 544x960 on 2x4 (latent 17x30 -> 18x32)
    (2, 4, 3, 18, 32, 17, 30, "causal"),
    (2, 4, 3, 18, 32, 17, 30, "zeros"),
    (2, 4, 2, 36, 64, 34, 60, "replicate"),  # 2x upsampled stage
    (2, 4, 3, 16, 32, 0, 0, "replicate"),  # nothing to mask
    (2, 4, 3, 16, 32, 16, 32, "replicate"),  # logical == padded: masks disabled
    (4, 8, 2, 36, 64, 34, 60, "replicate"),  # 1080p latent on 4x8
    (4, 2, 2, 16, 8, 10, 7, "replicate"),  # masked rows span two devices
]


@pytest.mark.parametrize("rh, rw, T, H, W, logical_h, logical_w, kind", CASES)
def test_halo_only_gather_matches_default(rh, rw, T, H, W, logical_h, logical_w, kind):
    torch.manual_seed(0)
    ph = pw = 1
    pt = 0 if kind == "causal" else 1
    mode = "zeros" if kind == "zeros" else "replicate"
    x = torch.randn(T, H, W, 2)  # values in the pad region are garbage, like real activations
    if kind == "causal":
        x = torch.cat([x[:1], x[:1], x], dim=0)
    h_dev, w_dev = H // rh, W // rw
    h_mask, w_mask = _host_masks(h_dev, w_dev, rh, rw, logical_h, logical_w)
    for r in range(rh):
        for c in range(rw):
            got = _gather(x, r, c, h_dev, w_dev, pt, ph, pw, mode, h_mask, w_mask)
            want = _golden(x, r, c, h_dev, w_dev, pt, ph, pw, mode, logical_h, logical_w)
            assert torch.equal(got, want), f"device ({r},{c}) differs"


def _mock_conv(monkeypatch, env):
    from unittest import mock

    ttnn = pytest.importorskip("ttnn")
    from models.tt_dit.models.vae.vae_ltx import LTXCausalConv3d
    from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig

    if env is None:
        monkeypatch.delenv("LTX_VAE_HALO_ONLY", raising=False)
    else:
        monkeypatch.setenv("LTX_VAE_HALO_ONLY", env)
    mesh = mock.MagicMock()
    mesh.compute_with_storage_grid_size.return_value = ttnn.CoreCoord(13, 10)
    mesh.arch.return_value = ttnn.device.Arch.BLACKHOLE
    pc = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=2, mesh_axis=0), width_parallel=ParallelFactor(factor=4, mesh_axis=1)
    )
    return LTXCausalConv3d(128, 128, kernel_size=3, mesh_device=mesh, parallel_config=pc, ccl_manager=mock.MagicMock())


@pytest.mark.parametrize("env, expected", [(None, True), ("0", False), ("1", True)])
def test_halo_only_default_on(monkeypatch, env, expected):
    assert _mock_conv(monkeypatch, env).halo_only is expected


@pytest.mark.parametrize("causal", [False, True])
def test_halo_only_forward_wiring(monkeypatch, causal):
    """With the flag on, forward skips neighbor_pad_persistent_buffer and hands conv3d the halo buffer,
    the spatial padding, the logical masks and a per-device pad offset."""
    from unittest import mock

    ttnn = pytest.importorskip("ttnn")
    import models.tt_dit.models.vae.vae_ltx as vae_ltx

    conv = _mock_conv(monkeypatch, "1")
    conv.weight = mock.MagicMock()
    conv.bias = mock.MagicMock()
    offsets = {}

    def fake_2dshard(t, device, shard_mapping, layout, dtype):
        offsets["t"], offsets["map"], offsets["dtype"] = t, shard_mapping, dtype
        return "pad_offset"

    monkeypatch.setattr(vae_ltx, "typed_tensor_2dshard", fake_2dshard)
    conv3d = mock.MagicMock(return_value="out")
    monkeypatch.setattr(ttnn.experimental, "conv3d", conv3d)
    concat = mock.MagicMock()
    monkeypatch.setattr(ttnn, "concat", concat)
    x = mock.MagicMock()
    x.layout = ttnn.ROW_MAJOR_LAYOUT
    x.shape = (1, 5, 9, 8, 128)
    x.__getitem__.return_value = x
    concat.return_value = x
    ccl = conv.ccl_manager
    ccl.num_links = 2
    ccl.neighbor_pad_halo_only.return_value = "halo"

    assert conv.forward(x, causal=causal, logical_h=17, logical_w=30) == "out"

    ccl.neighbor_pad_persistent_buffer.assert_not_called()
    np_kwargs = ccl.neighbor_pad_halo_only.call_args.kwargs
    assert np_kwargs["dims"] == [2, 3] and np_kwargs["pad_left"] == [1, 1] and np_kwargs["pad_right"] == [1, 1]
    assert np_kwargs["topology"] == ttnn.Topology.Linear
    kw = conv3d.call_args.kwargs
    assert kw["halo_buffer"] == "halo"
    assert kw["logical_h_mask"] == 17 and kw["logical_w_mask"] == 30
    assert kw["pad_offset_tensor"] == "pad_offset"
    if causal:
        assert kw["padding"] == (0, 1, 1) and kw["padding_mode"] == "zeros"
        concat.assert_called_once()
    else:
        assert kw["padding"] == (1, 1, 1) and kw["padding_mode"] == "replicate"
        concat.assert_not_called()
    assert offsets["map"] == {0: 0, 1: 1} and offsets["dtype"] == ttnn.uint32
    assert offsets["t"][:, :, 0].tolist() == [[0] * 4, [9] * 4]
    assert offsets["t"][:, :, 1].tolist() == [[0, 8, 16, 24]] * 2
