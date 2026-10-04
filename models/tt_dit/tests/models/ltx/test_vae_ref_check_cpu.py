# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the VAE A/B reference check: chip-boundary seam indexing and the pass/fail gates.

Run: python -m pytest --noconftest <this file>
"""

import pytest
import torch

from models.tt_dit.tests.models.ltx.tools import vae_ref_check as vrc


@pytest.mark.parametrize(
    "out_size, factor, exact, expected",
    [
        # 4x8 1080p columns: latent W 60 pads to 64 (8/chip, 256 px); exact after s0: 120/8 = 15 -> 240 px.
        (1920, 8, False, [256, 512, 768, 1024, 1280, 1536, 1792]),
        (1920, 8, True, sorted({256, 512, 768, 1024, 1280, 1536, 1792} | {240 * k for k in range(1, 8)})),
        # 4x8 1080p rows: latent H 34 pads to 36 (9/chip, 288 px); 68 % 4 == 0 after s0 -> 17/chip, 272 px.
        (1088, 4, False, [288, 576, 864]),
        (1088, 4, True, [272, 288, 544, 576, 816, 864]),
        # 2x4 544x960, the A/B shape: the same per-chip shard as 1080p on 4x8.
        (960, 4, False, [256, 512, 768]),
        (960, 4, True, [240, 256, 480, 512, 720, 768]),
        (544, 2, False, [288]),
        (544, 2, True, [272, 288]),
        # Divides from the latent on: exact shard changes nothing.
        (1024, 4, True, [256, 512, 768]),
        (1024, 1, False, []),
    ],
)
def test_chip_boundaries(out_size, factor, exact, expected):
    assert vrc.chip_boundaries(out_size, factor, exact_shard=exact) == expected


def test_boundaries_match_decoder_upsamples():
    """The tool's upsample chain must match the decoder's: walk the production blocks through _DECODER_STRIDE_MAP."""
    from models.tt_dit.models.vae import vae_ltx
    from models.tt_dit.tests.models.ltx.test_vae_ltx_exact_shard_ref import _PROD_DECODER_BLOCKS

    spatial = []
    for name, _ in reversed(_PROD_DECODER_BLOCKS):
        if name in vae_ltx._DECODER_STRIDE_MAP:
            _, p2, p3 = vae_ltx._DECODER_STRIDE_MAP[name]
            assert p2 == p3
            if p2 > 1:
                spatial.append(p2)
    assert tuple(spatial) == vrc.LTX_SPATIAL_UPSAMPLES


def test_boundaries_match_reshard_runtime_split(monkeypatch):
    """Exact-shard boundaries equal the runtime per-chip widths of the exact-shard CPU emulation."""
    from models.tt_dit.tests.models.ltx import test_vae_ltx_exact_shard_ref as es

    widths = es._runtime_conv_dims((4, 8), 1088, 1920, True, monkeypatch)  # per-chip (H, W) at each conv
    scales = [32] * 3 + [16] * 2 + [8] * 4 + [4] * 2  # px per unit at conv_in, before each block, conv_out
    expected_cols = set()
    for (_, w), s in zip(widths, scales):
        expected_cols.update(k * w * s for k in range(1, 8) if k * w * s < 1920)
    assert len(widths) == len(scales)
    assert vrc.chip_boundaries(1920, 8, exact_shard=True) == sorted(expected_cols)


def test_seam_index_bands_and_clipping():
    assert vrc.seam_index(100, [40, 70], band=4).tolist() == list(range(36, 44)) + list(range(66, 74))
    # Bands are clipped to the frame.
    assert vrc.seam_index(100, [2, 98], band=4).tolist() == list(range(0, 6)) + list(range(94, 100))
    assert vrc.seam_index(64, [], band=4).size == 0
    # Overlapping bands are de-duplicated.
    assert vrc.seam_index(64, [30, 32], band=4).tolist() == list(range(26, 36))


def _yuv(t, h, w, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(16, 236, (t, h * 3 // 2, w), dtype=torch.uint8, generator=g)


def _off_by_30(x):
    return torch.where(x > 128, x - 30, x + 30)


def test_seam_damage_fails_only_the_seam_gate():
    """A wrong halo at one column boundary: overall PSNR stays above 40 dB, the seam gate catches it."""
    h, w = 544, 960
    ref = _yuv(2, h, w)
    arm = ref.clone()
    col = vrc.chip_boundaries(w, 4)[0]
    arm[:, :h, col - 2 : col + 2] = _off_by_30(arm[:, :h, col - 2 : col + 2])
    m = vrc.compare(arm, ref, height=h, width=w, mesh_shape=(2, 4))
    assert m["psnr"] >= vrc.PSNR_MIN_DB
    assert m["seam_col_psnr"] < vrc.SEAM_PSNR_MIN_DB
    assert m["seam_row_psnr"] >= vrc.SEAM_PSNR_MIN_DB
    assert [f.split()[0] for f in vrc.failures(m)] == ["seam_col_psnr"]


def test_row_seam_and_identical():
    h, w = 544, 960
    ref = _yuv(2, h, w, seed=1)
    m = vrc.compare(ref, ref, height=h, width=w, mesh_shape=(2, 4))
    assert m["psnr"] == 99.0 and m["seam_col_psnr"] == 99.0 and abs(m["ssim_y"] - 1) < 1e-9
    assert vrc.failures(m) == []
    arm = ref.clone()
    row = vrc.chip_boundaries(h, 2)[0]
    arm[:, row - 2 : row + 2, :] = _off_by_30(arm[:, row - 2 : row + 2, :])
    assert [f.split()[0] for f in vrc.failures(vrc.compare(arm, ref, height=h, width=w, mesh_shape=(2, 4)))] == [
        "seam_row_psnr"
    ]


def test_global_noise_fails_overall():
    h, w = 64, 128
    ref = _yuv(1, h, w, seed=2)
    noise = torch.randint(-40, 41, ref.shape, generator=torch.Generator().manual_seed(3))
    arm = (ref.int() + noise).clamp(0, 255).to(torch.uint8)
    fails = vrc.failures(vrc.compare(arm, ref, height=h, width=w, mesh_shape=(1, 1)))
    assert fails and fails[0].startswith("psnr")


def test_float_reference_to_yuv():
    video = torch.full((1, 3, 2, 4, 4), -1.0)
    video[:, :, :, :2] = 1.0  # top half white, bottom half black
    yuv = vrc.float_to_yuv420p(video)
    assert yuv.shape == (2, 6, 4)
    y, c = vrc.split_yuv420p(yuv, 4, 4)
    assert y[:, :2].eq(235).all() and y[:, 2:].eq(16).all() and c.eq(128).all()


def test_check_against_reference_record_and_md5(tmp_path, monkeypatch):
    h, w, nf = 64, 128, 2
    lat = torch.randn(1, 4, 1, 2, 4)
    out = _yuv(nf, h, w).numpy()
    monkeypatch.setenv("LTX_VAE_REF_DIR", str(tmp_path))
    monkeypatch.delenv("LTX_VAE_REF", raising=False)
    kw = dict(latent=lat, num_frames=nf, height=h, width=w, mesh_shape=(2, 4))
    with pytest.raises(AssertionError, match="no VAE reference"):  # allow-pytest.raises: runs with --noconftest
        vrc.check_against_reference(out, "a", **kw)
    monkeypatch.setenv("LTX_VAE_REF_RECORD", "1")
    monkeypatch.setenv("LTX_VAE_HALO_ONLY", "1")
    with pytest.raises(AssertionError, match="HALO_ONLY=0"):  # allow-pytest.raises: runs with --noconftest
        vrc.check_against_reference(out, "a", **kw)
    monkeypatch.setenv("LTX_VAE_HALO_ONLY", "0")
    assert vrc.check_against_reference(out, "a", **kw)["psnr"] == 99.0
    monkeypatch.delenv("LTX_VAE_REF_RECORD")
    monkeypatch.setenv("LTX_VAE_REF", vrc.reference_path(h, w, nf, vrc.latent_md5(lat)))
    with pytest.raises(ValueError, match="decoded from latent"):  # allow-pytest.raises: runs with --noconftest
        vrc.check_against_reference(out, "a", **{**kw, "latent": lat + 1})
    bad = out.copy()
    bad[:, :h, 30:34] ^= 0xFF
    with pytest.raises(AssertionError, match="seam_col_psnr"):  # allow-pytest.raises: runs with --noconftest
        vrc.check_against_reference(bad, "b", **kw)
