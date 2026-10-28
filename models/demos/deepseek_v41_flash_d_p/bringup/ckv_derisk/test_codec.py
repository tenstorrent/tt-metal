# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests of codec.py (no ttnn import: importing ttnn opens the cluster).
Intended repo path: models/demos/deepseek_v41_flash_d_p/tests/test_ckv_codec.py.
Run: PYTHONPATH=<repo> python -m pytest -q /scratch/dnijemcevic/ckv_derisk/test_codec.py"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import codec  # noqa: E402

from models.demos.deepseek_v41_flash_d_p.reference import kernels as K  # noqa: E402

GOLD = (
    "/data/dnijemcevic/bringup/deepseek_v41_flash_d_p/golden_full/galaxy_8x4/s10240_c5120/kv_cache/layer_2.safetensors"
)


def _rows():
    g = torch.Generator().manual_seed(0)
    xs = [
        torch.randn(256, 512, generator=g),
        torch.randn(64, 512, generator=g) * 1e-3,  # tiny groups: e4m3 subnormal scales / clamp_min
        torch.zeros(4, 512),
        (torch.randint(-12, 13, (64, 512), generator=g).float() * 0.25),  # many exact ties
        torch.randn(64, 512, generator=g) * 300.0,
    ]
    x = torch.cat(xs)
    x[5, :16] = 0.0  # one all-zero group in a live row
    x[6, 3] = -0.0
    if os.path.exists(GOLD):
        from safetensors import safe_open

        with safe_open(GOLD, "pt") as h:
            x = torch.cat([x, h.get_slice("compress_kv_cache_layer_2")[:1024]])
    return x


def _bits(t):
    return t.float().contiguous().view(torch.int32)


def test_a_matches_reference_bit_exact():
    x = _rows()
    b = codec.encode_a(x)
    assert b.shape == (x.shape[0], codec.ROW_BYTES) and b.dtype == torch.uint8
    ref = K.fp4_quant_dequant(x, 16, e4m3_scale=True)
    got = codec.decode_a(b)
    assert torch.equal(got, ref) or torch.equal(_bits(got.abs()), _bits(ref.abs()))
    assert torch.equal(got.abs(), ref.abs()) and torch.equal(got, ref)


def test_a_decode_is_bf16_exact():
    got = codec.decode_a(codec.encode_a(_rows()))
    assert torch.equal(got.bfloat16().float(), got)


def test_a_roundtrip_bytes():
    b = codec.encode_a(_rows())
    assert torch.equal(codec.encode_a(codec.decode_a(b)), b)


def _bfp4_scalar(v: float, shared: int) -> float:
    """Scalar transcription of tt-metal convert_u32_to_bfp<Bfp4_b> + convert_bfp_to_u32 (is_exp_a = false)."""
    u = int(torch.tensor([v], dtype=torch.float32).view(torch.int32).item()) & 0xFFFFFFFF
    e, s, m = (u >> 23) & 0xFF, u >> 31, (u & 0x7FFFFF) | (1 << 23)
    if e == 0:
        return 0.0
    m >>= shared - e
    rv, m = m & ((1 << 21) - 1), m >> 21
    if rv > (1 << 20) or (rv == (1 << 20) and (m & 1)):
        m += 1
    m = min(m, 7)
    return (-1.0 if (s and m) else 1.0) * m / 4.0 * 2.0 ** (shared - 127)


def test_b_matches_scalar_transcription():
    x = _rows()[:96]
    got = codec.decode_b(codec.encode_b(x))
    for r in range(x.shape[0]):
        for g in range(32):
            grp = x[r, 16 * g : 16 * g + 16]
            sh = int(((grp.view(torch.int32) >> 23) & 0xFF).max())
            for j in range(16):
                assert got[r, 16 * g + j].item() == _bfp4_scalar(grp[j].item(), sh), (r, g, j)


def test_b_roundtrip_bytes():
    b = codec.encode_b(_rows())
    assert torch.equal(codec.encode_b(codec.decode_b(b)), b)


def test_b_tile_shuffle_inverse_and_layout():
    x = _rows()[:32]
    rows = codec.encode_b(x)
    tiles = codec.rows_to_tiles(rows)
    assert torch.equal(codec.tiles_to_rows(tiles), rows)
    # decode the tiles as tt-metal's bfp tile layout and compare with the row decode
    dec = codec.decode_b(rows)
    for c in range(16):
        t = tiles[c]
        for r in range(32):
            for col in range(32):
                face = (r // 16) * 2 + col // 16
                fr, fc = r % 16, col % 16
                e = int(t[face * 16 + fr])
                byte = int(t[64 + face * 128 + fr * 8 + fc // 2])
                code = (byte >> (4 * (fc % 2))) & 15
                v = (code & 7) / 4.0 * 2.0 ** (e - 127)
                v = -v if code >> 3 else v
                assert v == dec[r, 32 * c + col].item(), (c, r, col)


@pytest.mark.parametrize("enc", [codec.encode_a, codec.encode_b])
def test_row_bytes(enc):
    assert enc(torch.randn(3, 512)).shape == (3, 288)
