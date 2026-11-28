# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The CPU reference's device-format weight rounding (reference/mistral_large_4/device_rounding.py), without a device.
The ttnn cases need ttnn's host conversion and are skipped where it is unavailable."""

import struct

import pytest

torch = pytest.importorskip("torch")

from models.demos.deepseek_v3_d_p.reference.mistral_large_4.device_rounding import (  # noqa: E402
    BFP_BITS,
    RoundedExperts,
    Rounder,
    bfp_round,
    round_like_device,
    ttnn_round_trip,
)
from models.demos.deepseek_v3_d_p.reference.mistral_large_4.weights import dequant  # noqa: E402


def _weight(rows=64, cols=96, seed=0):
    """bf16-exact values whose scale changes by column (powers of two from 2^-8 to 2^7), so a group along the
    columns and a group along the rows see different shared exponents; a few exact zeros."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, cols, generator=g) * 2.0 ** torch.randint(-8, 8, (1, cols), generator=g).float()
    w[::7, ::5] = 0
    return w.to(torch.bfloat16).float()


def _scalar(vals, bits):
    """A line-by-line port of blockfloat_common.cpp convert_u32_to_bfp<Bfp*_b, false> and convert_bfp_to_u32."""
    us = [struct.unpack("<I", struct.pack("<f", v))[0] for v in vals]
    exps = [(u >> 23) & 0xFF for u in us]
    shared = max(exps)
    out = []
    for u, e in zip(us, exps):
        if e == 0:
            out.append(0.0)
            continue
        m, d = (u & 0x7FFFFF) | 0x800000, shared - e
        while d > 31:
            m, d = m >> 31, d - 31
        m >>= d
        sh = 24 - bits
        rest, q = m & ((1 << sh) - 1), m >> sh
        if rest > (1 << (sh - 1)) or (rest == (1 << (sh - 1)) and q & 1):
            q += 1
        q = min(q, (1 << bits) - 1)
        if q == 0:
            out.append(0.0)
            continue
        man, shift = q, 0
        while not man & (1 << (bits - 1)):
            man, shift = man << 1, shift + 1
        man = (man << 1) & ((1 << bits) - 1)
        word = ((u >> 31) << 31) | ((shared - shift) << 23) | (man << (23 - bits))
        out.append(struct.unpack("<f", struct.pack("<I", word))[0])
    return out


@pytest.mark.parametrize("dtype_name", ["bfloat4_b", "bfloat8_b"])
def test_emulation_matches_metal_scalar_code(dtype_name):
    bits = BFP_BITS[dtype_name]
    w = _weight(8, 64)
    # exact ties at the bfp4 width under a group max of 1.0: 1.125 rounds to even (1.0), 1.375 up (1.5); 1.9375
    # rounds past the largest mantissa and saturates
    w[0, :16] = torch.tensor([1.0, 1.125, 1.375, 1.9375, -1.125, -1.375, 0.5, 0.0] + [0.25] * 8)
    got = bfp_round(w, bits)
    want = torch.tensor([_scalar(row[i : i + 16].tolist(), bits) for row in w for i in range(0, 64, 16)]).view(8, 64)
    assert torch.equal(got, want)
    if bits == 3:
        assert got[0, :6].tolist() == [1.0, 1.0, 1.5, 1.75, -1.0, -1.5]


def test_orientation():
    w = _weight()
    kn = round_like_device(w, "bfloat4_b", "kn", impl="torch")
    nk = round_like_device(w, "bfloat4_b", "nk", impl="torch")
    assert kn.shape == nk.shape == w.shape
    assert torch.equal(kn, bfp_round(w.t().contiguous(), 3).t())  # groups of 16 along the output dim
    assert not torch.equal(kn, nk)
    assert torch.equal(round_like_device(kn, "bfloat4_b", "kn", impl="torch"), kn)  # a fixed point
    assert torch.equal(round_like_device(w, "bfloat16", "kn", impl="torch"), w)


def test_rounded_experts_use_device_layout():
    """LayerExperts-shaped stand-in: FP8 [out, in] with one power-of-two scale per 128 x 128 block."""

    class _Le:
        def __init__(self):
            g = torch.Generator().manual_seed(1)
            self.w = {(0, m): (torch.randn(128, 256, generator=g) * 50).to(torch.float8_e4m3fn) for m in ("w1", "w2")}
            self.s = {k: torch.tensor([[2.0**-9, 2.0**-4]]) for k in self.w}
            self.store = None

    class _Store:
        def __init__(self):
            self.le = _Le()

        def get(self, i, next_layer=None):
            return self.le

    store = _Store()
    layer = RoundedExperts(store, "bfloat4_b", Rounder("torch", check=False)).get(3)
    assert layer.w is store.le.w
    for m in ("w1", "w2"):
        full = dequant(store.le.w[(0, m)], store.le.s[(0, m)], torch.float32)
        want = bfp_round(full.t().contiguous(), 3).t()
        assert torch.equal(layer.weight(0, m, torch.float32), want)
        out = torch.empty(128, 256)
        assert layer.weight(0, m, torch.float32, out) is out and torch.equal(out, want)


@pytest.fixture(scope="module")
def ttnn_host():
    ttnn = pytest.importorskip("ttnn")
    try:
        ttnn_round_trip(torch.ones(32, 32), "bfloat8_b")
    except Exception as e:  # ttnn's host conversion reads the HAL
        pytest.skip(f"ttnn host conversion unavailable: {type(e).__name__}: {e}")
    return ttnn


@pytest.mark.parametrize("layout", ["kn", "nk"])
@pytest.mark.parametrize("dtype_name", ["bfloat4_b", "bfloat8_b"])
def test_round_trip_is_ttnns(ttnn_host, dtype_name, layout):
    ttnn = ttnn_host
    w = _weight()
    oriented = w.t().contiguous() if layout == "kn" else w
    host = ttnn.from_torch(oriented, dtype=getattr(ttnn, dtype_name), layout=ttnn.TILE_LAYOUT)
    want = ttnn.to_torch(host).float()
    want = want.t() if layout == "kn" else want
    got = round_like_device(w, dtype_name, layout, impl="ttnn")
    assert got.shape == w.shape
    assert torch.equal(got, want)
    assert torch.equal(round_like_device(w, dtype_name, layout, impl="torch"), want)
    # the TtExperts input dtype: bf16, the same values
    assert torch.equal(round_like_device(w.to(torch.bfloat16), dtype_name, layout, impl="ttnn"), want)


def test_rounder_checks_against_ttnn(ttnn_host):
    r = Rounder("torch")
    w = _weight()
    r(w, "bfloat4_b", "kn", tag=5)
    r(w, "bfloat4_b", "kn", tag=5)
    assert r.stats()["ttnn_checks"] == 1 and r.stats()["calls"] == 2
