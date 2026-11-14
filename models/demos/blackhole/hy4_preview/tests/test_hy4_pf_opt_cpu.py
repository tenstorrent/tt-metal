# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of HY4_PF_OPT: flag parsing (default OFF), 2D-mcast geometry for every real Hy4 prefill linear, rope1 identity-span algebra."""
import pytest
import torch

from models.demos.blackhole.hy4_preview.tt.hy4_pf_opt import ALL, PfOpt, mcast_params
from models.demos.blackhole.hy4_preview.tt.hy4_rope import apply_rope_host, rope_tables


def test_default_off_and_all(monkeypatch):
    monkeypatch.delenv("HY4_PF_OPT", raising=False)
    assert not PfOpt.from_env().any
    assert PfOpt.parse("all") == PfOpt(**{t: True for t in ALL})
    assert PfOpt.parse("mm2d,a2a2").a2a2 and not PfOpt.parse("mm2d,a2a2").kc64
    with pytest.raises(AssertionError):
        PfOpt.parse("bogus")


# per-chip (K, N) of every prefill linear at M = 512 (TP8 columns)
SHAPES = {
    "q_a": (6144, 256), "kv_a": (6144, 576), "q_b": (2048, 2048), "gate": (6144, 2048), "o_proj": (2048, 6144), "idx_wq_b": (2048, 4096),
    "idx_wk": (6144, 128), "idx_wp": (6144, 32), "dense_gu": (6144, 4608), "dense_down": (2304, 6144), "shared_gu": (6144, 512), "shared_down": (256, 6144),
}


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_mcast_geometry_covers_the_matmul(name):
    K, N = SHAPES[name]
    gx, gy, pm, pn, bw, sw = mcast_params(512, K, N)
    assert gx <= 12 and gy in (4, 8)
    assert gy * pm == 512 // 32 and gx * pn == N // 32  # the grid tiles M and N exactly
    assert (K // 32) % bw == 0 and pn % sw == 0 and sw <= 4


def test_decode_rows_not_touched():
    assert mcast_params(32, 6144, 2048) is None  # M < 256: stock path


@pytest.mark.parametrize("first", [False, True])
def test_rope1_identity_span_equals_slice_rope(first):
    torch.manual_seed(0)
    n = 64
    x = torch.randn(n, 128)
    pos = torch.arange(n)
    c, s = rope_tables(pos, 64, 1e7, True)
    sl = slice(0, 64) if first else slice(64, 128)
    ref = x.clone()
    ref[:, sl] = apply_rope_host(x[:, sl], c, s, True)
    ones, zeros = torch.ones(n, 64), torch.zeros(n, 64)
    c128 = torch.cat([c, ones] if first else [ones, c], -1)
    s128 = torch.cat([s, zeros] if first else [zeros, s], -1)
    # adjacent-pair rotation of every pair of the 128 vector
    rot = torch.stack([-x[:, 1::2], x[:, 0::2]], -1).reshape(n, 128)
    got = x * c128 + rot * s128
    assert torch.allclose(got, ref, atol=1e-6)


def test_mask1_equals_slice_add_concat():
    n, T, start = 32, 96, 64
    sc = torch.randn(n, T)
    m = torch.zeros(n, n).masked_fill(torch.ones(n, n, dtype=torch.bool).triu(1), float("-inf"))
    a = torch.cat([sc[:, :start], sc[:, start:] + m], 1)
    full = torch.zeros(n, T)
    full[:, start : start + n] = m
    assert torch.equal(a, sc + full)
