# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the ``GLM53_DSA_DECODE_STATE`` block (decode ledger item B5; no device, torch only):

* the fused state program's torch reference (``DsaDecodeState.golden``) against the composite path's 29-op math
  (``DsaDecodeState.composite_golden``: bf16 norm weights, fp32 compute, the same rounding points): with
  bf16-valued norm weights (the checkpoint's are bf16) q_resid, the appended latent row, both ring rows and the
  pooled key agree BIT FOR BIT and no other cache row moves; the only difference the two references can show is
  the weight precision (fp32 program constants vs bf16 device tensors), demonstrated with non-bf16 weights;
* the fold algebra (``fold_absorb_weights``): ``q_resid @ W_qb1`` equals the composite ``q_b`` matmul + per-head
  absorb bmm, and ``o_lat_packed @ W_o2`` the per-head absorb-o bmm + concat heads + ``o_proj``, in fp32 on random
  data (``_absorb_q`` / ``_o_lat_to_heads`` / ``_o_proj`` layouts);
* the free head-major views (``[1,1,B,H*L] -> [1,H,B,L]`` and the index query ``[1,1,B,Hi*D] -> [1,Hi,B,D]``, B <=
  32): the packed tile sequence IS the head-major tile sequence (tile-order emulation);
* the knob plumbing (``decode_state_default`` / ``dsa_folds`` / the modes).

    pytest models/demos/glm53_flash/tests/test_dsa_decode_state_host.py --noconftest -q
"""

from __future__ import annotations

import pytest
import torch

from models.demos.glm53_flash.custom_ops.dsa_decode_state.op import DsaDecodeState
from models.demos.glm53_flash.tt.dsa import (
    DSA_DECODE_STATE_MODES,
    DSA_FOLD_MODES,
    decode_state_default,
    dsa_folds,
    fold_absorb_weights,
)

Lq, L, D, P = 1536, 512, 128, 4
Hi = 32
EPS_RMS, EPS_LN = 1e-5, 1e-6
OFF = (0, Lq, Lq + L, Lq + L + D)  # stem column offsets: q_a | kv_a | k | g | w
WS = Lq + L + 2 * D + Hi
TILE = 32


def _bf(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).float()


def _case(
    B: int, seed: int, *, blocks_per_seq: int = 4, pool_blocks: int = 2, permute_pages: bool = False, bf16_weights=True
):
    """Random state like ``tests/custom_ops/test_dsa_decode_state.py::make_case``; the norm weights bf16-valued
    (the checkpoint's) unless ``bf16_weights`` is False."""
    gen = torch.Generator().manual_seed(seed)
    r = lambda *shape: torch.randn(*shape, generator=gen, dtype=torch.float32)
    n_blocks = B * blocks_per_seq
    pt = torch.arange(n_blocks, dtype=torch.int32).reshape(B, blocks_per_seq)
    if permute_pages:
        pt = pt.flatten()[torch.randperm(n_blocks, generator=gen)].reshape(B, blocks_per_seq)
    w = (lambda t: _bf(t)) if bf16_weights else (lambda t: t)
    x = dict(
        stem=_bf(r(B, WS)),
        latent=_bf(0.5 * r(n_blocks, 1, 32, L)),
        pt=pt,
        positions=torch.randint(0, blocks_per_seq * 32, (B,), generator=gen, dtype=torch.int32),
        ring_k=torch.zeros(B, 1, 32, D),
        ring_g=torch.zeros(B, 1, 32, D),
        pool=_bf(0.5 * r(B * pool_blocks, 1, 32, D)),
        wq=w(r(Lq) * 0.2 + 1.0),
        wkva=w(r(L) * 0.2 + 1.0),
        wk=w(r(D) * 0.2 + 1.0),
        bk=w(r(D) * 0.1),
        ape=w(r(P, D) * 0.5),
        pool_blocks=pool_blocks,
    )
    x["ring_k"][:, 0, :P] = _bf(r(B, P, D))
    x["ring_g"][:, 0, :P] = _bf(r(B, P, D))
    return x


def _run(fn, x):
    return fn(
        x["stem"],
        x["latent"],
        x["pt"],
        x["positions"],
        x["ring_k"],
        x["ring_g"],
        x["pool"],
        x["wq"],
        x["wkva"],
        x["wk"],
        x["bk"],
        x["ape"],
        Lq=Lq,
        L=L,
        D=D,
        P=P,
        pool_blocks=x["pool_blocks"],
        offsets=OFF,
        eps_rms=EPS_RMS,
        eps_ln=EPS_LN,
    )


def _ulps_bf16(a: torch.Tensor, b: torch.Tensor) -> int:
    """Max distance in bf16 ulps between two bf16-valued fp32 tensors (0 = bit-identical)."""
    ia = a.to(torch.bfloat16).view(torch.int16).to(torch.int32)
    ib = b.to(torch.bfloat16).view(torch.int16).to(torch.int32)
    return int((ia - ib).abs().max()) if a.numel() else 0


# ------------------------------------------------------------------- the state math: fused reference vs composite
@pytest.mark.parametrize(
    "B, permute, seed", [(1, False, 1), (4, False, 2), (4, True, 3), (32, True, 4), (40, False, 5)]
)
def test_state_golden_vs_composite_bit_identical(B, permute, seed):
    """With bf16-valued norm weights the program's reference equals the composite's 29-op math bit for bit: the
    same bf16 rounding points (the normed rows, the pooled key), the rings written raw / normed, the pool from the
    fp32-typecast ring rows. Every other cache row is untouched in both."""
    x = _case(B, seed, permute_pages=permute)
    q_f, lat_f, rk_f, rg_f, pool_f = _run(DsaDecodeState.golden, x)
    q_c, lat_c, rk_c, rg_c, pool_c = _run(DsaDecodeState.composite_golden, x)
    assert torch.equal(q_f, q_c), f"q_resid differs by {_ulps_bf16(q_f, q_c)} ulps"
    assert torch.equal(lat_f, lat_c), f"latent differs by {_ulps_bf16(lat_f, lat_c)} ulps"
    assert torch.equal(rk_f, rk_c), f"ring_k differs by {_ulps_bf16(rk_f, rk_c)} ulps"
    assert torch.equal(rg_f, rg_c), "ring_g (a raw move) differs"
    assert torch.equal(pool_f, pool_c), f"pools differ by {_ulps_bf16(pool_f, pool_c)} ulps"
    # exactly the written rows moved (both references), everything else is the input
    touched_lat = {(int(x["pt"][b, int(a) // 32]), int(a) % 32) for b, a in enumerate(x["positions"])}
    for page in range(lat_f.shape[0]):
        for row in range(32):
            if (page, row) not in touched_lat:
                assert torch.equal(lat_f[page, 0, row], x["latent"][page, 0, row])
    for b in range(B):
        a = int(x["positions"][b])
        for row in range(32):
            if row != a % P:
                assert torch.equal(rk_f[b, 0, row], x["ring_k"][b, 0, row])
                assert torch.equal(rg_f[b, 0, row], x["ring_g"][b, 0, row])
        assert torch.equal(rg_f[b, 0, a % P], x["stem"][b, OFF[3] : OFF[3] + D])  # the raw gate row
        p = a // P
        blk = b * x["pool_blocks"] + p // 32
        for row in range(32):
            if row != p % 32:
                assert torch.equal(pool_f[blk, 0, row], x["pool"][blk, 0, row])
    print(f"B={B} permute={permute}: fused reference == composite reference (bit-identical)")


def test_state_golden_vs_composite_weight_precision_only():
    """With NON-bf16 weights the two references differ (the program keeps fp32 constants, the composite bf16
    device tensors): the only difference between the two formulations, and it vanishes for the bf16 checkpoint
    norms (previous test). Also the composite reference with pre-rounded weights equals the fused one."""
    x = _case(8, 11, bf16_weights=False)
    q_f, lat_f, rk_f, rg_f, pool_f = _run(DsaDecodeState.golden, x)
    q_c, lat_c, rk_c, rg_c, pool_c = _run(DsaDecodeState.composite_golden, x)
    assert not torch.equal(q_f, q_c) or not torch.equal(lat_f, lat_c), "fp32 weights left no trace"
    # a weight-rounding effect only: one bf16 ulp on the normed rows (the k row carries the bias rounding too,
    # which near zero is many ulps but tiny in absolute terms)
    assert _ulps_bf16(q_f, q_c) <= 1 and _ulps_bf16(lat_f, lat_c) <= 1
    assert (rk_f - rk_c).abs().max() < 2e-2 and torch.equal(rg_f, rg_c)
    xr = dict(x, wq=_bf(x["wq"]), wkva=_bf(x["wkva"]), wk=_bf(x["wk"]), bk=_bf(x["bk"]))
    q_r, lat_r, rk_r, rg_r, pool_r = _run(DsaDecodeState.golden, xr)
    q_c2, lat_c2, rk_c2, rg_c2, pool_c2 = _run(DsaDecodeState.composite_golden, x)
    for a, b in ((q_r, q_c2), (lat_r, lat_c2), (rk_r, rk_c2), (rg_r, rg_c2), (pool_r, pool_c2)):
        assert torch.equal(a, b)


def test_state_multi_step_rings_and_pools():
    """Six steps on one evolving state (positions crossing ring slots and pool rows, pools spanning two pool
    blocks): the two references stay bit-identical step after step (the composite reads the rings it wrote)."""
    B = 4
    x = _case(B, 21, pool_blocks=8)
    x["positions"] = torch.tensor([5, 40, 71, 990], dtype=torch.int32)  # 990 // 4 = 247 -> pool block 7
    x["pt"] = torch.arange(B * 32, dtype=torch.int32).reshape(B, 32)
    x["latent"] = _bf(0.5 * torch.randn(B * 32, 1, 32, L, generator=torch.Generator().manual_seed(5)))
    f, c = dict(x), dict(x)
    gen = torch.Generator().manual_seed(99)
    for step in range(6):
        stem = _bf(torch.randn(B, WS, generator=gen))
        f["stem"] = c["stem"] = stem
        q_f, f["latent"], f["ring_k"], f["ring_g"], f["pool"] = _run(DsaDecodeState.golden, f)
        q_c, c["latent"], c["ring_k"], c["ring_g"], c["pool"] = _run(DsaDecodeState.composite_golden, c)
        for name, a, b in (
            ("q", q_f, q_c),
            ("latent", f["latent"], c["latent"]),
            ("ring_k", f["ring_k"], c["ring_k"]),
            ("ring_g", f["ring_g"], c["ring_g"]),
            ("pool", f["pool"], c["pool"]),
        ):
            assert torch.equal(a, b), f"step {step}: {name} differs by {_ulps_bf16(a, b)} ulps"
        f["positions"] = c["positions"] = f["positions"] + 1


# ------------------------------------------------------------------------------------------ the fold algebra
def _composite_q_lat(q_resid, w_qb_t, wkb1):
    """``_q`` + ``_absorb_q``: q = q_resid @ W_qb^T -> [T,H,Dq] -> per head q[h] @ wkb1[h] -> [H,T,L]."""
    T = q_resid.shape[0]
    H, Dq, L = wkb1.shape
    q = (q_resid @ w_qb_t).reshape(T, H, Dq).transpose(0, 1)  # [H,T,Dq]
    return torch.bmm(q, wkb1)  # [H,T,L]


def _composite_y(o_lat, wkb2, w_o_t):
    """``_o_lat_to_heads`` + ``o_proj``: o[h] = o_lat[h] @ wkb2[h]^T -> concat heads [T,H*Dv] -> @ W_o^T."""
    H, T, L = o_lat.shape
    o = torch.bmm(o_lat, wkb2.transpose(1, 2))  # [H,T,Dv]
    return o.transpose(0, 1).reshape(T, -1) @ w_o_t  # [T,hidden]


@pytest.mark.parametrize("T", [1, 4, 32])
def test_fold_algebra_fp32(T):
    """``q_resid @ W_qb1`` == q_b + absorb-q bmm (head-major packed [T, H*L] <-> [H,T,L]); ``o_packed @ W_o2`` ==
    absorb-o bmm + concat heads + o_proj; fp32, random data (double-precision reference)."""
    gen = torch.Generator().manual_seed(T)
    H, Dq, Dv, Lh, Lqh, hidden = 8, 32, 32, 64, 96, 128  # reduced dims (the identities are shape-generic)
    r = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float64)
    w_qb_t, wkb1, wkb2, w_o_t = r(Lqh, H * Dq), r(H, Dq, Lh), r(H, Dv, Lh), r(H * Dv, hidden)
    q_resid, o_lat = r(T, Lqh), r(H, T, Lh)
    w_qb1, w_o2 = fold_absorb_weights(w_qb_t, wkb1, wkb2, w_o_t)  # fp32 outputs
    assert w_qb1.shape == (Lqh, H * Lh) and w_o2.shape == (H * Lh, hidden) and w_qb1.dtype == torch.float32
    q_lat_fold = (q_resid.float() @ w_qb1).reshape(T, H, Lh).transpose(0, 1)  # packed head-major -> [H,T,L]
    q_lat_ref = _composite_q_lat(q_resid, w_qb_t, wkb1)
    torch.testing.assert_close(q_lat_fold.double(), q_lat_ref, rtol=1e-5, atol=1e-4)
    o_packed = o_lat.transpose(0, 1).reshape(T, H * Lh).float()  # [1,H,T,L] -> the packed [T, H*L] rows
    y_fold = o_packed @ w_o2
    y_ref = _composite_y(o_lat, wkb2, w_o_t)
    torch.testing.assert_close(y_fold.double(), y_ref, rtol=1e-5, atol=1e-3)


def test_fold_weights_model_shapes_and_sharding():
    """The model's shapes: W_qb1 [1536, 64*512], W_o2 [64*512, 4096]; the head-major column / row order is what the
    ``cols=-1`` / ``cols=-2`` mesh sharding splits into contiguous head groups (column c = heads [16c, 16c+16))."""
    gen = torch.Generator().manual_seed(7)
    H, Dq, Dv, Lh, hidden = 64, 256, 256, 512, 4096
    r = lambda *s: torch.randn(*s, generator=gen)
    w_qb_t, wkb1, wkb2, w_o_t = r(Lq, H * Dq), r(H, Dq, Lh), r(H, Dv, Lh), r(H * Dv, hidden)
    w_qb1, w_o2 = fold_absorb_weights(w_qb_t, wkb1, wkb2, w_o_t)
    assert w_qb1.shape == (Lq, H * Lh) and w_o2.shape == (H * Lh, hidden)
    # column c's shard of W_qb1 is the fold of heads [16c, 16c+16) alone
    c, Hl = 2, H // 4
    shard = w_qb1[:, c * Hl * Lh : (c + 1) * Hl * Lh]
    own = fold_absorb_weights(
        w_qb_t.reshape(Lq, H, Dq)[:, c * Hl : (c + 1) * Hl].reshape(Lq, Hl * Dq),
        wkb1[c * Hl : (c + 1) * Hl],
        wkb2[c * Hl : (c + 1) * Hl],
        w_o_t.reshape(H, Dv, hidden)[c * Hl : (c + 1) * Hl].reshape(Hl * Dv, hidden),
    )
    assert torch.equal(shard, own[0]) and torch.equal(w_o2[c * Hl * Lh : (c + 1) * Hl * Lh], own[1])


# ---------------------------------------------------------------------------------------- the free views
def _tile_sequence(shape: tuple[int, ...]) -> list[tuple]:
    """The (batch index..., tile row, tile col) of every tile of a TILE-layout tensor in buffer order, the last two
    dims padded to 32."""
    *outer, R, C = shape
    Rt, Ct = -(-R // TILE), -(-C // TILE)
    seq = []
    for o in torch.cartesian_prod(*[torch.arange(n) for n in outer]).reshape(-1, len(outer)).tolist():
        for i in range(Rt):
            for j in range(Ct):
                seq.append((tuple(o), i, j))
    return seq


@pytest.mark.parametrize("B", [1, 4, 8, 32])
@pytest.mark.parametrize("H, W", [(16, 512), (64, 512), (32, 128)])
def test_free_head_major_view(B, H, W):
    """``[1,1,B,H*W] -> [1,H,B,W]`` (B <= 32) is a pure relabelling of the tile buffer: tile j of the packed row
    holds users 0..B-1 x columns [32j, 32j+32) = head j // (W/32), head columns [(j % (W/32))*32, ...), which is
    tile (h, 0, j % (W/32)) of the head-major tensor at the same buffer index. Hence the view (no data movement)
    used for q_lat (H local heads, W = L) and the index query (H = Hi, W = D)."""
    packed = _tile_sequence((1, 1, B, H * W))
    major = _tile_sequence((1, H, B, W))
    assert len(packed) == len(major) == H * (W // TILE)
    for k, ((_, i, j), (o, i2, j2)) in enumerate(zip(packed, major)):
        assert i == 0 and i2 == 0
        assert o[1] == j // (W // TILE) and j2 == j % (W // TILE), (k, j, o, j2)
    # and the logical map the view implies: major[0, h, b, :] == packed[0, 0, b, h*W : (h+1)*W]
    gen = torch.Generator().manual_seed(B)
    x = torch.randn(1, 1, B, H * W, generator=gen)
    major_t = x.reshape(B, H, W).transpose(0, 1)  # [H,B,W]
    for h in (0, H // 2, H - 1):
        for b in (0, B - 1):
            assert torch.equal(major_t[h, b], x[0, 0, b, h * W : (h + 1) * W])


# ---------------------------------------------------------------------------------------------- the knob
def test_knob_plumbing(monkeypatch):
    monkeypatch.delenv("GLM53_DSA_DECODE_STATE", raising=False)
    assert decode_state_default() == "0", "the default must be today's composite path"
    for mode in DSA_DECODE_STATE_MODES:
        monkeypatch.setenv("GLM53_DSA_DECODE_STATE", mode)
        assert decode_state_default() == mode
    monkeypatch.setenv("GLM53_DSA_DECODE_STATE", "yes")
    with pytest.raises(ValueError):  # allow-pytest.raises: --noconftest host test, no fixture
        decode_state_default()
    monkeypatch.delenv("GLM53_DSA_FOLD", raising=False)
    assert dsa_folds() == (True, True), "GLM53_DSA_FOLD keeps its default (both)"
    assert dsa_folds("q") == (True, False) and dsa_folds("o") == (False, True) and dsa_folds("none") == (False, False)
    assert set(DSA_FOLD_MODES) == {"both", "q", "o", "none"}
    with pytest.raises(ValueError):  # allow-pytest.raises: --noconftest host test, no fixture
        dsa_folds("zz")
    assert dsa_folds("zz", strict=False) == (False, False)  # the dense fused block's lenient reading
    monkeypatch.setenv("GLM53_DSA_FOLD", "o")
    assert dsa_folds() == (False, True)
