# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact ternary prefill linear (tt/tprefill) against an fp32 reference of the same ternary weights."""

import os
import time

import numpy as np
import pytest
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.tprefill.linear import TernaryPrefillLinear, reference


def _weights(K, N, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.integers(-1, 2, size=(K, N)).astype(np.int8)
    s = (rng.random((N, K // 128)) * 0.02 + 0.005).astype(np.float16)
    return t, s


@pytest.mark.parametrize(
    "T, K, N",
    [(64, 1024, 256), (256, 5120, 8192), (128, 5120, 17408), (128, 17408, 5120), (128, 6144, 5120)],
    ids=["small", "k5120n8192", "k5120n17408", "k17408n5120", "k6144n5120"],
)
def test_tprefill(device, T, K, N):
    t, s = _weights(K, N)
    x = torch.randn(T, K).bfloat16()
    lin = TernaryPrefillLinear.from_host(device, t, s)
    xt = ttnn.from_torch(x.reshape(1, 1, T, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y = ttnn.to_torch(lin(xt)).float().reshape(T, N)
    if "DBG_P" in os.environ.get("BONSAI_TPRE_DEFINES", ""):
        s = np.ones_like(s)
    ref = reference(x, t, s)
    bad = ~torch.isfinite(y)
    if bad.any():
        r, c = bad.nonzero()[0].tolist()
        print(
            f"non-finite {bad.float().mean().item():.4f} first at row {r} col {c}; y[0,:4]={y[0, :4].tolist()} ref={ref[0, :4].tolist()}"
        )
    rel = ((y - ref).norm() / ref.norm()).item()
    worst = ((y - ref).abs().max() / ref.abs().max()).item()
    print(f"tprefill T={T} K={K} N={N} rel={rel:.5f} worst={worst:.5f}")
    if rel > 0.01:
        e = (y - ref).reshape(T, N // 32, 32).norm(dim=(0, 2)) / ref.reshape(T, N // 32, 32).norm(dim=(0, 2))
        badt = (e > 0.01).nonzero().flatten().tolist()
        print(f"bad tile columns ({len(badt)}): {badt[:40]}")
    assert rel < 0.01, rel


@pytest.mark.parametrize("K, N", [(5120, 17408), (17408, 5120), (5120, 8192)])
def test_tprefill_time(device, K, N):
    T, R = 2048, 5
    t, s = _weights(K, N)
    lin = TernaryPrefillLinear.from_host(device, t, s)
    xt = ttnn.from_torch(
        torch.randn(1, 1, T, K).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    lin(xt)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(R):
        lin(xt)
    ttnn.synchronize_device(device)
    dt = (time.perf_counter() - t0) / R
    w = ttnn.from_torch(torch.randn(K, N).bfloat16(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.LoFi)
    ttnn.linear(xt, w, compute_kernel_config=cfg)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(R):
        ttnn.linear(xt, w, compute_kernel_config=cfg)
    ttnn.synchronize_device(device)
    dd = (time.perf_counter() - t0) / R
    tf = 2 * T * K * N / 1e12
    print(
        f"TPRETIME T={T} K={K} N={N}: ternary {dt * 1e3:.3f} ms ({tf / dt:.0f} TFLOPS), dense bf8 LoFi ttnn.linear {dd * 1e3:.3f} ms ({tf / dd:.0f} TFLOPS)"
    )


@pytest.mark.parametrize("T, K, N", [(128, 5120, 17408), (64, 1024, 512)], ids=["mlp", "small"])
def test_tprefill_glu(device, T, K, N):
    from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import pack_glu_records, pack_records_inline

    tg, sg = _weights(K, N, seed=1)
    tu, su = _weights(K, N, seed=2)
    rg, ppr = pack_records_inline(tg, sg)
    ru, _ = pack_records_inline(tu, su)
    rec = ttnn.from_torch(
        torch.from_numpy(pack_glu_records(rg, ru, ppr).view(np.int32)),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    lin = TernaryPrefillLinear(device, rec, ppr, K, 2 * N, glu=True)
    x = torch.randn(T, K).bfloat16()
    xt = ttnn.from_torch(x.reshape(1, 1, T, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y = ttnn.to_torch(lin(xt)).float().reshape(T, N)
    ref = torch.nn.functional.silu(reference(x, tg, sg)) * reference(x, tu, su)
    rel = ((y - ref).norm() / ref.norm()).item()
    print(f"tprefill glu T={T} K={K} N={N} rel={rel:.5f}")
    assert rel < 0.02, rel


@pytest.mark.parametrize("K", [5120, 17408])
def test_prefill_rotation(device, K):
    from models.demos.blackhole.bonsai2.tt.ternary.gguf_weights import fwht_blocks
    from models.demos.blackhole.bonsai2.tt.tprefill.rotate import PrefillRotation

    T = 128
    rng = np.random.default_rng(3)
    signs = rng.choice([-1.0, 1.0], size=K).astype(np.float32)
    x = torch.randn(T, K).bfloat16()
    rot = PrefillRotation(device, signs)
    xt = ttnn.from_torch(x.reshape(1, 1, T, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y = ttnn.to_torch(rot(xt)).float().reshape(T, K)
    ref = torch.from_numpy(fwht_blocks(x.float().numpy() * signs, 1024)).float()
    rel = ((y - ref).norm() / ref.norm()).item()
    print(f"prefill rotation K={K} rel={rel:.6f}")
    assert rel < 0.005, rel


def _time(device, fn, R=5):
    fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(R):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / R * 1e3


def test_tprefill_parts_time(device):
    """The exact prefill MLP's pieces at T=2048: rotations, gate/up GLU, down."""
    from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import pack_glu_records, pack_records_inline
    from models.demos.blackhole.bonsai2.tt.tprefill.rotate import PrefillRotation

    T = 2048
    out = []
    for K in (5120, 17408):
        rot = PrefillRotation(device, np.ones(K, np.float32))
        xt = ttnn.from_torch(
            torch.randn(1, T, K).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        out.append(f"rot{K}={_time(device, lambda: rot(xt)):.3f}")
    K, N = 5120, 17408
    tg, sg = _weights(K, N, seed=1)
    rg, ppr = pack_records_inline(tg, sg)
    rec = ttnn.from_torch(
        torch.from_numpy(pack_glu_records(rg, rg, ppr).view(np.int32)),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    glu = TernaryPrefillLinear(device, rec, ppr, K, 2 * N, glu=True)
    xt = ttnn.from_torch(torch.randn(1, T, K).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out.append(f"gateup_glu={_time(device, lambda: glu(xt)):.3f}")
    t, s = _weights(N, K)
    down = TernaryPrefillLinear.from_host(device, t, s)
    ht = ttnn.from_torch(torch.randn(1, T, N).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out.append(f"down={_time(device, lambda: down(ht)):.3f}")
    print("PARTS_MS " + " ".join(out))


@pytest.mark.parametrize(
    "T, K, N, glu",
    [
        (320, 1024, 512, False),
        (640, 1024, 512, False),
        (640, 5120, 512, False),
        (640, 1024, 4096, False),
        (960, 1024, 512, False),
        (1280, 1024, 512, False),
        (1600, 1024, 512, False),
        (2240, 1024, 512, False),
        (2048, 17408, 5120, False),
        (2048, 5120, 8192, False),
        (2048, 5120, 4096, True),
    ],
    ids=["small", "m2", "m2k", "m2n", "m3", "m4", "m5", "m7", "down", "k5120n8192", "glu"],
)
def test_tprefill2(device, T, K, N, glu):
    from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import pack_glu_records, pack_records_inline
    from models.demos.blackhole.bonsai2.tt.tprefill.linear2 import TernaryPrefillLinear2
    from models.demos.blackhole.bonsai2.tt.tprefill.linear6 import TernaryPrefillLinear6, round_sig

    six = os.environ.get("BONSAI_TPRE_CLASS") == "6"  # the shared-fold linear (5-bit scales) instead
    cls = TernaryPrefillLinear6 if six else TernaryPrefillLinear2
    rs = (lambda s: round_sig(s).astype(np.float32)) if six else (lambda s: s)
    x = torch.randn(T, K).bfloat16()
    xt = ttnn.from_torch(x.reshape(1, 1, T, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    if glu:
        tg, sg = _weights(K, N, seed=1)
        tu, su = _weights(K, N, seed=2)
        rg, ppr = pack_records_inline(tg, sg)
        ru, _ = pack_records_inline(tu, su)
        rec = pack_glu_records(rg, ru, ppr)
        rec_dev = ttnn.from_torch(
            torch.from_numpy(rec.view(np.int32)),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        lin = cls.from_records(device, rec_dev, rec, ppr, K, 2 * N, glu=True)
        ref = torch.nn.functional.silu(reference(x, tg, rs(sg))) * reference(x, tu, rs(su))
    else:
        t, s = _weights(K, N)
        lin = cls.from_host(device, t, s)
        s = rs(s)
        if "T2_NOSCALE" in os.environ.get("BONSAI_T2_DEFINES", ""):
            s = np.ones_like(s)
        ref = reference(x, t, s)
    y = ttnn.to_torch(lin(xt)).float().reshape(T, -1)
    rel = ((y - ref).norm() / ref.norm()).item()
    print(f"tprefill2 T={T} K={K} N={N} glu={glu} rel={rel:.5f}")
    if rel > 0.02:
        e = (y - ref).reshape(T // 32, 32, -1, 32).norm(dim=(1, 3)) / ref.reshape(T // 32, 32, -1, 32).norm(dim=(1, 3))
        bad = (e > 0.02).nonzero().tolist()
        print(f"bad (row tile, col tile) ({len(bad)}): {bad[:20]}")
    assert rel < 0.02, rel


@pytest.mark.parametrize("K, N", [(17408, 5120), (5120, 17408), (6144, 5120)])
def test_tprefill2_time(device, K, N):
    from models.demos.blackhole.bonsai2.tt.tprefill.linear2 import TernaryPrefillLinear2

    T = 2048
    t, s = _weights(K, N)
    lin2 = TernaryPrefillLinear2.from_host(device, t, s)
    lin1 = TernaryPrefillLinear.from_host(device, t, s)
    xt = ttnn.from_torch(
        torch.randn(1, 1, T, K).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    print(
        f"T2TIME K={K} N={N}: 2D {_time(device, lambda: lin2(xt)):.3f} ms, 1D {_time(device, lambda: lin1(xt)):.3f} ms"
    )
    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
        ttnn.ReadDeviceProfiler(device)


def test_tprefill2_model_shapes_time(device):
    """2D vs 1D on the model's prefill shapes at T=2048 (GLU gate/up as pair records)."""
    from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import pack_glu_records, pack_records_inline
    from models.demos.blackhole.bonsai2.tt.tprefill.linear2 import TernaryPrefillLinear2

    T = 2048
    out = []
    K, N = 5120, 17408
    tg, sg = _weights(K, N, seed=1)
    rg, ppr = pack_records_inline(tg, sg)
    rec = pack_glu_records(rg, rg, ppr)
    rec_dev = ttnn.from_torch(
        torch.from_numpy(rec.view(np.int32)),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    g2 = TernaryPrefillLinear2.from_records(device, rec_dev, rec, ppr, K, 2 * N, glu=True)
    g1 = TernaryPrefillLinear(device, rec_dev, ppr, K, 2 * N, glu=True)
    xt = ttnn.from_torch(
        torch.randn(1, 1, T, K).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    out.append(f"gateup_glu 2D={_time(device, lambda: g2(xt)):.3f} 1D={_time(device, lambda: g1(xt)):.3f}")
    for N in (16384, 12288, 2048):
        t, s = _weights(K, N)
        l2 = TernaryPrefillLinear2.from_host(device, t, s)
        l1 = TernaryPrefillLinear.from_host(device, t, s)
        out.append(f"k5120n{N} 2D={_time(device, lambda: l2(xt)):.3f} 1D={_time(device, lambda: l1(xt)):.3f}")
    print("SHAPES_MS " + " | ".join(out))


@pytest.mark.parametrize("add, want_n", [(False, False), (False, True), (True, False), (True, True)])
def test_prefill_norm_rot(device, add, want_n):
    """norm_rot.PrefillNormRot against torch: h = a (+ b), n = h / rms(h) * w, x' = fwht_blocks(bf16(n) * d)."""
    from models.demos.blackhole.bonsai2.tt.ternary.gguf_weights import fwht_blocks
    from models.demos.blackhole.bonsai2.tt.tprefill.norm_rot import PrefillNormRot

    T, K = 2048, 5120
    rng = np.random.default_rng(5)
    signs = rng.choice([-1.0, 1.0], size=K).astype(np.float32)
    w = (1.0 + 0.3 * torch.randn(K)).bfloat16().float()
    a = (torch.randn(T, K) * 3).bfloat16()
    b = (torch.randn(T, K) * 2).bfloat16() if add else None
    tt = lambda v: ttnn.from_torch(v.reshape(1, 1, T, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    op = PrefillNormRot(device, signs, w, 1e-6)
    h, n, xr = op(tt(a), tt(b) if add else None, want_n=want_n)
    # the residual as the model's ttnn.add produces it (its packer rounding), and the bit-exact match the fused op gives
    href = ttnn.to_torch(ttnn.add(tt(a), tt(b))).float().reshape(T, K) if add else a.float()
    if add:
        rn = (href - (a.float() + b.float()).bfloat16().float()).abs().max().item()
        print(f"ttnn.add vs round-to-nearest max diff {rn}")
    nref = href / torch.sqrt((href * href).mean(-1, keepdim=True) + 1e-6) * w
    xref = torch.from_numpy(fwht_blocks((nref * torch.from_numpy(signs)).bfloat16().float().numpy(), 1024)).float()
    rel = lambda y, r: ((ttnn.to_torch(y).float().reshape(T, K) - r).norm() / r.norm()).item()
    msg = f"NORMROT add={add} n={want_n} xr rel {rel(xr, xref):.5f}"
    if add:
        msg += f" h rel {rel(h, href):.6f}"
    if want_n:
        msg += f" n rel {rel(n, nref):.5f}"
    print(msg)
    assert rel(xr, xref) < 0.01
    if add:
        assert rel(h, href) < 1e-6, "h must equal ttnn.add bit for bit"
    if want_n:
        assert rel(n, nref) < 0.01


@pytest.mark.parametrize("N, na, ha, hb", [(12288, 192, 8, 0), (2048, 32, 8, 8)], ids=["q_gate", "k_v"])
def test_tprefill2_split(device, N, na, ha, hb):
    """split outputs (head-major / token-major) are the unsplit output's columns, relaid."""
    from models.demos.blackhole.bonsai2.tt.tprefill.linear2 import TernaryPrefillLinear2

    T, K = 2048, 5120
    t, s = _weights(K, N, seed=2)
    lin = TernaryPrefillLinear2.from_host(device, t, s)
    xt = ttnn.from_torch(
        torch.randn(1, 1, T, K).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    full = ttnn.to_torch(lin(xt)).float().reshape(T, N)
    a, b = lin(xt, split=(na, ha, hb))

    def relay(cols, h):
        return cols.reshape(T, -1, h * 32).permute(1, 0, 2) if h else cols

    ea = (ttnn.to_torch(a).float().reshape(relay(full[:, : na * 32], ha).shape) - relay(full[:, : na * 32], ha)).abs()
    eb = (ttnn.to_torch(b).float().reshape(relay(full[:, na * 32 :], hb).shape) - relay(full[:, na * 32 :], hb)).abs()
    print(f"SPLIT N={N} max diff a {ea.max().item()} b {eb.max().item()} shapes {list(a.shape)} {list(b.shape)}")
    assert ea.max().item() == 0 and eb.max().item() == 0


def test_prefill_norm_rot_time(device):
    from models.demos.blackhole.bonsai2.tt.tprefill.norm_rot import PrefillNormRot

    T, K = 2048, 5120
    signs = np.random.default_rng(5).choice([-1.0, 1.0], size=K).astype(np.float32)
    op = PrefillNormRot(device, signs, torch.ones(K), 1e-6)
    tt = lambda: ttnn.from_torch(
        torch.randn(1, 1, T, K).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    a, b = tt(), tt()
    print(
        f"NROT_MS attn {_time(device, lambda: op(a, want_n=True)):.3f} ffn {_time(device, lambda: op(a, b)):.3f} "
        f"xr-only {_time(device, lambda: op(a)):.3f}"
    )


@pytest.mark.parametrize("H", [24, 4])
def test_qk_norm_rope(device, H):
    """qk_rope.QKNormRope against torch: rms_norm(x) * w over each head's 256, then partial RoPE on the first 64."""
    from models.demos.blackhole.bonsai2.tt.tprefill.qk_rope import QKNormRope

    T, D, R = 2048, 256, 64
    x = (torch.randn(1, H, T, D) * 2).bfloat16()
    w = (1.0 + 0.3 * torch.randn(D)).bfloat16().float()
    ang = torch.arange(T).float()[:, None] * (1.0 / 10000 ** (torch.arange(0, R, 2).float() / R))[None]
    emb = torch.cat([ang, ang], -1)
    cos, sin = emb.cos().bfloat16(), emb.sin().bfloat16()
    xf = x.float()
    n = xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + 1e-6) * w
    r, p = n[..., :R], n[..., R:]
    rh = torch.cat([-r[..., R // 2 :], r[..., : R // 2]], -1)
    ref = torch.cat([r * cos.float() + rh * sin.float(), p], -1)
    tt = lambda v: ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    op = QKNormRope(device, w, 1e-6)
    y = ttnn.to_torch(op(tt(x), tt(cos.reshape(1, T, R)), tt(sin.reshape(1, T, R)))).float()
    rel = ((y - ref).norm() / ref.norm()).item()
    xt = tt(x)
    print(
        f"QKR H={H} rel {rel:.5f} time {_time(device, lambda: op(xt, tt(cos.reshape(1, T, R)), tt(sin.reshape(1, T, R)))):.3f} ms"
    )
    assert rel < 0.01, rel


def test_tprefill2_residual(device):
    """residual= adds a tensor to the output in the output stage."""
    from models.demos.blackhole.bonsai2.tt.tprefill.linear import reference
    from models.demos.blackhole.bonsai2.tt.tprefill.linear2 import TernaryPrefillLinear2

    T, K, N = 2048, 17408, 5120
    t, s = _weights(K, N, seed=3)
    lin = TernaryPrefillLinear2.from_host(device, t, s)
    x = torch.randn(T, K).bfloat16()
    h = (torch.randn(T, N) * 4).bfloat16()
    tt = lambda v, k: ttnn.from_torch(
        v.reshape(1, 1, T, k), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    y = ttnn.to_torch(lin(tt(x, K), residual=tt(h, N))).float().reshape(T, N)
    ref = reference(x, t, s) + h.float()
    rel = ((y - ref).norm() / ref.norm()).item()
    yr = reference(x, t, s)
    r2 = lambda a, b: ((a - b).norm() / b.norm()).item()
    print(
        f"RESID rel {rel:.5f} | y-h vs y_ref {r2(y - h.float(), yr):.5f} | y vs h {r2(y, h.float()):.5f} | y vs y_ref {r2(y, yr):.5f}"
    )
    e = (y - ref).reshape(T // 32, 32, N // 32, 32).norm(dim=(1, 3)) / ref.reshape(T // 32, 32, N // 32, 32).norm(
        dim=(1, 3)
    )
    print("bad tiles", (e > 0.05).sum().item(), "of", e.numel(), "first", (e > 0.05).nonzero()[:8].tolist())
    assert rel < 0.02, rel


def test_prefill_rotation_gate(device):
    """PrefillRotation(x_hm, gate=g) == PrefillRotation(concat_heads(x_hm) * sigmoid(g))."""
    from models.demos.blackhole.bonsai2.tt.ternary.gguf_weights import fwht_blocks
    from models.demos.blackhole.bonsai2.tt.tprefill.rotate import PrefillRotation

    T, H, D = 2048, 24, 256
    K = H * D
    signs = np.random.default_rng(4).choice([-1.0, 1.0], size=K).astype(np.float32)
    att = torch.randn(1, H, T, D).bfloat16()
    g = (torch.randn(1, T, K) * 2).bfloat16()
    rot = PrefillRotation(device, signs)
    tt = lambda v: ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y = ttnn.to_torch(rot(tt(att), gate=tt(g))).float().reshape(T, K)
    xg = att.float().permute(0, 2, 1, 3).reshape(T, K) * torch.sigmoid(g.float().reshape(T, K))
    ref = torch.from_numpy(fwht_blocks(xg.bfloat16().float().numpy() * signs, 1024)).float()
    rel = ((y - ref).norm() / ref.norm()).item()
    print(f"ROTGATE rel {rel:.5f}")
    assert rel < 0.01, rel
