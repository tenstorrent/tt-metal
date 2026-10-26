# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Op-level gate of the fused qk_post (lever L06): (B) attn_pre without cache write vs concat + nlp_create_qkv_heads_decode + 2x to_memory_config + rope_inplace
(bit-exact claim); (A) qk_norm (slice + 2 rms_norm in one program) vs ttnn.slice + ttnn.rms_norm (bit-exact claim: one core per norm running the reference LLK sequence)."""
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import attn_fused as AF

HD, QL, NH = 512, 1280, 9


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _up(md, t):
    return ttnn.from_torch(
        t.to(torch.bfloat16).contiguous(),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )


def _dn(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def traced_us(md, f, n_lo=5, n_hi=25, reps=3):
    """marginal device cost per call: (wall of n_hi calls - wall of n_lo calls) / (n_hi - n_lo), each inside one trace (min over reps)."""
    ws = {}
    for n in (n_lo, n_hi):
        f()  # compile
        ttnn.synchronize_device(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        for _ in range(n):
            f()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        best = 1e9
        for _ in range(reps):
            ttnn.synchronize_device(md)
            t0 = time.perf_counter()
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            best = min(best, time.perf_counter() - t0)
        ttnn.release_trace(md, tid)
        ws[n] = best
    return (ws[n_hi] - ws[n_lo]) / (n_hi - n_lo) * 1e6


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [pytest.param({"l1_small_size": 16384, "trace_region_size": 50000000}, id="p")], indirect=True
)
@pytest.mark.parametrize("seed", [2, 3, 4])
@pytest.mark.parametrize("T", [4, 6, 8])
@torch.no_grad()
def test_qkpost_norm(mesh_device, T, seed):
    from models.demos.blackhole.deepseek_v41_flash.tt.qk_norm import make_gamma, qk_norm

    md = mesh_device
    torch.manual_seed(seed)
    y = torch.randn(1, 1, T, QL + HD) * 3
    y[..., 7] *= 6  # an outlier channel
    qn, kn = torch.rand(QL) * 2 + 0.1, torch.rand(HD) * 2 + 0.1
    ty = _up(md, y)
    tq = _up(md, qn.reshape(1, 1, 1, QL))
    tk = _up(md, kn.reshape(1, 1, 1, HD))
    tg = ttnn.from_torch(
        make_gamma(qn.to(torch.bfloat16), kn.to(torch.bfloat16)),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )

    def old():
        a = ttnn.rms_norm(ttnn.slice(ty, [0, 0, 0, 0], [1, 1, T, QL]), weight=tq, epsilon=1e-20)
        b = ttnn.rms_norm(ttnn.slice(ty, [0, 0, 0, QL], [1, 1, T, QL + HD]), weight=tk, epsilon=1e-20)
        return a, b

    def new():
        return qk_norm(ty, tg, T)

    (a0, b0), (a1, b1) = old(), new()
    for nm, a, b in (("q", a0, a1), ("kv", b0, b1)):
        a, b = _dn(a)[:, :, :T], _dn(b)[:, :, :T]
        eq = (a == b).float().mean().item()
        print(
            f"T={T} {nm}: pcc={pcc(a, b):.7f} maxabs={float((a - b).abs().max()):.3e} (ref absmax {float(a.abs().max()):.2f}) bit-equal frac={eq:.4f}",
            flush=True,
        )
        assert torch.equal(a, b), f"{nm}: not bit-equal"
    for name, f in (("old (2 slice + 2 rms_norm)", old), ("new (qk_norm)", new)) if seed == 2 else ():
        print(f"T={T} norm {name}: traced marginal {traced_us(md, f):.2f} us/call", flush=True)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [pytest.param({"l1_small_size": 16384, "trace_region_size": 50000000}, id="p")], indirect=True
)
@pytest.mark.parametrize("T", [4, 6, 8])
@torch.no_grad()
def test_qkpost_heads(mesh_device, T):
    from models.demos.blackhole.deepseek_v41_flash.tt.attn_pre import attn_pre

    md = mesh_device
    torch.manual_seed(1)
    ang = torch.rand(T, 32) * 6.0
    C = torch.ones(1, T, 1, HD)
    S = torch.zeros(1, T, 1, HD)
    C[..., 448:], S[..., 448:] = ang.cos().repeat_interleave(2, -1).reshape(1, T, 1, 64), ang.sin().repeat_interleave(
        2, -1
    ).reshape(1, T, 1, 64)
    kv = torch.randn(1, 1, T, HD) * 2
    q = torch.randn(1, 1, T, 8 * HD) * 3
    ucfg = ttnn.create_sharded_memory_config(
        shape=(32, HD),
        core_grid=ttnn.num_cores_to_corerangeset(T, ttnn.CoreCoord(8, 8), row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tkv, tq, tC, tS = _up(md, kv), _up(md, q), _up(md, C), _up(md, S)

    def old():
        cat = ttnn.concat([tkv, tq, tkv, tkv], dim=3)
        qh, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(cat, num_heads=NH, num_kv_heads=1, memory_config=ucfg)
        ttnn.deallocate(v)
        ttnn.deallocate(k)
        qd = ttnn.to_memory_config(qh, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(qh)
        AF.rope_inplace(qd, tC, tS, T)
        return qd

    def new():
        return attn_pre(tkv, tq, None, None, None, None, tC, tS, T)

    from models.demos.blackhole.deepseek_v41_flash.tt.attn_pre import qk_heads

    ref = _dn(old())[:, :, :NH]
    for upc, upr in ((8, 4),):
        os.environ["DSV41_QKH_UPC"] = str(upc)
        os.environ["DSV41_QKH_UPR"] = str(upr)
        g = _dn(qk_heads(tkv, tq, tC, tS, T))[:, :, :NH]
        full = _dn(qk_heads(tkv, tq, tC, tS, T))
        print(
            f"T={T} UPC={upc}/{upr} qk_heads bit-exact={bool(torch.equal(ref, g))} pad rows zero={bool((full[:, :, NH:] == 0).all())}",
            flush=True,
        )
        assert torch.equal(ref, g)
        print(
            f"T={T} UPC={upc}/{upr} qk_heads: traced marginal {traced_us(md, lambda: qk_heads(tkv, tq, tC, tS, T)):.2f} us/call",
            flush=True,
        )
    print(f"T={T} old chain (5 ops): traced marginal {traced_us(md, old):.2f} us/call", flush=True)
    print(f"T={T} attn_pre (no cache): traced marginal {traced_us(md, new):.2f} us/call", flush=True)
