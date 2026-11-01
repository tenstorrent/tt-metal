# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused prefill attention glue (tt/fuse_attn.py) vs the unfused ttnn chains of tt/prefill_attention.py, on one chip, U=2 users x C=512 rows, 8 heads.
Prints bit equality and PCC.  FA_T_TRACE=<ref|fused> captures a trace of the chain (qprep + kv + post, FA_T_REPS times) and replays it (profiler: per-op kernel time of ONE replay).
FA_T_REAL=<file.pt> loads real activations ({'q','kv','o'} tensors saved from a prefill window) instead of random data."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import fuse_attn as FA
from models.demos.blackhole.deepseek_v41_flash.tt import pf_tune

U, C, NH, D = 2, 512, 8, 512


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def make(md):
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t, layout=ttnn.TILE_LAYOUT: ttnn.from_torch(
        t.to(torch.bfloat16).contiguous(),
        device=md,
        dtype=ttnn.bfloat16,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    return up


def tables(up, per_user):
    nu = U if per_user else 1
    pos = torch.arange(nu * C).float().view(nu, 1, C, 1)
    inv = 1.0 / (10000.0 ** (torch.arange(32).float() / 32))
    ang = pos * inv.view(1, 1, 1, 32)
    cos, sin = ang.cos().repeat_interleave(2, -1), ang.sin().repeat_interleave(2, -1)
    c = torch.ones(nu, 1, C, D)
    s = torch.zeros(nu, 1, C, D)
    c[..., 448:], s[..., 448:] = cos, sin
    return up(c), up(s), up(-s)


def down(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def real(key, shape, scale=1.0):
    p = os.environ.get("FA_T_REAL")
    if p:
        return torch.load(p)[key].float().reshape(shape)
    return torch.randn(*shape) * scale


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [pytest.param({"l1_small_size": 16384, "trace_region_size": 200000000}, id="p")], indirect=True
)
@pytest.mark.parametrize("per_user", [True, False])
@pytest.mark.parametrize("mode", [0, 1])
@torch.no_grad()
def test_fuse_attn_kernels(mesh_device, per_user, mode):
    os.environ["DSV41_FA_ROPE_MODE"] = str(mode)
    print(f"ROPE MODE {mode} per_user {per_user}", flush=True)
    md = mesh_device
    up = make(md)
    P = pf_tune.p64(md)
    ckc = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    tabs = tables(up, per_user)
    q = up(real("q", [1, 1, U * C, NH * D], 3.0))
    kvn = up(real("kv", [1, 1, U * C, D], 3.0))
    o_t = real("o", [U, 32, C, D], 2.0)
    o_list = [up(o_t[u : u + 1], ttnn.ROW_MAJOR_LAYOUT) for u in range(U)]
    # ---- q, tile output
    qr = ttnn.reshape(q, [U, 1, C, NH * D])
    qh, _, _ = ttnn.experimental.nlp_create_qkv_heads(qr, num_heads=NH, num_kv_heads=0, transpose_k_heads=False)
    ref = down(pf_tune.rope_pe(qh, tabs[0], tabs[1], P, ckc))
    got = down(FA.qprep(q, tabs, P, U, C, NH, sparse=False))
    print(
        f"qprep tile: equal {torch.equal(ref, got)} max|d| {(ref - got).abs().max():.4g} pcc {pcc(ref, got):.7f}",
        flush=True,
    )
    assert pcc(ref, got) > 0.9999
    # ---- q, sparse padded row-major buffers
    bufs = FA.qprep(q, tabs, P, U, C, NH, sparse=True)
    for u in range(U):
        g = down(bufs[u])
        w = ref[u : u + 1]
        print(
            f"qprep sparse u{u}: heads<8 equal {torch.equal(g[:, :NH], w)} zero heads ok {bool((g[:, NH:] == 0).all())} pcc {pcc(g[:, :NH], w):.7f}",
            flush=True,
        )
        assert pcc(g[:, :NH], w) > 0.9999 and bool((g[:, NH:] == 0).all())
    # ---- kv
    kvr = ttnn.reshape(kvn, [U, 1, C, D])
    ref = down(pf_tune.rope_pe(kvr, tabs[0], tabs[1], P, ckc))
    got = down(FA.kv_rope(kvn, tabs, P, U, C))
    print(
        f"kv rope: equal {torch.equal(ref, got)} max|d| {(ref - got).abs().max():.4g} pcc {pcc(ref, got):.7f}",
        flush=True,
    )
    assert pcc(ref, got) > 0.9999
    # ---- post
    outs = []
    for u in range(U):
        ou = ttnn.to_layout(ttnn.slice(o_list[u], [0, 0, 0, 0], [1, NH, C, D]), ttnn.TILE_LAYOUT)
        outs.append(ou)
    o = ttnn.concat(outs, dim=0)
    o = pf_tune.rope_pe(o, tabs[0], tabs[2], P, ckc)
    cc = ttnn.experimental.nlp_concat_heads(o)
    zero = ttnn.zeros([U, 1, C, D], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md)
    cc = ttnn.reshape(ttnn.concat([zero, cc], dim=3), [1, 1, U * C, (NH + 1) * D])
    ref = down(cc)
    got = down(FA.post(o_list, tabs, P, U, C, NH))
    print(
        f"post: equal {torch.equal(ref, got)} max|d| {(ref - got).abs().max():.4g} pcc {pcc(ref, got):.7f}", flush=True
    )
    assert pcc(ref, got) > 0.9999


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [pytest.param({"l1_small_size": 16384, "trace_region_size": 200000000}, id="p")], indirect=True
)
@torch.no_grad()
def test_fuse_attn_trace(mesh_device):
    """traced replay of the glue chain, mode ref (the unfused ops) or fused; run under the profiler, read the trace rows of the report."""
    mode = os.environ.get("FA_T_TRACE", "fused")
    reps = int(os.environ.get("FA_T_REPS", "3"))
    md = mesh_device
    up = make(md)
    P = pf_tune.p64(md)
    ckc = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    tabs = tables(up, True)
    q = up(torch.randn(1, 1, U * C, NH * D) * 3)
    kvn = up(torch.randn(1, 1, U * C, D) * 3)
    o_t = torch.randn(U, 32, C, D) * 2
    o_list = [up(o_t[u : u + 1], ttnn.ROW_MAJOR_LAYOUT) for u in range(U)]
    zero_heads = ttnn.zeros([1, 32 - NH, C, D], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=md)
    zero = ttnn.zeros([U, 1, C, D], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md)

    def chain():
        if mode == "ref":
            qr = ttnn.reshape(q, [U, 1, C, NH * D])
            qh, _, _ = ttnn.experimental.nlp_create_qkv_heads(qr, num_heads=NH, num_kv_heads=0, transpose_k_heads=False)
            qh = pf_tune.rope_pe(qh, tabs[0], tabs[1], P, ckc)
            kv = pf_tune.rope_pe(ttnn.reshape(kvn, [U, 1, C, D]), tabs[0], tabs[1], P, ckc)
            for u in range(U):
                qu = ttnn.to_layout(ttnn.slice(qh, [u, 0, 0, 0], [u + 1, NH, C, D]), ttnn.ROW_MAJOR_LAYOUT)
                qu = ttnn.concat([qu, zero_heads], dim=1)
            outs = [
                ttnn.to_layout(ttnn.slice(o_list[u], [0, 0, 0, 0], [1, NH, C, D]), ttnn.TILE_LAYOUT) for u in range(U)
            ]
            o = pf_tune.rope_pe(ttnn.concat(outs, dim=0), tabs[0], tabs[2], P, ckc)
            cc = ttnn.experimental.nlp_concat_heads(o)
            return ttnn.reshape(ttnn.concat([zero, cc], dim=3), [1, 1, U * C, (NH + 1) * D])
        FA.qprep(q, tabs, P, U, C, NH, sparse=True)
        FA.kv_rope(kvn, tabs, P, U, C)
        return FA.post(o_list, tabs, P, U, C, NH)

    chain()  # compile + persistent buffers
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    for _ in range(reps):
        chain()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    for _ in range(4):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    ttnn.release_trace(md, tid)
    print("TRACE_DONE", mode, flush=True)
