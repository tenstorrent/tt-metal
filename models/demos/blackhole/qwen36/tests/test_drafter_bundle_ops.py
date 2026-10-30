# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device op tests of the r6 drafter bundle (8-chip TP ring, e.g. one Galaxy row with QWEN36_FABRIC_RING=1), at the
MTP drafter's shapes. Each test checks the numbers and logs traced per-call device times (flag off vs on).

  test_mtp_fc_progcfg      QWEN36_MTP_FC_PROGCFG: the fc [rows, 10240] x [10240, 640/dev] BFP8 HiFi2 on ttnn's auto
                           config vs explicit 1D decode configs (in0_block_w sweep, in0 in L1 / DRAM, 20 / 10 cores);
                           every config must match torch (PCC) and the auto output closely.
  test_mtp_embed_row       QWEN36_MTP_EMBED_ROW: Qwen36MTP._embed_rows == today's TILE lookup + reshape, bit for bit.
  test_mtp_sdpa_window     QWEN36_MTP_SDPA_WINDOW: the drafter's paged decode SDPA (8 users, 3 q heads, head_dim 256,
                           bf8 paged KV, 64 cores per head batch) with sliding_window_size W == a torch reference that
                           attends [pos + 1 - W, pos]; users with pos < W are bit-identical to the windowless call;
                           32K timings for W in {1024, 2048, 4096, 8192}.

  source /tmp/penv.sh; pytest models/demos/blackhole/qwen36/tests/test_drafter_bundle_ops.py -s
"""
import math
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt import tp_common as tpc

TRACE_REGION = 96 * 1024 * 1024


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else float(torch.equal(a, b))


def _rep(mesh, t, dtype, layout, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), memory_config=mem
    )


def _per_dev(mesh, t):
    """[nd, ...] torch: device d's tensor at index d (concat on a new leading axis)."""
    nd = mesh.get_num_devices()
    full = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    return full.reshape(nd, *full.shape[1:]) if full.shape[0] == nd else full.reshape(nd, -1, *full.shape[1:])


def _trace_us(mesh, fn, reps=20, replays=10):
    """Mean device time of one fn() call: ``reps`` calls captured in one trace, replayed ``replays`` times. fn() frees
    everything it allocates (or hands back persistent buffers)."""
    fn()  # compile outside the capture
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(reps):
        fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(replays):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    us = (time.perf_counter() - t0) / (replays * reps) * 1e6
    ttnn.release_trace(mesh, tid)
    return us


# ------------------------------------------------------------------------------------------------------------------ #
# QWEN36_MTP_FC_PROGCFG
# ------------------------------------------------------------------------------------------------------------------ #
@torch.no_grad()
@parametrize_mesh_tp(trace_region_size=TRACE_REGION)
@pytest.mark.parametrize("rows", [8, 32])
def test_mtp_fc_progcfg(mesh_device, rows, reset_seeds):
    mesh = mesh_device
    nd = mesh.get_num_devices()
    K, N = 10240, 640  # 2 * dim, dim / tp at TP8 (Qwen3.8-27B)
    gw = mesh.compute_with_storage_grid_size().x
    g = torch.Generator().manual_seed(rows)
    w_t = (torch.randn(K, N * nd, generator=g) * 0.02).to(torch.bfloat16)
    x_t = torch.randn(1, 1, rows, K, generator=g).to(torch.bfloat16)
    w = ttnn.from_torch(
        w_t,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
    )
    x_dram = _rep(mesh, x_t, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    x_l1 = _rep(mesh, x_t, ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.L1_MEMORY_CONFIG)
    cc = tpc.COMPUTE_HIFI2
    ref = (x_t.float() @ w_t.float()).reshape(1, 1, rows, nd, N)  # device d's columns at [..., d, :]

    def pc(ibw, cores=20):
        return tpc.create_matmul_1d_decode_progcfg(32, K, N, num_cores=cores, grid_w=gw, max_in0_block_w=ibw)

    variants = {"auto_dram": (x_dram, None), "auto_l1": (x_l1, None)}
    for ibw in (8, 16, 20, 32, 40, 64):
        variants[f"pc{ibw}_l1"] = (x_l1, pc(ibw))
    variants["pc32_dram"] = (x_dram, pc(32))
    variants["pc32_l1_c10"] = (x_l1, pc(32, 10))
    variants["pc16_l1_c10"] = (x_l1, pc(16, 10))

    outs, times, failed = {}, {}, {}
    for name, (x, p) in variants.items():
        kw = {} if p is None else {"program_config": p}

        def run(x=x, kw=kw):
            return ttnn.linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=cc, **kw)

        try:
            o = run()
        except Exception as e:  # a config that does not fit (e.g. L1 at a large in0_block_w) is reported, not fatal
            failed[name] = str(e).splitlines()[0][:160]
            continue
        outs[name] = _per_dev(mesh, o).reshape(nd, rows, N).float()
        ttnn.deallocate(o)
        times[name] = _trace_us(mesh, lambda: ttnn.deallocate(run()))
    base = outs["auto_dram"]
    lines = []
    for name, o in outs.items():
        pcc = min(_pcc(o[d], ref[0, 0, :, d, :]) for d in range(nd))
        dmax = float((o - base).abs().max())
        lines.append(f"{name}: {times[name]:.1f} us, PCC vs torch {pcc:.6f}, max |diff| vs auto {dmax:.4g}")
        assert pcc > 0.999, f"{name}: PCC {pcc}"
    logger.info(f"[mtp fc rows={rows}] " + " | ".join(lines) + (f" | FAILED {failed}" if failed else ""))
    assert "pc32_l1" in outs, f"the flag's default config failed: {failed.get('pc32_l1')}"


# ------------------------------------------------------------------------------------------------------------------ #
# QWEN36_MTP_EMBED_ROW
# ------------------------------------------------------------------------------------------------------------------ #
@torch.no_grad()
@parametrize_mesh_tp(trace_region_size=TRACE_REGION)
@pytest.mark.parametrize("rows", [1, 4, 8, 16, 32])
def test_mtp_embed_row(mesh_device, rows, reset_seeds):
    from types import SimpleNamespace

    from models.demos.blackhole.qwen36.tt.mtp import Qwen36MTP

    mesh = mesh_device
    nd = mesh.get_num_devices()
    vocab, dim = 4096, 5120
    g = torch.Generator().manual_seed(100 + rows)
    table = torch.randn(1, 1, vocab, dim, generator=g).to(torch.bfloat16)
    weights = ttnn.as_tensor(
        table,
        dtype=ttnn.bfloat16,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ids_t = torch.randint(0, vocab, (rows, 1), generator=g, dtype=torch.int32)
    ids = _rep(mesh, ids_t, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    head = Qwen36MTP.__new__(Qwen36MTP)
    head.embd = SimpleNamespace(weights=weights)

    def old():
        e = ttnn.embedding(ids, weights, layout=ttnn.TILE_LAYOUT)  # Embedding.forward
        return ttnn.reshape(e, (1, 1, e.shape[0] * e.shape[1], e.shape[-1]))  # Qwen36MTP.forward_decode

    def new():
        return head._embed_rows(ids)

    o_old, o_new = old(), new()
    assert tuple(o_old.shape) == tuple(o_new.shape) == (1, 1, rows, dim // nd)
    assert o_new.layout == ttnn.TILE_LAYOUT and o_new.dtype == ttnn.bfloat16
    a = _per_dev(mesh, o_old).reshape(nd, rows, dim // nd)
    b = _per_dev(mesh, o_new).reshape(nd, rows, dim // nd)
    want = table[0, 0, ids_t[:, 0].long()].reshape(rows, nd, dim // nd).permute(1, 0, 2)
    assert torch.equal(a, b), "row-major embedding differs from today's lookup"
    assert torch.equal(b, want), "embedding rows differ from the table"
    ttnn.deallocate(o_old)
    ttnn.deallocate(o_new)
    t_old = _trace_us(mesh, lambda: ttnn.deallocate(old()))
    t_new = _trace_us(mesh, lambda: ttnn.deallocate(new()))
    logger.info(f"[mtp embed rows={rows}] today's lookup {t_old:.1f} us, one-tile-row lookup {t_new:.1f} us: identical")


# ------------------------------------------------------------------------------------------------------------------ #
# QWEN36_MTP_SDPA_WINDOW
# ------------------------------------------------------------------------------------------------------------------ #
BLOCK, NH, HD = 64, 3, 256  # paged block, local q heads at TP8, head dim
WINDOW_POS = {
    # pos < W, pos = W - 1, W, W + 1, chunk / tile edges, mid and long context (W = 1024 and 2048 below)
    "mixed": [4095, 1023, 1024, 1025, 300, 2047, 5000, 63],
    "32k": [32767, 32766, 32700, 32000, 31999, 32768 - 513, 30001, 32640],
}


def _drafter_sdpa_cfg(mesh):
    """The drafter's decode SDPA program config (TPAttention._forward_decode_lean with decode_sdpa_max_cores=64)."""
    grid = mesh.compute_with_storage_grid_size()
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        exp_approx_mode=False,
        q_chunk_size=0,
        k_chunk_size=0,
        max_cores_per_head_batch=64,
    )


@torch.no_grad()
@parametrize_mesh_tp(trace_region_size=TRACE_REGION)
@pytest.mark.parametrize("pos_set", list(WINDOW_POS))
def test_mtp_sdpa_window(mesh_device, pos_set, reset_seeds):
    mesh = mesh_device
    pos = WINDOW_POS[pos_set]
    B = len(pos)
    nb = math.ceil((max(pos) + 1) / BLOCK)
    nblocks = B * nb
    g = torch.Generator().manual_seed(7)
    k_init = torch.randn(nblocks, 1, BLOCK, HD, generator=g).to(torch.bfloat16)
    v_init = torch.randn(nblocks, 1, BLOCK, HD, generator=g).to(torch.bfloat16)
    pt = torch.randperm(nblocks, generator=g).reshape(B, nb).to(torch.int32)
    q = torch.randn(1, B, NH, HD, generator=g).to(torch.bfloat16)
    kc = _rep(mesh, k_init, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    vc = _rep(mesh, v_init, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    pt_tt = _rep(mesh, pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    pos_tt = _rep(mesh, torch.tensor(pos, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    q_tt = _rep(mesh, q, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    cfg = _drafter_sdpa_cfg(mesh)
    scale = HD**-0.5

    def call(w):
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_tt,
            kc,
            vc,
            page_table_tensor=pt_tt,
            cur_pos_tensor=pos_tt,
            scale=scale,
            program_config=cfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            **({"sliding_window_size": w} if w else {}),
        )

    def dev0(t):
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1].float()  # [1, B, NH, HD]

    kd = ttnn.to_torch(kc, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:nblocks].float()
    vd = ttnn.to_torch(vc, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:nblocks].float()

    def reference(w):
        out = torch.zeros(B, NH, HD)
        for u in range(B):
            k_u = kd[pt[u].long()].reshape(nb * BLOCK, HD)
            v_u = vd[pt[u].long()].reshape(nb * BLOCK, HD)
            p = pos[u]
            lo = max(0, p + 1 - w) if w else 0
            s = (q[0, u].float() @ k_u[lo : p + 1].T) * scale
            out[u] = torch.softmax(s, dim=-1) @ v_u[lo : p + 1]
        return out

    o = call(0)
    full = dev0(o)[0]
    ttnn.deallocate(o)
    res = []
    for w in (1024, 2048):
        o = call(w)
        got = dev0(o)[0]
        ttnn.deallocate(o)
        ref = reference(w)
        for u in range(B):
            pcc = _pcc(got[u], ref[u])
            assert pcc > 0.99, f"W={w} user {u} pos {pos[u]}: PCC {pcc}"
            if pos[u] + 1 <= w:  # the window covers the whole context: exactly today's call
                assert torch.equal(got[u], full[u]), f"W={w} user {u} pos {pos[u]} < W differs from no window"
        res.append((w, round(min(_pcc(got[u], ref[u]) for u in range(B)), 5)))
    logger.info(f"[mtp sdpa window {pos_set}] min PCC vs windowed torch: {res}")

    if pos_set == "32k":
        times = {w: _trace_us(mesh, lambda w=w: ttnn.deallocate(call(w)), reps=10) for w in (0, 1024, 2048, 4096, 8192)}
        logger.info(
            "[mtp sdpa window 32k] per call (8 users, 3 heads, bf8 KV, 64 cores/head batch): "
            + ", ".join(f"W={w or 'full'} {t:.1f} us" for w, t in times.items())
        )
    for t in (kc, vc, pt_tt, pos_tt, q_tt):
        ttnn.deallocate(t)
