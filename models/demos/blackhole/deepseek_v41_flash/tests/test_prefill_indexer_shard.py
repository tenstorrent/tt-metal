# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Column-sharded prefill indexer (DSV41_PIDX_SHARD, tt/prefill_sparse.py ``select_dyn_sharded`` / ``add_keys_dyn_sharded``) against the replicated FIFO path with the same random weights, keys
and queries: a few chunks of the FIFO update, then the top-512 of the last chunk. Reports the share of rows whose selected id SET is identical, the mean set overlap, the sentinel counts, the
time per select and the per-column buffer bytes. Env: PS_L (entries, default 16384), PS_C (chunk tokens, 512), PS_CHUNKS (10), PS_U (1)."""

import os
import time
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.indexer import DIM as IDIM
from models.demos.blackhole.deepseek_v41_flash.tt.indexer import HEADS as IHEADS
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_dyn import visibility_mask
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_sparse import SKIP, TOPK, DSV41PrefillIndexer
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

DP = {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100_000_000}
R = 4
HEAD_DIM = 512


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DP, id="ring")], indirect=True)
@pytest.mark.timeout(3600)
@torch.no_grad()
def test_prefill_indexer_shard(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    L = int(os.environ.get("PS_L", "16384"))
    C = int(os.environ.get("PS_C", "512"))
    n_chunks = int(os.environ.get("PS_CHUNKS", "10"))
    U = int(os.environ.get("PS_U", "1"))
    Cc = C // R
    torch.manual_seed(0)
    mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    w = {
        "wq_b": torch.randn(IHEADS * IDIM, 1280).to(torch.bfloat16) * 0.03,
        "weights_proj": torch.randn(IHEADS, 5120).to(torch.bfloat16) * 0.03,
        "wk": torch.randn(IDIM, HEAD_DIM).to(torch.bfloat16) * 0.05,
        "k_norm": torch.ones(IDIM).to(torch.bfloat16),
    }
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t, dt=ttnn.bfloat16, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(
        t.contiguous(), device=md, dtype=dt, layout=lay, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
    )
    ix_r = DSV41PrefillIndexer(md, U, R, L, w_idx=w, freqs_cis=None)
    ix_s = DSV41PrefillIndexer(md, U, R, L, dec=ix_r.dec, key_owner=None)
    ix_r.KL = L
    ix_r.keys = ttnn.zeros([U, 1, L, IDIM], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md)
    ctx = SimpleNamespace(C=C)
    ix_s.alloc_shard(ctx, L, cols, mc, ccl, True)
    Lc = ix_s.sh["Lc"]
    print(
        f"PIS L={L} C={C} U={U}: per-column slab {Lc} slots (+32 pad), key bytes {Lc * IDIM * 2 / 2**20:.1f} MiB (replicated {L * IDIM * 2 / 2**20:.1f})",
        flush=True,
    )

    def rope_tab(n):  # (cos, sin) [1,1,n,512]: 1 / 0 off the rotated dims, random angles on the last 64
        ang = torch.rand(n, 32) * 6.28
        c, s = torch.ones(n, HEAD_DIM), torch.zeros(n, HEAD_DIM)
        c[:, -64:] = torch.cos(ang).repeat_interleave(2, dim=1)
        s[:, -64:] = torch.sin(ang).repeat_interleave(2, dim=1)
        return up(c.reshape(1, 1, n, HEAD_DIM)), up(s.reshape(1, 1, n, HEAD_DIM))

    pcol_r = up(torch.arange(L, dtype=torch.float32).reshape(1, 1, 1, -1), ttnn.float32)
    res = {}
    for i in range(n_chunks):
        s0 = i * C
        lat = torch.randn(U, 1, Cc, HEAD_DIM).to(torch.bfloat16)
        lat_t, tab_l = up(lat), rope_tab(Cc)
        ix_r.add_keys_dyn(lat_t, tab_l, Cc)
        ix_s.add_keys_dyn_sharded(lat_t, tab_l, Cc)
        h = up(torch.randn(U, 1, C, 5120).to(torch.bfloat16))
        qr = up(torch.randn(U, 1, C, 1280).to(torch.bfloat16))
        tab_t = rope_tab(C)
        pos = s0 + torch.arange(C)
        lim = up(((pos + 1) // R).float().reshape(1, 1, C, 1), ttnn.float32)
        off = up(torch.full((1, 1, 1, 1), float(L - (s0 + C) // R)), ttnn.float32)
        vis_r = visibility_mask(pcol_r, off, lim, C, L)
        vis_s = visibility_mask(ix_s.sh["pcol"], off, lim, C, Lc + 32)
        ids_r = ix_r.select_dyn(h, qr, tab_t, vis_r, C, L)
        ids_s = ix_s.select_dyn_sharded(h, qr, tab_t, vis_s, C, L)
        ttnn.synchronize_device(md)
        a = ttnn.to_torch(ttnn.get_device_tensors(ids_r)[0]).reshape(U, C, TOPK).long() & 0xFFFFFFFF
        devs = ttnn.get_device_tensors(ids_s)
        b = ttnn.to_torch(devs[0]).reshape(U, C, TOPK).long() & 0xFFFFFFFF
        b_last = ttnn.to_torch(devs[-1]).reshape(U, C, TOPK).long() & 0xFFFFFFFF
        same_dev = bool(all(set(b[0, r_].tolist()) == set(b_last[0, r_].tolist()) for r_ in range(C)))
        nvis = (s0 + C) // R
        exact_rows, overlap, sk_a, sk_b = 0, 0.0, int((a == SKIP).sum()), int((b == SKIP).sum())
        for u in range(U):
            for r_ in range(C):
                sa, sb = set(a[u, r_].tolist()), set(b[u, r_].tolist())
                exact_rows += sa == sb
                overlap += len(sa & sb) / TOPK
        n = U * C
        print(
            f"PIS chunk {i} (visible entries up to {nvis}): identical id sets {exact_rows}/{n} rows, mean overlap {overlap / n:.5f}, sentinels replicated {sk_a} sharded {sk_b}, "
            f"column 0 == column 7 result {same_dev}",
            flush=True,
        )
        res[i] = (exact_rows / n, overlap / n, sk_a == sk_b, same_dev)
        if (
            i == n_chunks - 1
        ):  # the scores of the two id sets (replicated masked scores) must be the same multiset in every row: the sets only differ in ties at the 512th score
            d = ix_r.dec
            w_ = ttnn.to_layout(ttnn.linear(h, d.wproj, compute_kernel_config=ix_r.ckc), ttnn.ROW_MAJOR_LAYOUT)
            q_ = ttnn.to_layout(ttnn.linear(qr, d.wq_b, compute_kernel_config=ix_r.ckc), ttnn.ROW_MAJOR_LAYOUT)
            qh = ttnn.to_layout(ttnn.permute(ttnn.reshape(q_, [U, C, IHEADS, IDIM]), (0, 2, 1, 3)), ttnn.TILE_LAYOUT)
            qh = ix_r._rope(
                qh,
                (
                    ttnn.slice(tab_t[0], [0, 0, 0, HEAD_DIM - IDIM], [1, 1, C, HEAD_DIM]),
                    ttnn.slice(tab_t[1], [0, 0, 0, HEAD_DIM - IDIM], [1, 1, C, HEAD_DIM]),
                ),
            )
            qh = ix_r._fp4(qh)
            wt = ttnn.to_layout(ttnn.reshape(w_, [U, 1, C, IHEADS]), ttnn.TILE_LAYOUT)
            sc = ttnn.experimental.indexer_score_dsa(
                ttnn.slice(qh, [0, 0, 0, 0], [1, IHEADS, C, IDIM]),
                ix_r.keys,
                ttnn.slice(wt, [0, 0, 0, 0], [1, 1, C, IHEADS]),
                chunk_start_idx=L - 32,
                kv_len=L,
                program_config=ix_r.cfg,
                cache_batch_idx=0,
            )
            scm = (
                ttnn.to_torch(
                    ttnn.get_device_tensors(ttnn.where(vis_r, ttnn.to_layout(sc, ttnn.TILE_LAYOUT), float("-inf")))[0]
                )
                .float()
                .reshape(C, L)
            )
            bad_rows = 0
            for r_ in range(C):
                ia, ib = a[0, r_], b[0, r_]
                va, vb = scm[r_, ia[ia != SKIP]].sort().values, scm[r_, ib[ib != SKIP]].sort().values
                bad_rows += not torch.equal(va, vb)
            print(
                f"PIS selected-score multisets differ in {bad_rows}/{C} rows (0 = the id sets only differ in score ties)",
                flush=True,
            )
            res["multiset_bad"] = bad_rows
        for t in (h, qr, ids_r, ids_s, vis_r, vis_s):
            ttnn.deallocate(t)

    kf = ix_s.keys_full()
    ttnn.synchronize_device(md)
    ka = ttnn.to_torch(ttnn.get_device_tensors(ix_r.keys)[0]).float().reshape(U, L, IDIM)
    kb = ttnn.to_torch(ttnn.get_device_tensors(kf)[0]).float().reshape(U, L, IDIM)
    kb_last = ttnn.to_torch(ttnn.get_device_tensors(kf)[-1]).float().reshape(U, L, IDIM)
    print(
        f"PIS keys_full vs replicated FIFO: bit-exact {bool(torch.equal(ka, kb))}, max|diff| {float((ka - kb).abs().max()):.3g}, last device equal {bool(torch.equal(kb, kb_last))}",
        flush=True,
    )
    assert torch.equal(ka, kb)

    def timed(fn, n=3):
        fn()
        ttnn.synchronize_device(md)
        t = time.perf_counter()
        for _ in range(n):
            o = fn()
            ttnn.deallocate(o)
        ttnn.synchronize_device(md)
        return (time.perf_counter() - t) / n * 1e3

    h = up(torch.randn(U, 1, C, 5120).to(torch.bfloat16))
    qr = up(torch.randn(U, 1, C, 1280).to(torch.bfloat16))
    tab_t = rope_tab(C)
    vis_r = visibility_mask(pcol_r, off, lim, C, L)
    vis_s = visibility_mask(ix_s.sh["pcol"], off, lim, C, Lc + 32)
    # trace capture after an eager pass (the serving warm-up does exactly this): no program may be compiled under the capture
    ix_s.select_dyn_sharded(h, qr, tab_t, vis_s, C, L)
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    ids_t = ix_s.select_dyn_sharded(h, qr, tab_t, vis_s, C, L)
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
    print("PIS trace capture + replay of select_dyn_sharded OK", flush=True)
    ttnn.release_trace(md, tid)
    ms_r = timed(lambda: ix_r.select_dyn(h, qr, tab_t, vis_r, C, L))
    ms_s = timed(lambda: ix_s.select_dyn_sharded(h, qr, tab_t, vis_s, C, L))
    print(f"PIS select time: replicated {ms_r:.2f} ms, sharded {ms_s:.2f} ms (L={L}, C={C}, U={U})", flush=True)
    last = res[n_chunks - 1]
    assert res.pop("multiset_bad") == 0
    assert all(v[2] and v[3] for v in res.values()), "sentinel counts or columns differ"
    assert last[1] > 0.995, f"mean overlap {last[1]}"
