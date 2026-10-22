# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-trace indexer (tt/prefill_single.py: ``add_keys_single`` / ``select_single``) against the replicated FIFO path (``add_keys_dyn`` / ``select_dyn``) with the same random weights, keys and
queries, chunk after chunk up to a deep position. Compares the selected ENTRY sets (FIFO slot - off | scratch / history slot -> entry). Env: PSI_R (compress ratio 1 or 2, default 1),
PSI_CHUNKS (chunks of the whole prompt = capacity, default 300 = 153,600 tokens at C=512), PSI_CHECK (check every n-th chunk, default 12), PSI_U (1)."""

import os
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import prefill_single as PS
from models.demos.blackhole.deepseek_v41_flash.tt.indexer import DIM as IDIM
from models.demos.blackhole.deepseek_v41_flash.tt.indexer import HEADS as IHEADS
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_dyn import visibility_mask
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_sparse import SKIP, TOPK, DSV41PrefillIndexer
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

DP = {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100_000_000}
HEAD_DIM = 512


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DP, id="ring")], indirect=True)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_prefill_indexer_single(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    R = int(os.environ.get("PSI_R", "1"))
    C = 512
    n_chunks = int(os.environ.get("PSI_CHUNKS", "300"))
    check = int(os.environ.get("PSI_CHECK", "12"))
    U = int(os.environ.get("PSI_U", "1"))
    Cc = C // R
    L = (
        int(os.environ.get("PSI_LCAP", str(n_chunks * C))) // R
    )  # entries of the build's whole context (= Wmax / R); chunks beyond the prompt stay empty
    os.environ["DSV41_PF_SINGLE"] = "1"
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
    ctx = SimpleNamespace(C=C, ratios=[R])
    ctx.sg = PS.SingleCtx(md, ctx, U)
    ix_s.alloc_single(ctx, L, cols, mc, ccl, True)
    print(f"PSI R={R} C={C} chunks={n_chunks} L={L} entries: per-column slab {ix_s.sh['Lc']} slots", flush=True)

    def rope_tab(n):
        ang = torch.rand(n, 32) * 6.28
        c, s = torch.ones(n, HEAD_DIM), torch.zeros(n, HEAD_DIM)
        c[:, -64:] = torch.cos(ang).repeat_interleave(2, dim=1)
        s[:, -64:] = torch.sin(ang).repeat_interleave(2, dim=1)
        return up(c.reshape(1, 1, n, HEAD_DIM)), up(s.reshape(1, 1, n, HEAD_DIM))

    pcol_r = up(torch.arange(L, dtype=torch.float32).reshape(1, 1, 1, -1), ttnn.float32)
    worst = 1.0
    for i in range(n_chunks):
        s0 = i * C
        lat = torch.randn(U, 1, Cc, HEAD_DIM).to(torch.bfloat16)
        lat_t, tab_l = up(lat), rope_tab(Cc)
        ctx.sg.update(ctx, s0)
        ix_r.add_keys_dyn(lat_t, tab_l, Cc)
        ix_s.add_keys_single(lat_t, tab_l, Cc, ctx)
        if i % check and i != n_chunks - 1:
            continue
        h = up(torch.randn(U, 1, C, 5120).to(torch.bfloat16))
        qr = up(torch.randn(U, 1, C, 1280).to(torch.bfloat16))
        tab_t = rope_tab(C)
        pos = s0 + torch.arange(C)
        lim = up(((pos + 1) // R).float().reshape(1, 1, C, 1), ttnn.float32)
        off_f = L - (s0 + C) // R
        off = up(torch.full((1, 1, 1, 1), float(off_f)), ttnn.float32)
        vis_r = visibility_mask(pcol_r, off, lim, C, L)
        ids_r = ix_r.select_dyn(h, qr, tab_t, vis_r, C, L)
        ids_s = ix_s.select_single(h, qr, tab_t, C, ctx)
        ttnn.synchronize_device(md)
        a = ttnn.to_torch(ttnn.get_device_tensors(ids_r)[0]).reshape(U, C, TOPK).long() & 0xFFFFFFFF
        b = ttnn.to_torch(ttnn.get_device_tensors(ids_s)[0]).reshape(U, C, TOPK).long() & 0xFFFFFFFF
        ea = torch.where(a == SKIP, torch.full_like(a, -1), a - off_f)
        eb = torch.where(b == SKIP, torch.full_like(b, -1), torch.where(b < Cc, s0 // R + b, b - Cc))
        exact, overlap = 0, 0.0
        for r_ in range(C):
            sa, sb = set(ea[0, r_].tolist()), set(eb[0, r_].tolist())
            exact += sa == sb
            overlap += len(sa & sb) / max(1, len(sa))
        worst = min(worst, overlap / C)
        if os.environ.get("PSI_SCORE_EVERY", "60") != "0" and (
            i % int(os.environ.get("PSI_SCORE_EVERY", "60")) == 0 or i == n_chunks - 1
        ):
            # the reference scores (replicated masked scores in FIFO slot order = entry + off) of the entries each path selected: a path that MISSES high-scoring entries has a lower sum
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
            scr_ = ttnn.experimental.indexer_score_dsa(
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
                    ttnn.get_device_tensors(ttnn.where(vis_r, ttnn.to_layout(scr_, ttnn.TILE_LAYOUT), float("-inf")))[0]
                )
                .float()
                .reshape(C, L)
            )
            ok = ea[0] >= 0
            sa_ = torch.where(ok, torch.gather(scm, 1, (ea[0].clamp(min=0) + off_f).clamp(max=L - 1)), torch.zeros(1))
            ok_b = eb[0] >= 0
            sb_ = torch.where(ok_b, torch.gather(scm, 1, (eb[0].clamp(min=0) + off_f).clamp(max=L - 1)), torch.zeros(1))
            ra, rb = sa_.sum(1), sb_.sum(1)
            rel = (ra - rb).abs() / ra.abs().clamp(min=1e-6)
            print(
                f"PSI   score sums ref vs single: mean rel diff {float(rel.mean()):.3e}, max {float(rel.max()):.3e}, rows with single sum lower by >1e-3: {int(((ra - rb) / ra.abs().clamp(min=1e-6) > 1e-3).sum())}",
                flush=True,
            )
        print(
            f"PSI chunk {i} (s0 {s0}, entries visible to the last row {(s0 + C) // R}): identical entry sets {exact}/{C} rows, mean overlap {overlap / C:.5f}, "
            f"sentinels ref {int((a == SKIP).sum())} single {int((b == SKIP).sum())}",
            flush=True,
        )
    print(f"PSI DONE worst mean overlap {worst:.5f}", flush=True)
