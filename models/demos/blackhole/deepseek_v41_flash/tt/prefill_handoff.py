# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PREFILL -> DECODE hand-off into the paged KV pool, for prompts of DIFFERENT lengths per user.

* ``PagedStateSink``: the ``state_sink`` of every ``DSV41PrefillAttention`` (called once per chunk). Writes, per layer and chunk, into the SAME pool
  the decode attention reads (``paged_ops.PagedKVPool``): the window ring rows (last 128 real positions of every user, slot = position % 128),
  the compressed latents of the kv-source layers through the users' page tables, and the ratio-2 compressor's "previous token" state ``prev_cs``
  (the [kv | score] projection of the LAST prompt token of every user).
* ``GenPrefillModel``: ``DSV41PrefillModel`` whose head tail takes the last real token of every user at its own position (ragged lengths: the
  mesh rows run the same program, so every row reads its own row index through the chunk that holds it) and greedy-samples on the device.
"""

import os
import time

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import paged_ops as P
from models.demos.blackhole.deepseek_v41_flash.tt.attention import HEAD_DIM
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_layer import colsplit_active
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_model import DSV41PrefillModel, T

SKIP = 0xFFFFFFFF


class PagedStateSink:
    """Hand-off writer. All per-chunk index data lives in PERSISTENT device tensors (one set per chunk length C) that ``update(s0, C)`` refreshes from the host
    before every chunk (eager or trace replay); ``write`` itself is static-shape and trace-safe."""

    def __init__(self, md, pool, users_per_row):
        self.md, self.pool, self.U = md, pool, users_per_row
        self.rows, self.cols = tuple(md.shape)
        self.shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(self.rows, self.cols))
        self.lens = None  # [B] real prompt lengths (global user order: row-major over mesh rows)
        self.bufs = {}  # C -> {"ring": t, "lat": {src: t}, "idx": t, "mask": t}

    def set_lengths(self, lens):
        self.lens = torch.as_tensor(lens).long()

    def _host(self, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, mesh_mapper=self.shard)

    def _host_values(self, s0, C):
        """host tensors of the chunk: ring ids, {src: latent ids}, last-token idx, mask."""
        U, R, pool, RING = self.U, self.rows, self.pool, self.pool.ring_rows
        pos = torch.arange(s0, s0 + C)
        ring = torch.full((R, U * C), SKIP, dtype=torch.int64)
        lat = {s: torch.full((R, U * (C // P.SRC_RATIO[s])), SKIP, dtype=torch.int64) for s in P.SOURCES}
        idx = torch.zeros(R * U, 1, dtype=torch.int32)
        m = torch.zeros(R, 1, U, 1)
        K = C // 32
        hm = torch.zeros(
            R * U * K, 1, 1, 1
        )  # ragged-head selector: 1 at (user u, 32-token sub-chunk c) holding the user's last prompt token
        for r in range(R):
            for u in range(U):
                b = r * U + u
                S = int(self.lens[b])
                ok = (pos < S) & (pos >= S - RING)
                ring[r, u * C : (u + 1) * C] = torch.where(ok, u * RING + pos % RING, torch.full_like(pos, SKIP))
                for s in P.SOURCES:
                    rt = P.SRC_RATIO[s]
                    Cc = C // rt
                    ent = torch.arange(s0 // rt, s0 // rt + Cc)
                    okc = ent < S // rt
                    if bool(okc.any()):
                        seg = torch.full((Cc,), SKIP, dtype=torch.int64)
                        seg[okc] = pool.phys_rows(b, s, ent[okc]).long()
                        lat[s][r, u * Cc : (u + 1) * Cc] = seg
                p = S - 1
                if s0 <= p < s0 + C:
                    idx[b, 0] = u * C + (p - s0)
                    m[r, 0, u, 0] = 1.0
                    hm[(r * U + u) * K + (p - s0) // 32] = 1.0
        return ring, lat, idx, m, hm

    def _cs_mask(self, hm, C):
        """Column-split ragged head: chunk i of a mesh row lives in column i % cols (own-chunk list element i // cols). Per (row, column, user, element g) selector
        = 1 where the user's last-token sub-chunk is 8g + column. Host [rows*U*n8, cols, 1, 1], sharded over (rows, cols) -> per device [U*n8,1,1,1].
        """
        R, U, cols, K = self.rows, self.U, self.cols, C // 32
        n8 = U * K // cols
        out = torch.zeros(R * U * n8, cols, 1, 1)
        h = hm.reshape(R, U, K)
        for r in range(R):
            for u in range(U):
                if h[r, u].any():
                    i = u * K + int(h[r, u].argmax())
                    out[(r * U + u) * n8 + i // cols, i % cols] = 1.0
        return out

    def _host_cs(self, hm, C):
        return ttnn.from_torch(
            self._cs_mask(hm, C),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.md, dims=(0, 1), mesh_shape=(self.rows, self.cols)),
        )

    def bind(self, C):
        """Allocate the persistent index tensors of chunk length C (call before any trace capture)."""
        if C in self.bufs:
            return self.bufs[C]
        ring, lat, idx, m, hm = self._host_values(0, C)
        up = lambda t, dt, lay=ttnn.ROW_MAJOR_LAYOUT: ttnn.from_torch(
            t.contiguous(),
            device=self.md,
            dtype=dt,
            layout=lay,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.shard,
        )
        self.bufs[C] = {
            "ring": up(ring.reshape(self.rows, 1, -1).to(torch.int32), ttnn.uint32),
            "lat": {s: up(v.reshape(self.rows, 1, -1).to(torch.int32), ttnn.uint32) for s, v in lat.items()},
            "idx": up(idx, ttnn.uint32),
            "mask": up(m, ttnn.float32, ttnn.TILE_LAYOUT),
            "hmask": up(hm, ttnn.float32, ttnn.TILE_LAYOUT),
        }
        if colsplit_active(self.U, C):
            self.bufs[C]["hmask_cs"] = ttnn.to_device(
                self._host_cs(hm, C), self.md, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        return self.bufs[C]

    def update(self, s0, C):
        """Refresh the persistent index tensors for the chunk of positions [s0, s0 + C) (host work + 6 small uploads; call before each chunk / replay)."""
        bufs = self.bind(C)
        ring, lat, idx, m, hm = self._host_values(s0, C)
        ttnn.copy_host_to_device_tensor(
            self._host(ring.reshape(self.rows, 1, -1).to(torch.int32), ttnn.uint32), bufs["ring"]
        )
        for s, v in lat.items():
            ttnn.copy_host_to_device_tensor(
                self._host(v.reshape(self.rows, 1, -1).to(torch.int32), ttnn.uint32), bufs["lat"][s]
            )
        ttnn.copy_host_to_device_tensor(self._host(idx, ttnn.uint32), bufs["idx"])
        ttnn.copy_host_to_device_tensor(self._host(m, ttnn.float32, ttnn.TILE_LAYOUT), bufs["mask"])
        ttnn.copy_host_to_device_tensor(self._host(hm, ttnn.float32, ttnn.TILE_LAYOUT), bufs["hmask"])
        if colsplit_active(self.U, C):
            ttnn.copy_host_to_device_tensor(self._host_cs(hm, C), bufs["hmask_cs"])

    # ---- the sink (trace-safe: static shapes, persistent index tensors) --------------------------------------------------------------
    def write(self, attn, pa, kv, lat, cs, s0, C, h):
        pool, U = self.pool, self.U
        C = int(kv.shape[2])
        bufs = self.bufs[C]
        if os.environ.get("DSV41_SINKDBG") == "1":
            kk = ttnn.to_torch(ttnn.get_device_tensors(kv)[0]).float()
            hh = ttnn.to_torch(ttnn.get_device_tensors(h)[0]).float()
            print(
                f"SINKDBG kv absmax {float(kk.abs().max()):.3f} nan {int(kk.isnan().sum())} h absmax {float(hh.abs().max()):.3f} nan {int(hh.isnan().sum())}",
                flush=True,
            )
            print(
                f"SINKDBG write ring_slot {attn.ring_slot} base {pool.ring_base(attn.ring_slot)} kv {tuple(kv.shape)} {kv.dtype} lat {None if lat is None else tuple(lat.shape)} cs {cs is not None}",
                flush=True,
            )
        skip = os.environ.get("DSV41_SINKSKIP", "")
        if "all" in skip:
            return
        src = ttnn.to_layout(ttnn.reshape(kv, [1, 1, U * C, HEAD_DIM]), ttnn.ROW_MAJOR_LAYOUT)
        P.paged_scatter_rows(pool.pool, src, bufs["ring"], base_offset=pool.ring_base(attn.ring_slot))
        ttnn.deallocate(src)
        if lat is not None and "lat" not in skip:  # kv-source layer: its latents live in the shared pages
            Cc = C // attn.ratio
            src = ttnn.to_layout(ttnn.reshape(lat, [1, 1, U * Cc, HEAD_DIM]), ttnn.ROW_MAJOR_LAYOUT)
            P.paged_scatter_rows(pool.pool, src, bufs["lat"][attn.src_layer])
            ttnn.deallocate(src)
        if (
            cs is not None
        ):  # ratio-2 owner: prev_cs <- [kv|score] projection of each user's last prompt token (where it is inside this chunk)
            R = h.shape[2]
            table = ttnn.to_layout(ttnn.reshape(h, [R, h.shape[3]]), ttnn.ROW_MAJOR_LAYOUT)
            emb = ttnn.embedding(bufs["idx"], table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # [U,1,D]
            ttnn.deallocate(table)
            emb = ttnn.reshape(emb, [1, 1, U, h.shape[3]])
            new = ttnn.linear(emb, attn.c_wcat, compute_kernel_config=attn.ckc, dtype=ttnn.float32)  # [1,1,U,1024]
            keep = ttnn.add(ttnn.neg(bufs["mask"]), 1.0)
            upd = ttnn.add(ttnn.multiply(attn.prev_cs, keep), ttnn.multiply(new, bufs["mask"]))
            ttnn.copy(upd, attn.prev_cs)
            for t in (emb, new, keep, upd):
                ttnn.deallocate(t)


class GenPrefillModel(DSV41PrefillModel):
    """``DSV41PrefillModel`` with ragged last-token extraction + on-device greedy sampling of the first token."""

    def forward_device(self, bufs, S, s0, C, hook=None, profile=False, dyn=False):
        """Dynamic-chunk forward + (dyn only) the RAGGED HEAD INSIDE the same trace: for every user the 32-token sub-chunk holding its last prompt token is
        selected on the device with a per-(mesh row, user, sub-chunk) 0/1 mask (persistent tensor refreshed by ``PagedStateSink.update``), run through the
        head and sampled greedily. The outputs (tokens [32,1], logits shard [1,1,32,vocab/cols] per user) are trace outputs read by ``Model._post_chunk``:
        NO eager device op or allocation happens between replays (they hang / corrupt traced programs)."""
        out = super().forward_device(bufs, S, s0, C, hook=hook, profile=profile, dyn=dyn)
        if dyn:
            self._ragged_head_traced(C)
        return out

    def _sum_cols(self, x):
        """sum over the 8 mesh columns of x [T,...] (one column holds the data, the others zeros): all-gather along dim 0, then add the 8 blocks."""
        g = self.mesh_config.allgather(x, self.ccl, axis=1, dim=0)
        T_ = x.shape[0]
        out = ttnn.slice(g, [0, 0, 0, 0], [T_, g.shape[1], g.shape[2], g.shape[3]])
        for j in range(1, self.cols):
            out = ttnn.add(out, ttnn.slice(g, [j * T_, 0, 0, 0], [(j + 1) * T_, g.shape[1], g.shape[2], g.shape[3]]))
        return out

    def _ragged_head_traced(self, C):
        xs, pres = self.dyn_out
        U, K = self.U, C // 32
        hmask = self.sink.bufs[C]["hmask"]
        self.head_out = (
            []
        )  # (the outputs of the eager compile pass are leaked on purpose: no deallocation inside a capture)
        for u in range(U):
            sel_x = sel_p = None
            if colsplit_active(
                U, C
            ):  # own chunks only: select locally with the per-column mask, then sum over the 8 columns (exactly one is non-zero)
                n8 = len(xs)
                hcs = self.sink.bufs[C]["hmask_cs"]
                for g in range(n8):
                    m = ttnn.slice(hcs, [u * n8 + g, 0, 0, 0], [u * n8 + g + 1, 1, 1, 1])
                    xc, pc = ttnn.multiply(xs[g], m), ttnn.multiply(pres[g], m)
                    sel_x = xc if sel_x is None else ttnn.add(sel_x, xc)
                    sel_p = pc if sel_p is None else ttnn.add(sel_p, pc)
                sel_x, sel_p = self._sum_cols(sel_x), self._sum_cols(sel_p)
                lg = self.head.forward(sel_x, sel_p)
                tk = self.head.sample_global(lg, self.mesh_config, self.ccl)
                self.head_out.append((lg, tk))
                continue
            for c in range(K):
                m = ttnn.slice(hmask, [u * K + c, 0, 0, 0], [u * K + c + 1, 1, 1, 1])
                xc, pc = ttnn.multiply(xs[u * K + c], m), ttnn.multiply(pres[u * K + c], m)
                sel_x = xc if sel_x is None else ttnn.add(sel_x, xc)
                sel_p = pc if sel_p is None else ttnn.add(sel_p, pc)
            lg = self.head.forward(sel_x, sel_p)
            tk = self.head.sample_global(lg, self.mesh_config, self.ccl)
            self.head_out.append((lg, tk))

    def last_logits(self, xs, pres, S, s0, C):
        """run_traced_chunks calls this after its replay loop with the LAST chunk's streams and the scalar max length: the ragged per-user head already ran in
        the post-replay hooks (``Model._post_chunk``), so do nothing (and leave no tensors alive between replays)."""
        return {}

    def read_logits(self, lg):
        return None

    def set_head_sampling(self, mesh_config, ccl):
        self.mesh_config, self.ccl = mesh_config, ccl

    def forward_device_legacy(self, bufs, S, s0, C, hook=None, profile=False, last_pos=None, want_logits=False):
        """One chunk of positions [s0, s0+C) of every user (S = max prompt length). ``last_pos`` [B] = per-user index of the last prompt token.
        Returns {global user b: (first token int, logits [vocab] fp32 or None)} for the users whose last token is inside this chunk.
        """

        tok_dev, erows_dev = bufs

        def sync(name, t0):
            if profile:
                self.sync(name, t0)

        t0 = time.perf_counter()
        n32 = self.U * C // T
        xs = [self.embedding.forward(ttnn.slice(tok_dev, [c * T, 0], [(c + 1) * T, 1]))[0] for c in range(n32)]
        pres = [self.pre32 for _ in xs]
        sync("embedding", t0)
        for lid, pl in self.layers:
            if lid in self.engram:
                t0 = time.perf_counter()
                kin = erows_dev[lid].shape[3]
                new = [
                    self.engram[lid].forward_v2(
                        x,
                        ttnn.to_layout(
                            ttnn.slice(erows_dev[lid], [0, 0, c * T, 0], [1, 1, (c + 1) * T, kin]), ttnn.TILE_LAYOUT
                        ),
                    )
                    for c, x in enumerate(xs)
                ]
                for x in xs:
                    ttnn.deallocate(x)
                xs = new
                sync("engram_dev", t0)
            t0 = time.perf_counter()
            outs, pouts = pl.forward(xs, pres, S, s0=s0)
            if xs[0] is not outs[0]:
                for x in xs:
                    ttnn.deallocate(x)
            if pres[0] is not self.pre32:
                for p in pres:
                    ttnn.deallocate(p)
            xs, pres = outs, pouts
            sync("layers", t0)
            if hook is not None:
                hook(lid, xs, pres)
        t0 = time.perf_counter()
        out = self.ragged_tail(xs, pres, s0, C, last_pos, want_logits)
        for x in xs:
            ttnn.deallocate(x)
        for p in pres:
            ttnn.deallocate(p)
        sync("head", t0)
        return out

    def ragged_tail(self, xs, pres, s0, C, last_pos, want_logits=False):
        """Head + greedy sampling for the users whose last prompt token lies in the chunk [s0, s0 + C) whose output streams are ``xs`` / ``pres`` (lists of
        32-token chunks). Eager, does not free xs / pres. -> {global user b: (token, logits row or None)}."""
        U, rows, cols = self.U, self.rows, self.cols
        need = {}
        for b in range(rows * U):
            p = int(last_pos[b])
            if s0 <= p < s0 + C:
                r, u = divmod(b, U)
                c32, off = divmod(u * C + p - s0, T)
                need.setdefault(c32, []).append((r, off, b))
        out = {}
        for c32, items in need.items():
            lg = self.head.forward(xs[c32], pres[c32])
            tk = self.head.sample_global(
                lg, self.mesh_config, self.ccl
            )  # [T,1] uint32, identical on all devices of a row
            devs = ttnn.get_device_tensors(ttnn.from_device(tk))
            full = self.head.gather_logits(lg).reshape(rows, T, -1) if want_logits else None
            for r, off, b in items:
                t = int(ttnn.to_torch(devs[r * cols]).reshape(-1)[off])
                out[b] = (t, None if full is None else full[r, off].clone())
            ttnn.deallocate(lg)
            ttnn.deallocate(tk)
        return out
