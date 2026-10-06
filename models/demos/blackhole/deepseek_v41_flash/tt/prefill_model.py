# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1-Flash device PREFILL of a batch of prompts: tokens -> embedding -> layers (+Engram) -> last-token logits, leaving
the decode state of every layer on the device (see docs/superpowers/specs/2026-10-02-dsv41-device-prefill-design.md).

Layout: 4 mesh rows x U users per row (U = 4 at batch 16), every user's prompt padded to ``Sp`` (multiple of 64) with its last token;
the mesh row's R = U*Sp tokens are processed in chunks of T = 32 (user-major) by the token-wise blocks (mHC, router, MoE, shared expert,
Engram), attention runs over all R tokens of the row (``DSV41PrefillAttention``).
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.h2d import h2d, recording, replay
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_attention import clear_chunk_caches, pad_len
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_layer import colsplit_active, unpack_streams

T = 32
ER_RM = (
    os.environ.get("DSV41_ENGRAM_UPLOAD", "rm") == "rm"
)  # Engram rows uploaded row-major and tilized on the device (="tile": old host tilize)
HEAD_FUSED = (
    os.environ.get("DSV41_HEAD_FUSED", "1") == "1"
)  # one head trace for all users, last-token rows picked on the device


def to_chunks(x, rows, users, Sp, S, tail):
    """x [rows*users, S, ...] host -> list of chunks [rows*T, 1, *tail] (per row users x Sp tokens user-major, padded with the last real token)."""
    B = x.shape[0]
    xp = torch.cat([x, x[:, -1:].expand(B, Sp - S, *x.shape[2:])], dim=1)
    xr = xp.reshape(rows, users * Sp, *x.shape[2:])
    return [xr[:, c * T : (c + 1) * T].reshape(rows * T, 1, *tail) for c in range(users * Sp // T)]


def from_chunks(ch, rows, users, Sp, S, tail):
    """inverse of ``to_chunks`` for host chunks [rows*T, ...] -> [rows*users, S, *tail]."""
    xr = torch.stack([c.reshape(rows, T, *tail) for c in ch], dim=1).reshape(rows, users * Sp, *tail)
    return xr.reshape(rows * users, Sp, *tail)[:, :S]


def engram_own():
    """DSV41_PF_ENGRAM_OWN=1: column-split Engram on the own chunks only (forward_v2_own); read at call time."""
    return os.environ.get("DSV41_PF_ENGRAM_OWN", "0") == "1"


class DSV41PrefillModel:
    def __init__(self, md, layers, embedding, head, engram=None, host_rows=None, users_per_row=4):
        """layers: list of (layer_id, DSV41PrefillLayer); engram: {layer_id: DSV41DeviceEngram}; host_rows: HostEngramRows."""
        self.md, self.layers, self.embedding, self.head = md, layers, embedding, head
        self.engram, self.host_rows = engram or {}, host_rows
        self.rows, self.cols = tuple(md.shape)
        self.U = users_per_row
        self.shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(self.rows, self.cols))
        self.shard_rc = ttnn.ShardTensor2dMesh(
            md, dims=(0, 1), mesh_shape=(self.rows, self.cols)
        )  # column-split: row r, column c
        self.pre32 = ttnn.from_torch(
            torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(T, 1, 1, 1),
            device=md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        self.timing = {}
        self._bufs = {}
        self.pre_replay_hooks, self.post_replay_hooks = [], []
        self.cs = False
        self.dyn = self.dyn_trace = None
        self.fake_rows, self._fake = False, {}

    def _up(self, t, dtype, layout):
        return ttnn.from_torch(
            t.contiguous(),
            device=self.md,
            dtype=dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.shard,
        )

    def sync(self, name, t0):
        ttnn.synchronize_device(self.md)
        self.timing[name] = self.timing.get(name, 0.0) + time.perf_counter() - t0

    # ---- device-input staging (persistent buffers per chunk length: a single-chunk forward can be captured as ONE trace) --------
    def alloc_inputs(self, C):
        """Persistent device buffers for ONE prompt chunk of ``C`` tokens per user: tokens [R,1] (one per mesh row, R = U*C) and, per Engram layer,
        the rows [1,1,R,Kin] (token = tile row); the 32-token slices are taken on the device. One upload per tensor instead of one per 32 tokens.
        """
        self.cs = colsplit_active(
            self.U, C
        )  # column split of the token-wise work for this chunk size (falls back automatically)
        for _, pl in self.layers:
            pl.colsplit = self.cs
        if C in self._bufs:
            return self._bufs[C]
        R = self.U * C
        rows = self.rows
        if self.cs:  # per device [R/8, 1]: the tokens of the 32-token chunks this column owns
            tok = ttnn.from_torch(
                torch.zeros(rows * (R // self.cols), self.cols, dtype=torch.int32),
                device=self.md,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.shard_rc,
            )
        else:
            tok = self._up(torch.zeros(rows * R, 1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        er = {
            lid: self._up(
                torch.zeros(rows, 1, R, e.kin, dtype=torch.bfloat16),
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT if ER_RM else ttnn.TILE_LAYOUT,
            )
            for lid, e in self.engram.items()
        }
        self._bufs[C] = (tok, er)
        return self._bufs[C]

    def _host(self, t, dtype, layout):
        return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, mesh_mapper=self.shard)

    def prep_inputs(self, tokens, hashes=None):
        """CPU part of a chunk's inputs (safe to run in a worker thread): tokens [B, C] (padded) and the Engram row gather -> host tensors."""
        B, C = tokens.shape
        rows, R = self.rows, self.U * C
        out = {"tok": tokens.reshape(rows * R, 1).to(torch.int32)}
        for lid in self.engram:
            t1 = time.perf_counter()
            if (
                self.fake_rows
            ):  # throughput runs on hosts without the RAM tables: constant synthetic rows (the gather cost is measured separately)
                fr = self._fake.setdefault(lid, torch.randn(1, 1, self.engram[lid].kin).to(torch.bfloat16))
                out[lid] = fr.expand(rows, 1, R, fr.shape[-1]).contiguous()
                continue
            r = self.host_rows.rows(lid, hashes)  # [B, C, Kin] bf16, user-major: table gather + fp8 -> bf16 dequant
            t2 = time.perf_counter()
            out[lid] = r.reshape(rows, 1, R, r.shape[-1]).to(torch.bfloat16)
            self.timing["host_rows_gather_dequant"] = self.timing.get("host_rows_gather_dequant", 0.0) + t2 - t1
        return out

    def upload_inputs(self, prepped, bufs):
        t0 = time.perf_counter()
        tok, er = bufs
        if self.cs:  # chunk i (32 tokens) of a row goes to column i % 8: [rows*R,1] -> [rows*(R/8), cols]
            tk = prepped["tok"].reshape(self.rows, -1, self.cols, T).permute(0, 1, 3, 2)  # [rows, n8, T, cols]
            htok = ttnn.from_torch(
                tk.reshape(self.rows * tk.shape[1] * T, self.cols).contiguous(),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self.shard_rc,
            )
            h2d(htok, tok)
        else:
            h2d(self._host(prepped["tok"], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), tok)
        for lid, dev in er.items():
            t1 = time.perf_counter()
            h = self._host(
                prepped[lid], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT if ER_RM else ttnn.TILE_LAYOUT
            )  # row-major upload: the 32-token slices are tilized on the device (host tilize was ~0.7 s per 16k-token chunk)
            t2 = time.perf_counter()
            h2d(h, dev)
            t3 = time.perf_counter()
            self.timing["host_tilize"] = self.timing.get("host_tilize", 0.0) + t2 - t1
            self.timing["h2d_copy"] = self.timing.get("h2d_copy", 0.0) + t3 - t2
            self.h2d_bytes = (
                self.h2d_bytes + prepped[lid].numel() * 2 * self.cols
                if hasattr(self, "h2d_bytes")
                else prepped[lid].numel() * 2 * self.cols
            )
        self.timing["host_upload"] = self.timing.get("host_upload", 0.0) + time.perf_counter() - t0

    def write_inputs(self, tokens, hashes=None, bufs=None):
        t0 = time.perf_counter()
        pre = self.prep_inputs(tokens, hashes)
        self.timing["host_gather"] = self.timing.get("host_gather", 0.0) + time.perf_counter() - t0
        self.upload_inputs(pre, bufs or self.alloc_inputs(tokens.shape[1]))

    def forward_device(self, bufs, S, s0, C, hook=None, profile=False, dyn=False):
        """One prompt chunk (positions [s0, s0+C) of every user): embedding -> layers (+Engram) -> (last chunk only) head on the chunks holding the
        last real token of each user. Only enqueues device ops (traceable when ``hook`` / ``profile`` are off). Returns {local user: logits shard
        [1,1,1,vocab/cols] of its last token} for the last chunk, else None."""
        tok_dev, erows_dev = bufs
        last = s0 + C >= S
        if (
            dyn
        ):  # traced-chunk mode: the masks are built on the device from the per-chunk tensors, the outputs stay in ``self.dyn_out``
            self.dyn.build_masks()

        def sync(name, t0):
            if profile:
                self.sync(name, t0)

        t0 = time.perf_counter()
        n32 = self.U * C // T // (self.cols if self.cs else 1)
        xs = [self.embedding.forward(ttnn.slice(tok_dev, [c * T, 0], [(c + 1) * T, 1]))[0] for c in range(n32)]
        pres = [self.pre32 for _ in xs]
        sync("embedding", t0)
        if not getattr(self, "_moe_warm", False) and os.environ.get("DSV41_PREFILL_MOE_WARM", "1") != "0":
            self._moe_warm = True  # eager, before the first prefill compile / capture (see DSV41PrefillMoE.warmup)
            seen = set()
            for _, pl_ in self.layers:
                if getattr(pl_, "umoe", None) is not None and os.environ.get("DSV41_UNI_NODECODE") == "1":
                    continue  # prefill-only unified mode: no moe_compute weights to warm the baseline program with
                if id(pl_.pmoe) not in seen:
                    seen.add(id(pl_.pmoe))
                    pl_.pmoe.warmup()
        for lid, pl in self.layers:
            if lid in self.engram:
                xs, pres = unpack_streams(xs, pres)
                t0 = time.perf_counter()
                fe = (
                    self.engram[lid].forward
                    if os.environ.get("DSV41_ENGRAM_V2") == "0"
                    else self.engram[lid].forward_v2
                )
                kin = erows_dev[lid].shape[3]
                sl = lambda c: ttnn.to_layout(
                    ttnn.slice(erows_dev[lid], [0, 0, c * T, 0], [1, 1, (c + 1) * T, kin]), ttnn.TILE_LAYOUT
                )  # (to_layout is a no-op for tile rows)
                if (
                    self.cs
                    and engram_own()
                    and fe.__name__ == "forward_v2"
                    and self.engram[lid].mesh_config is not None
                ):
                    # own chunks only: one kv matmul per 8-chunk group + all_to_all of the kv shards (no gather of x, no 8 calls, no reduce_scatter)
                    new = []
                    for g, x in enumerate(xs):
                        rg = ttnn.to_layout(
                            ttnn.slice(
                                erows_dev[lid], [0, 0, g * self.cols * T, 0], [1, 1, (g + 1) * self.cols * T, kin]
                            ),
                            ttnn.TILE_LAYOUT,
                        )
                        new.append(self.engram[lid].forward_v2_own(x, rg))
                        ttnn.deallocate(rg)
                elif (
                    self.cs
                ):  # own-token chunks -> replicated 8-chunk groups -> Engram -> back (reduce_scatter of 8 identical copies, x1/8 exact)
                    mc, cc = pl.L.mesh_config, pl.L.ccl
                    new = []
                    for g, x in enumerate(xs):
                        xg = mc.allgather(x, cc, axis=1, dim=0)  # [8T,1,4,D]: chunk 8g + j at rows jT..
                        outs8 = []
                        if (
                            os.environ.get("DSV41_PFA_ENGRAM_BATCH", "0") == "1" and fe.__name__ == "forward_v2"
                        ):  # ONE T = 8*32 forward per group instead of 8 forwards at T = 32 (same maths, 8x fewer ops / CCLs)
                            rg = ttnn.to_layout(
                                ttnn.slice(
                                    erows_dev[lid], [0, 0, g * self.cols * T, 0], [1, 1, (g + 1) * self.cols * T, kin]
                                ),
                                ttnn.TILE_LAYOUT,
                            )
                            cat8 = fe(xg, rg)
                        else:
                            for j in range(self.cols):
                                xj = ttnn.slice(xg, [j * T, 0, 0, 0], [(j + 1) * T, 1, xg.shape[2], xg.shape[3]])
                                rj = sl(g * self.cols + j)
                                outs8.append(
                                    fe(xj, rj if fe.__name__ == "forward_v2" else ttnn.reshape(rj, [T, 1, 1, kin]))
                                )
                            cat8 = ttnn.concat(outs8, dim=0)
                        if (
                            os.environ.get("DSV41_PROF_EVERY", "0") != "0"
                        ):  # op-table profiling: drain the device profiler per group
                            ttnn.synchronize_device(self.md)
                            ttnn.ReadDeviceProfiler(self.md)
                        rs8 = ttnn.experimental.reduce_scatter_minimal_async(
                            cat8,
                            dim=0,
                            multi_device_global_semaphore=cc.get_rs_ping_pong_semaphore(),
                            num_links=cc.num_links,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            topology=cc.topology,
                            cluster_axis=1,
                            barrier_semaphore=cc.get_barrier_semaphore(),
                        )
                        new.append(ttnn.multiply(rs8, 1.0 / self.cols))
                        for t_ in (xg, cat8, rs8, *outs8):
                            ttnn.deallocate(t_)
                else:
                    new = [
                        fe(x, sl(c) if fe.__name__ == "forward_v2" else ttnn.reshape(sl(c), [T, 1, 1, kin]))
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
                xs, pres = unpack_streams(xs, pres)
                hook(lid, xs, pres)
        xs, pres = unpack_streams(xs, pres)
        if dyn:
            self.dyn_out = (xs, pres)
            return None
        if not last:
            for x in xs:
                ttnn.deallocate(x)
            for p in pres:
                ttnn.deallocate(p)
            return None
        t0 = time.perf_counter()
        lg = self.last_logits(xs, pres, S, s0, C)
        sync("head", t0)
        return lg

    # ---- fused head: ONE trace for all users; the last-token rows are selected on the device with one-hot matmuls ----------------------------
    def _head_fused_alloc(self, x0, p0):
        """Persistent buffers (allocated before any capture): per user a stash of the 32-token stream chunk holding its last token (column split: the own
        chunk, gathered over the columns inside the trace) and a one-hot row selector [1,1,1,N]."""
        self.head_stash = [(ttnn.clone(x0), ttnn.clone(p0)) for _ in range(self.U)]
        n = T * (self.cols if self.cs else 1)
        self.head_sel = [
            ttnn.from_torch(
                torch.zeros(1, 1, 1, n),
                device=self.md,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.md),
            )
            for _ in range(self.U)
        ]

    def _head_fused_body(self):
        L = self.layers[0][1].L
        xs_sel, ps_sel = [], []
        for u in range(self.U):
            x, p = self.head_stash[u]
            if self.cs:
                x = L.mesh_config.allgather(x, L.ccl, axis=1, dim=0)  # [8T,1,4,D]: row j*T + t = column j's chunk row t
                p = L.mesh_config.allgather(p, L.ccl, axis=1, dim=0)
            N = x.shape[0]
            D = x.shape[3]
            xr = ttnn.to_layout(
                ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), [1, 1, N, 4 * D]), ttnn.TILE_LAYOUT
            )
            pr = ttnn.to_layout(ttnn.reshape(ttnn.to_layout(p, ttnn.ROW_MAJOR_LAYOUT), [1, 1, N, 4]), ttnn.TILE_LAYOUT)
            sx = ttnn.matmul(self.head_sel[u], xr, compute_kernel_config=self.head.ckc32)  # [1,1,1,4D]
            sp = ttnn.matmul(self.head_sel[u], pr, compute_kernel_config=self.head.ckc32)  # [1,1,1,4]
            xs_sel.append(ttnn.reshape(ttnn.to_layout(sx, ttnn.ROW_MAJOR_LAYOUT), [1, 1, 4, D]))
            ps_sel.append(ttnn.reshape(ttnn.to_layout(sp, ttnn.ROW_MAJOR_LAYOUT), [1, 1, 1, 4]))
        X = ttnn.to_layout(ttnn.concat(xs_sel, dim=0), ttnn.TILE_LAYOUT)  # [U,1,4,D]
        P = ttnn.to_layout(ttnn.concat(ps_sel, dim=0), ttnn.TILE_LAYOUT)  # [U,1,1,4]
        return self.head.forward(X, P)  # [1,1,U,vocab/cols]

    def _head_fused_capture(self):
        self._head_fused_body()  # compile
        ttnn.synchronize_device(self.md)
        self.head_fused_trace = ttnn.begin_trace_capture(self.md, cq_id=0)
        self.head_fused_out = self._head_fused_body()
        ttnn.end_trace_capture(self.md, self.head_fused_trace, cq_id=0)

    def last_logits_fused(self, S, s0, C):
        xs, pres = self.dyn_out
        rows, U = self.rows, self.U
        idx = []
        for u in range(U):
            c, off = divmod(u * C + S - 1 - s0, T)
            idx.append((c, off))
            src = c // self.cols if self.cs else c
            ttnn.copy(xs[src], self.head_stash[u][0])
            ttnn.copy(pres[src], self.head_stash[u][1])
            n = T * (self.cols if self.cs else 1)
            sel = torch.zeros(1, 1, 1, n)
            sel[0, 0, 0, (c % self.cols) * T + off if self.cs else off] = 1.0
            h2d(
                ttnn.from_torch(
                    sel, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(self.md)
                ),
                self.head_sel[u],
            )
        ttnn.synchronize_device(self.md)
        ttnn.execute_trace(self.md, self.head_fused_trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.md)
        return self.head.gather_logits(self.head_fused_out)[: rows * U]  # [rows * U, vocab], row r*U + u

    def _head_body(self, key):
        """Head on the persistent ``head_in`` / ``head_pre``. Column split: the owner column ``key`` of the 8 holds the chunk, so gather first."""
        if not self.cs:
            return self.head.forward(self.head_in, self.head_pre)
        L = self.layers[0][1].L
        xg = L.mesh_config.allgather(self.head_in, L.ccl, axis=1, dim=0)
        pg = L.mesh_config.allgather(self.head_pre, L.ccl, axis=1, dim=0)
        xo = ttnn.slice(xg, [key * T, 0, 0, 0], [(key + 1) * T, 1, xg.shape[2], xg.shape[3]])
        po = ttnn.slice(pg, [key * T, 0, 0, 0], [(key + 1) * T, 1, pg.shape[2], pg.shape[3]])
        out = self.head.forward(xo, po)
        for t in (xg, pg, xo, po):
            ttnn.deallocate(t)
        return out

    def last_logits_traced(self, S, s0, C):
        """First-token logits [B, vocab] with NO allocation between trace replays: the head is a second trace on persistent head_in / head_pre,
        fed from per-chunk stashes (copied out before the first head replay: its temporaries may overlap the chunk trace's outputs).
        Column split: chunk c of a row lives in column c % 8 (list element c // 8); one head trace per owner column gathers it first.
        """
        xs, pres = self.dyn_out
        rows, U = self.rows, self.U
        out = torch.zeros(rows * U, 129280)
        need = {}
        for u in range(U):
            c, off = divmod(u * C + S - 1 - s0, T)
            need.setdefault(c, []).append((u, off))
        src = lambda c: c // self.cols if self.cs else c
        for k, c in enumerate(need):
            ttnn.copy(xs[src(c)], self.head_stash[k][0])
            ttnn.copy(pres[src(c)], self.head_stash[k][1])
        ttnn.synchronize_device(self.md)
        for k, (c, users) in enumerate(need.items()):
            ttnn.copy(self.head_stash[k][0], self.head_in)
            ttnn.copy(self.head_stash[k][1], self.head_pre)
            tid, head_out = self.head_traces[c % self.cols if self.cs else None]
            ttnn.execute_trace(self.md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.md)
            g = self.head.gather_logits(head_out).reshape(rows, T, -1)
            for u, off in users:
                for r in range(rows):
                    out[r * U + u] = g[r, off]
        return out

    def last_logits(self, xs, pres, S, s0, C):
        """Head on the 32-token chunks that hold the last real token of each user -> {local user: logits shard of its last token}."""
        full, lg = {}, {}
        for u in range(self.U):
            c, off = divmod(u * C + S - 1 - s0, T)
            if c not in full:
                if (
                    self.cs
                ):  # chunk c is owned by column c % 8 (list element c // 8): gather over the columns, take the owner's block
                    mc, cc = self.layers[0][1].L.mesh_config, self.layers[0][1].L.ccl
                    g, o = divmod(c, self.cols)
                    xg = mc.allgather(xs[g], cc, axis=1, dim=0)
                    pg = mc.allgather(pres[g], cc, axis=1, dim=0)
                    xo = ttnn.slice(xg, [o * T, 0, 0, 0], [(o + 1) * T, 1, xg.shape[2], xg.shape[3]])
                    po = ttnn.slice(pg, [o * T, 0, 0, 0], [(o + 1) * T, 1, pg.shape[2], pg.shape[3]])
                    full[c] = self.head.forward(xo, po)
                else:
                    full[c] = self.head.forward(xs[c], pres[c])
            lg[u] = ttnn.slice(
                full[c], [0, 0, off, 0], [1, 1, off + 1, full[c].shape[3]]
            )  # only the row of the last real token is read back
        return lg

    def read_logits(self, lg):
        """Host [B, vocab] logits of the last prompt token of every user from ``forward_device``'s outputs."""
        rows, U = self.rows, self.U
        out = torch.zeros(rows * U, 129280)
        for u in range(U):
            g = self.head.gather_logits(lg[u]).reshape(rows, -1)  # [rows, vocab]
            for r in range(rows):
                out[r * U + u] = g[r]
        return out

    def chunk_plan(self, S, chunk=None):
        """[(s0, C)] covering the prompt: chunks of ``chunk`` tokens (multiple of 128), the last one padded up to a multiple of 64."""
        Sp = pad_len(S)
        if not chunk or chunk >= Sp:
            return [(0, Sp)]
        assert chunk % 128 == 0
        plan, s0 = [], 0
        while s0 < S:
            plan.append((s0, chunk if S - s0 > chunk else pad_len(S - s0)))
            s0 += chunk
        return plan

    def run(self, tokens, chunk=None, hook=None, hashes=None):
        """DEFAULT prefill: one trace of a chunk forward replayed for every chunk (``run_traced_chunks``). ``chunk`` = tokens per user per chunk
        (multiple of 128; default ``default_chunk``). DSV41_PREFILL_EAGER=1 (or a per-layer ``hook``) selects the eager reference path.
        """
        if hook is not None or os.environ.get("DSV41_PREFILL_EAGER") == "1":
            return self.run_eager(tokens, chunk=chunk, hook=hook, hashes=hashes)
        return self.run_traced_chunks(tokens, chunk or self.default_chunk(tokens.shape[1]), hashes=hashes)

    def default_chunk(self, S):
        """Chunk size per user: the padded prompt, capped at DSV41_PREFILL_CHUNK_TOKENS (default 4096) tokens per mesh row / users per row."""
        cap = max(128, int(os.environ.get("DSV41_PREFILL_CHUNK_TOKENS", "4096")) // self.U // 128 * 128)
        return min(-(-S // 128) * 128, cap)

    def run_eager(self, tokens, chunk=None, hook=None, hashes=None):
        """Eager: tokens [B, S] -> logits [B, vocab] (host fp32) of the LAST prompt token of every user, the decode state of every layer left on the
        device. ``chunk``: tokens per user per chunk (multiple of 128; None = the whole prompt at once). ``hook(layer_id, outs, pouts)`` is
        called after every layer of a single-chunk run (diagnostics)."""
        B, S = tokens.shape
        assert B == self.rows * self.U
        if getattr(self, "dyn", None) is not None:
            self.teardown_dyn()  # the static eager path and the traced-chunk mode keep different carried state
        self.timing = {}
        self.S = S
        plan = self.chunk_plan(S, chunk)
        self.plan = plan
        if self.engram and hashes is None:
            t0 = time.perf_counter()
            hashes = self.host_rows.hashes(tokens, 0)  # [B, S, ...]; the host history must see the prompt once
            self.timing["engram_host_hash"] = time.perf_counter() - t0
        for _, pl in self.layers:
            pl.pa.begin()
        lg = None

        def prep(
            ci,
        ):  # CPU gather of chunk ci (runs in a worker thread, overlapped with the device work of the previous chunk)
            s0, C = plan[ci]
            n = min(C, S - s0)
            tk = tokens[:, s0 : s0 + n]
            hs = None if hashes is None else hashes[:, s0 : s0 + n]
            if n < C:  # pad with the last real token (hash)
                tk = torch.cat([tk, tk[:, -1:].expand(B, C - n)], dim=1)
                if hs is not None:
                    hs = torch.cat([hs, hs[:, -1:].expand(B, C - n, *hs.shape[2:])], dim=1)
            return self.prep_inputs(tk, hs)

        t_run = time.perf_counter()
        pool = ThreadPoolExecutor(1)
        fut = pool.submit(prep, 0)
        for ci, (s0, C) in enumerate(plan):
            t0 = time.perf_counter()
            pre = fut.result()
            self.timing["host_gather_wait"] = self.timing.get("host_gather_wait", 0.0) + time.perf_counter() - t0
            if ci + 1 < len(plan):
                fut = pool.submit(prep, ci + 1)
            bufs = self.alloc_inputs(C)
            self.upload_inputs(pre, bufs)
            lg = self.forward_device(bufs, S, s0, C, hook=hook if len(plan) == 1 else None, profile=True)
            if len(plan) > 1:
                ttnn.synchronize_device(self.md)
                print(
                    f"  prefill chunk {ci + 1}/{len(plan)} (s0={s0}, C={C}) done at {time.perf_counter() - t_run:.1f} s",
                    flush=True,
                )
            if len(plan) > 1:  # single chunk: keep the constants, a trace capture cannot upload
                clear_chunk_caches()
        pool.shutdown()
        t0 = time.perf_counter()
        out = self.read_logits(lg)
        self.timing["readback"] = time.perf_counter() - t0
        return out

    # ---- traced chunks: ONE trace of the chunk forward, replayed for every chunk of the prompt -------------------------------
    def setup_dyn(self, C, S_pad):
        from models.demos.blackhole.deepseek_v41_flash.tt.prefill_dyn import DynCtx

        ratios, rope_src = set(), {}
        for _, pl in self.layers:
            ratios.add(pl.pa.ratio)
            rope_src.setdefault(pl.pa.compressed, pl.pa.a)
        self.dyn = DynCtx(self.md, C, S_pad, ratios, rope_src)
        for _, pl in self.layers:
            pl.pa.alloc_dyn(self.dyn)
        self.dyn_trace = None
        self.pre_replay_hooks = getattr(self, "pre_replay_hooks", [])
        self.post_replay_hooks = getattr(self, "post_replay_hooks", [])

    def teardown_dyn(self):
        """Release the chunk trace and the per-chunk buffers (a different chunk size / padded prompt length needs a new capture)."""
        if getattr(self, "dyn_trace", None) is not None:
            ttnn.release_trace(self.md, self.dyn_trace)
        if getattr(self, "head_fused_trace", None) is not None:
            ttnn.release_trace(self.md, self.head_fused_trace)
            self.head_fused_trace = None
        for tid, _ in getattr(self, "head_traces", {}).values():
            ttnn.release_trace(self.md, tid)
        self.head_traces = {}
        self.dyn_trace = None
        for _, pl in self.layers:
            pl.pa.dyn = pl.pa.halo = pl.pa.lat_buf = None
            if pl.pa.sparse is not None:
                pl.pa.sparse.free_dyn()
        self.dyn = None

    def begin_chunk(self, s0, C):
        """Host side of a chunk: refresh every per-chunk persistent tensor (RoPE tables, masks, latent offsets) and call the registered hooks
        (e.g. the paged-state sink's own uploads) right before the replay."""
        t0 = time.perf_counter()
        self.dyn.update(s0)
        t1 = time.perf_counter()
        for h in self.pre_replay_hooks:
            h(s0, C)
        t2 = time.perf_counter()
        self.timing["w_dynupd"] = self.timing.get("w_dynupd", 0.0) + t1 - t0
        self.timing["w_prehooks"] = self.timing.get("w_prehooks", 0.0) + t2 - t1

    def run_traced_chunks(self, tokens, chunk, hashes=None, S_pad_max=None):
        """tokens [B, S] -> logits [B, vocab] of the last prompt token. One trace of the chunk forward (C = ``chunk`` tokens per user, multiple of
        128), replayed for every chunk; the last partial chunk is padded. The first call compiles + captures (``self.timing['compile']``).
        """
        B, S = tokens.shape
        C = chunk
        assert C % 128 == 0 and B == self.rows * self.U
        n = -(-S // C)
        S_pad = n * C
        self.timing = {}
        self.S, self.plan = S, [(c * C, C) for c in range(n)]
        if self.engram and hashes is None:
            hashes = self.host_rows.hashes(tokens, 0)
        want = S_pad_max or S_pad
        if getattr(self, "dyn", None) is None or self.dyn.C != C or self.dyn.S_pad != want:
            self.teardown_dyn()
            self.setup_dyn(C, want)
        pad = lambda t: torch.cat([t, t[:, -1:].expand(B, S_pad - S, *t.shape[2:])], dim=1) if S_pad > S else t
        tk, hs = pad(tokens), None if hashes is None else pad(hashes)
        bufs = self.alloc_inputs(C)

        def prep(ci):
            return self.prep_inputs(tk[:, ci * C : (ci + 1) * C], None if hs is None else hs[:, ci * C : (ci + 1) * C])

        def prep_rec(ci):
            """prep + ALL the host work of the chunk's uploads (Engram rows tensors, rope tables, masks, sink index tables) in the worker thread; the
            device copies are only recorded and replayed by the main thread."""
            t0 = time.perf_counter()
            pre = prep(ci)
            t1 = time.perf_counter()
            with recording() as ops:
                self.upload_inputs(pre, bufs)
                t2 = time.perf_counter()
                self.begin_chunk(ci * C, C)
            t3 = time.perf_counter()
            tm = self.timing
            tm["w_prep"] = tm.get("w_prep", 0.0) + t1 - t0
            tm["w_upload"] = tm.get("w_upload", 0.0) + t2 - t1
            tm["w_begin"] = tm.get("w_begin", 0.0) + t3 - t2
            return ops

        pool = ThreadPoolExecutor(1)
        if self.dyn_trace is None:  # compile pass (eager, chunk 0), then capture
            t0 = time.perf_counter()
            self.upload_inputs(prep(0), bufs)
            self.begin_chunk(0, C)
            self.forward_device(bufs, S, 0, C, dyn=True)
            ttnn.synchronize_device(self.md)
            own_head = (
                type(self).last_logits is DSV41PrefillModel.last_logits
            )  # subclasses with their own head (ragged hand-off) keep it
            xs0, pres0 = self.dyn_out
            if own_head and HEAD_FUSED:
                self._head_fused_alloc(xs0[0], pres0[0])
            elif own_head:
                # head trace FIRST: its persistent buffers must exist before the chunk trace is captured (nothing may be allocated between replays)
                self.head_in, self.head_pre = ttnn.clone(xs0[0]), ttnn.clone(pres0[0])
                # one stash per user: the head trace's temporaries may overlap the chunk trace's outputs, so every needed chunk of the stream
                # is copied out before the first head replay
                self.head_stash = [(ttnn.clone(xs0[0]), ttnn.clone(pres0[0])) for _ in range(self.U)]
            for x in list(xs0) + list(pres0):
                ttnn.deallocate(x)
            if own_head and HEAD_FUSED:
                self._head_fused_capture()
            elif own_head:
                self.head_traces = {}
                for key in range(self.cols) if self.cs else [None]:
                    self._head_body(key)  # compile
                    ttnn.synchronize_device(self.md)
                    tid = ttnn.begin_trace_capture(self.md, cq_id=0)
                    out_t = self._head_body(key)
                    ttnn.end_trace_capture(self.md, tid, cq_id=0)
                    self.head_traces[key] = (tid, out_t)
            for _, pl in self.layers:
                pl.pa.reset_dyn()  # before any capture only
            ttnn.synchronize_device(self.md)
            self.dyn_trace = ttnn.begin_trace_capture(self.md, cq_id=0)
            self.forward_device(bufs, S, 0, C, dyn=True)
            ttnn.end_trace_capture(self.md, self.dyn_trace, cq_id=0)
            ttnn.synchronize_device(self.md)
            self.timing["compile_and_capture"] = time.perf_counter() - t0
        MODE = os.environ.get(
            "DSV41_PF_ASYNC", "1"
        )  # "1" async, "0" sync (host work still prepared in the worker), "legacy": the original loop
        legacy = MODE == "legacy"
        fut = pool.submit(prep if legacy else prep_rec, 0)
        t_run = time.perf_counter()
        # Asynchronous replay loop (DSV41_PF_ASYNC=1): the per-chunk uploads and the trace are enqueued on CQ 0 (in-order: a chunk's uploads only land
        # after the previous replay finished), so the host work of chunk i+1 (index tables, Engram rows, uploads) overlaps the replay of chunk i. The
        # host runs at most 1 replay ahead (event of replay ci-1 awaited before chunk ci's copies). Chunks whose post hooks read the replay's outputs
        # (ragged last-token head) synchronize as before. DSV41_PF_ASYNC=0 restores the per-chunk synchronize_device.
        events = []
        PF_ASYNC = MODE == "1"  # traced chunk replays enqueued without a per-chunk device sync
        for ci in range(n):
            t0 = time.perf_counter()
            if PF_ASYNC and ci >= 1:
                # host copies block (holding the GIL) until the queue reaches them, i.e. until the previous replay is done: wait for it in a
                # GIL-free call so the worker thread keeps preparing chunk ci+1, then the copies are only a few ms
                ttnn.event_synchronize(events[ci - 1])
            t_ev = time.perf_counter()
            ops = fut.result()
            if ci + 1 < n:
                fut = pool.submit(prep if legacy else prep_rec, ci + 1)
            t_g = time.perf_counter()
            if legacy:
                self.upload_inputs(ops, bufs)
                t_u = time.perf_counter()
                self.begin_chunk(ci * C, C)
                t1 = time.perf_counter()
            else:
                replay(ops)
                del ops
                t_u = time.perf_counter()
                t1 = t_u
            ttnn.execute_trace(self.md, self.dyn_trace, cq_id=0, blocking=False)
            need = (not PF_ASYNC) or any(
                getattr(hk, "needs_sync", lambda s0_, C_: True)(ci * C, C) for hk in self.post_replay_hooks
            )
            if PF_ASYNC:
                events.append(ttnn.record_event(self.md, 0))
            if need:
                ttnn.synchronize_device(self.md)
                for (
                    hk
                ) in (
                    self.post_replay_hooks
                ):  # e.g. the ragged last-token head: dyn_out = (xs, pres) is valid for THIS chunk only
                    hk(ci * C, C)
            t2 = time.perf_counter()
            if os.environ.get(
                "DSV41_PROF_REPLAY"
            ):  # device-profiler drain after every replay (profiling runs only; excluded from the timing)
                ttnn.ReadDeviceProfiler(self.md)
            tm = self.timing
            tm["dev_wait"] = tm.get("dev_wait", 0.0) + t_ev - t0
            tm["gather_wait"] = tm.get("gather_wait", 0.0) + t_g - t_ev
            tm["upload"] = tm.get("upload", 0.0) + t_u - t_g
            tm["begin_chunk"] = tm.get("begin_chunk", 0.0) + t1 - t_u
            tm["host_per_chunk"] = tm.get("host_per_chunk", 0.0) + t1 - t_ev
            tm["replay_per_chunk"] = tm.get("replay_per_chunk", 0.0) + t2 - t1 + t_ev - t0
            if n > 4 and (ci + 1) % 8 == 0:
                print(f"  traced chunk {ci + 1}/{n} enqueued at {time.perf_counter() - t_run:.1f} s", flush=True)
        t0 = time.perf_counter()
        ttnn.synchronize_device(self.md)
        self.timing["replay_per_chunk"] += time.perf_counter() - t0
        pool.shutdown()
        t0 = time.perf_counter()
        if type(self).last_logits is DSV41PrefillModel.last_logits:
            out = (self.last_logits_fused if HEAD_FUSED else self.last_logits_traced)(S, (n - 1) * C, C)
        else:
            xs, pres = self.dyn_out
            lg = self.last_logits(xs, pres, S, (n - 1) * C, C)
            out = self.read_logits(lg)
        self.timing["head_readback"] = time.perf_counter() - t0
        self.timing["total_replay_loop"] = time.perf_counter() - t_run
        return out

    def capture_trace(self, S):
        """Capture the whole single-chunk prefill (embedding .. head) as ONE trace; call after an eager ``run`` warmed every program."""
        ((s0, C),) = self.chunk_plan(S)
        self._tr = (self.alloc_inputs(C), S, C)
        for _, pl in self.layers:
            pl.pa.begin()
        ttnn.synchronize_device(self.md)
        self.trace_id = ttnn.begin_trace_capture(self.md, cq_id=0)
        self.trace_out = self.forward_device(self._tr[0], S, 0, C)
        ttnn.end_trace_capture(self.md, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.md)

    def run_traced(self, tokens, hashes=None):
        self.timing = {}
        bufs, S, C = self._tr
        B = tokens.shape[0]
        tk = torch.cat([tokens, tokens[:, -1:].expand(B, C - S)], dim=1)
        if self.engram and hashes is None:
            hashes = self.host_rows.hashes(tokens, 0)
        hs = None if hashes is None else torch.cat([hashes, hashes[:, -1:].expand(B, C - S, *hashes.shape[2:])], dim=1)
        self.write_inputs(tk, hs, bufs)
        t0 = time.perf_counter()
        ttnn.execute_trace(self.md, self.trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.md)
        self.timing["trace_replay"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = self.read_logits(self.trace_out)
        self.timing["readback"] = time.perf_counter() - t0
        return out

    def last_token_logits(self, xs, pres, S, Sp):
        """Final hc_pre + norm + LM head on the chunks that hold the last real token of each user -> [B, vocab] host fp32."""
        rows, U = self.rows, self.U
        out = torch.zeros(rows * U, 129280)
        cache = {}
        for u in range(U):
            c, off = divmod(u * Sp + S - 1, T)
            if c not in cache:
                lg = self.head.forward(xs[c], pres[c])  # [1,1,T,vocab/cols] per device
                cache[c] = self.head.gather_logits(lg).reshape(rows, T, -1)  # [rows, T, vocab]
                ttnn.deallocate(lg)
            for r in range(rows):
                out[r * U + u] = cache[c][r, off]
        return out
