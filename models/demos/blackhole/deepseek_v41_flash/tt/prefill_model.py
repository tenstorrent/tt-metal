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
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_attention import clear_chunk_caches, pad_len

T = 32


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


class DSV41PrefillModel:
    def __init__(self, md, layers, embedding, head, engram=None, host_rows=None, users_per_row=4):
        """layers: list of (layer_id, DSV41PrefillLayer); engram: {layer_id: DSV41DeviceEngram}; host_rows: HostEngramRows."""
        self.md, self.layers, self.embedding, self.head = md, layers, embedding, head
        self.engram, self.host_rows = engram or {}, host_rows
        self.rows, self.cols = tuple(md.shape)
        self.U = users_per_row
        self.shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(self.rows, self.cols))
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
        if C in self._bufs:
            return self._bufs[C]
        R = self.U * C
        rows = self.rows
        tok = self._up(torch.zeros(rows * R, 1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        er = {
            lid: self._up(torch.zeros(rows, 1, R, e.kin, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
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
        ttnn.copy_host_to_device_tensor(self._host(prepped["tok"], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), tok)
        for lid, dev in er.items():
            t1 = time.perf_counter()
            h = self._host(prepped[lid], ttnn.bfloat16, ttnn.TILE_LAYOUT)  # host tilize + shard
            t2 = time.perf_counter()
            ttnn.copy_host_to_device_tensor(h, dev)
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
        n32 = self.U * C // T
        xs = [self.embedding.forward(ttnn.slice(tok_dev, [c * T, 0], [(c + 1) * T, 1]))[0] for c in range(n32)]
        pres = [self.pre32 for _ in xs]
        sync("embedding", t0)
        for lid, pl in self.layers:
            if lid in self.engram:
                t0 = time.perf_counter()
                fe = (
                    self.engram[lid].forward
                    if os.environ.get("DSV41_ENGRAM_V2") == "0"
                    else self.engram[lid].forward_v2
                )
                kin = erows_dev[lid].shape[3]
                sl = lambda c: ttnn.slice(erows_dev[lid], [0, 0, c * T, 0], [1, 1, (c + 1) * T, kin])
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
                hook(lid, xs, pres)
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

    def last_logits(self, xs, pres, S, s0, C):
        """Head on the 32-token chunks that hold the last real token of each user -> {local user: logits shard of its last token}."""
        full, lg = {}, {}
        for u in range(self.U):
            c, off = divmod(u * C + S - 1 - s0, T)
            if c not in full:
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
        self.dyn_trace = None
        for _, pl in self.layers:
            pl.pa.dyn = pl.pa.halo = pl.pa.lat_buf = None
            if pl.pa.sparse is not None:
                pl.pa.sparse.dyn_on = False
        self.dyn = None

    def begin_chunk(self, s0, C):
        """Host side of a chunk: refresh every per-chunk persistent tensor (RoPE tables, masks, latent offsets) and call the registered hooks
        (e.g. the paged-state sink's own uploads) right before the replay."""
        self.dyn.update(s0)
        for h in self.pre_replay_hooks:
            h(s0, C)

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

        pool = ThreadPoolExecutor(1)
        fut = pool.submit(prep, 0)
        if self.dyn_trace is None:  # compile pass (eager, chunk 0), then capture
            t0 = time.perf_counter()
            self.upload_inputs(fut.result(), bufs)
            fut = pool.submit(prep, 1) if n > 1 else None
            self.begin_chunk(0, C)
            self.forward_device(bufs, S, 0, C, dyn=True)
            ttnn.synchronize_device(self.md)
            for x in self.dyn_out[0]:
                ttnn.deallocate(x)
            for _, pl in self.layers:
                pl.pa.reset_dyn()
            ttnn.synchronize_device(self.md)
            self.dyn_trace = ttnn.begin_trace_capture(self.md, cq_id=0)
            self.forward_device(bufs, S, 0, C, dyn=True)
            ttnn.end_trace_capture(self.md, self.dyn_trace, cq_id=0)
            ttnn.synchronize_device(self.md)
            self.timing["compile_and_capture"] = time.perf_counter() - t0
            fut = pool.submit(prep, 0)
        for _, pl in self.layers:
            pl.pa.reset_dyn()
        t_run = time.perf_counter()
        for ci in range(n):
            t0 = time.perf_counter()
            pre = fut.result()
            if ci + 1 < n:
                fut = pool.submit(prep, ci + 1)
            self.upload_inputs(pre, bufs)
            self.begin_chunk(ci * C, C)
            t1 = time.perf_counter()
            ttnn.execute_trace(self.md, self.dyn_trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.md)
            t2 = time.perf_counter()
            for (
                hk
            ) in (
                self.post_replay_hooks
            ):  # e.g. the ragged last-token head: dyn_out = (xs, pres) is valid for THIS chunk only
                hk(ci * C, C)
            self.timing["host_per_chunk"] = self.timing.get("host_per_chunk", 0.0) + t1 - t0
            self.timing["replay_per_chunk"] = self.timing.get("replay_per_chunk", 0.0) + t2 - t1
            if n > 4 and (ci + 1) % 8 == 0:
                print(f"  traced chunk {ci + 1}/{n} done at {time.perf_counter() - t_run:.1f} s", flush=True)
        pool.shutdown()
        t0 = time.perf_counter()
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
