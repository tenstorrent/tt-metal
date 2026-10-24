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
from models.demos.blackhole.deepseek_v41_flash.tt import moe_overlap, pf_tune
from models.demos.blackhole.deepseek_v41_flash.tt import prefill_single as _PS
from models.demos.blackhole.deepseek_v41_flash.tt.h2d import h2d, recording, replay
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_attention import clear_chunk_caches, pad_len
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_layer import colsplit_active, unpack_streams
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_taps import TAP_LAYERS

T = 32
DIM = 5120
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


def engram_reduce_scatter(cat8, cc, free_after=()):
    """reduce_scatter over mesh axis 1 along dim 0 of the 8 stacked Engram outputs. DSV41_PFA_ENGRAM_RS_SPLIT=k (default 4 with DSV41_VISION=1, else 1 = one op) cuts the
    hidden dim into k pieces and reduces them one after the other (elementwise sum, same maths), so that the op's DRAM buffers need k times
    smaller contiguous blocks: inside the prefill trace capture of a fragmented 1M-context heap the single op's buffer does not fit.
    """
    k = int(os.environ.get("DSV41_PFA_ENGRAM_RS_SPLIT", "4" if os.environ.get("DSV41_VISION", "0") == "1" else "1"))

    def rs(t):
        return ttnn.experimental.reduce_scatter_minimal_async(
            t,
            dim=0,
            multi_device_global_semaphore=cc.get_rs_ping_pong_semaphore(),
            num_links=cc.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=cc.topology,
            cluster_axis=1,
            barrier_semaphore=cc.get_barrier_semaphore(),
        )

    if os.environ.get("DSV41_PFA_ENGRAM_RS_LOG") == "1":
        print(f"engram_reduce_scatter: cat8 {list(cat8.shape)} {cat8.dtype} {cat8.layout} split {k}", flush=True)
    if k <= 1:
        return rs(cat8)
    sh = list(cat8.shape)
    D = sh[3]
    assert D % k == 0 and (D // k) % 32 == 0, (D, k)
    w = D // k
    pieces = [ttnn.slice(cat8, [0, 0, 0, i * w], [sh[0], sh[1], sh[2], (i + 1) * w]) for i in range(k)]
    for t_ in (cat8, *free_after):
        if t_.is_allocated():
            ttnn.deallocate(t_)
    outs = []
    for p in pieces:
        outs.append(rs(p))
        ttnn.deallocate(p)
    out = ttnn.concat(outs, dim=3)
    for o in outs:
        ttnn.deallocate(o)
    return out


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
        self.vis = (
            os.environ.get("DSV41_VISION", "0") == "1"
        )  # image input: per-chunk embedding override / image mask buffers, bias_vl routing (colsplit + unified MoE only)
        self._vis_dirty, self._vctx = {}, None
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
        vb = None
        if self.vis:
            assert (
                self.cs
            ), "image input needs the column-split prefill (users per row x chunk a multiple of 256 tokens, grouped / unified MoE)"
            n_own = R // self.cols
            vb = (  # per device the rows of its own 32-token chunks (the layout of ``tok``): embedding override rows and the image mask
                ttnn.from_torch(
                    torch.zeros(rows * n_own, self.cols * DIM, dtype=torch.bfloat16),
                    device=self.md,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=self.shard_rc,
                ),
                ttnn.from_torch(
                    torch.zeros(rows * n_own, self.cols, dtype=torch.float32),
                    device=self.md,
                    dtype=ttnn.float32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=self.shard_rc,
                ),
            )
            self._vis_dirty[C] = False
        self._bufs[C] = (tok, er, vb)
        return self._bufs[C]

    def _own_layout(self, x):
        """[B, C, *tail] (user-major per mesh row) -> [rows * n8 * T, cols * prod(tail)]: the layout of the column-split ``tok`` upload (chunk 8g + c of a row -> column c)."""
        B, C = x.shape[:2]
        tail = x.shape[2:]
        w = int(torch.tensor(tail).prod()) if tail else 1
        n8 = self.U * C // T // self.cols
        x = x.reshape(self.rows, n8, self.cols, T, w).permute(0, 1, 3, 2, 4)  # [rows, n8, T, cols, w]
        return x.reshape(self.rows * n8 * T, self.cols * w).contiguous()

    def _host(self, t, dtype, layout):
        return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, mesh_mapper=self.shard)

    def prep_inputs(self, tokens, hashes=None, active=None, vis=None):
        """CPU part of a chunk's inputs (safe to run in a worker thread): tokens [B, C] (padded) and the Engram row gather -> host tensors.
        ``active`` (LongTensor of user indices, interleaved prefill): ``hashes`` then holds only those users ([len(active), C, ...]) and only their rows are
        gathered from the Engram tables (the other users are masked fillers: zero rows).
        ``vis`` (DSV41_VISION=1): None, or (image [B, C] bool, override [B, C, dim] bf16): the positions inside an image span take the override rows as embeddings, route with
        bias_vl and get no Engram contribution (their Engram rows are zero: the contribution is gate * value and value = wkv(0) = 0, exactly the checkpoint's masked gate).
        """
        B, C = tokens.shape
        rows, R = self.rows, self.U * C
        out = {"tok": tokens.reshape(rows * R, 1).to(torch.int32)}
        keep = None
        if self.vis:
            if vis is not None:
                img, ovr = vis
                out["vl"] = self._own_layout(img.float())
                out["ov"] = self._own_layout(ovr.to(torch.bfloat16))
                keep = (~img).reshape(B, C, 1).to(torch.bfloat16)
                self._vis_dirty[C] = True
            elif self._vis_dirty.get(C):  # a window without images after one with: clear the buffers once
                out["vl"] = torch.zeros(rows * (R // self.cols), self.cols)
                out["ov"] = torch.zeros(rows * (R // self.cols), self.cols * DIM, dtype=torch.bfloat16)
                self._vis_dirty[C] = False
        if active is not None:
            act = torch.as_tensor(active).long()
            for lid in self.engram:
                t1 = time.perf_counter()
                full = torch.zeros(B, C, self.engram[lid].kin, dtype=torch.bfloat16)
                if len(act):
                    full[act] = self.host_rows.rows(lid, hashes).to(torch.bfloat16)
                if keep is not None:
                    full = full * keep
                out[lid] = full.reshape(rows, 1, R, full.shape[-1])
                self.timing["host_rows_gather_dequant"] = (
                    self.timing.get("host_rows_gather_dequant", 0.0) + time.perf_counter() - t1
                )
            return out
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
            r = r.to(torch.bfloat16) if keep is None else r.to(torch.bfloat16) * keep
            out[lid] = r.reshape(rows, 1, R, r.shape[-1])
            self.timing["host_rows_gather_dequant"] = self.timing.get("host_rows_gather_dequant", 0.0) + t2 - t1
        return out

    def upload_inputs(self, prepped, bufs):
        t0 = time.perf_counter()
        tok, er, vb = bufs
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
        if "ov" in prepped:
            ov_dev, vl_dev = vb
            h2d(
                ttnn.from_torch(
                    prepped["ov"], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.shard_rc
                ),
                ov_dev,
            )
            h2d(
                ttnn.from_torch(
                    prepped["vl"], dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.shard_rc
                ),
                vl_dev,
            )
        self.timing["host_upload"] = self.timing.get("host_upload", 0.0) + time.perf_counter() - t0

    def write_inputs(self, tokens, hashes=None, bufs=None):
        t0 = time.perf_counter()
        pre = self.prep_inputs(tokens, hashes)
        self.timing["host_gather"] = self.timing.get("host_gather", 0.0) + time.perf_counter() - t0
        self.upload_inputs(pre, bufs or self.alloc_inputs(tokens.shape[1]))

    def _vision_ctx(self, vb, C):
        """Per-forward tensors of the image input (DSV41_VISION=1) from the persistent buffers [rows*n_own, cols*dim] / [rows*n_own, cols] (per device the own rows): the
        embedding override rows (tile bf16 [1,1,n,D]), keep = 1 - image mask and the image mask itself (tile fp32 [1,1,n,1], the routers' ``vl_rows``).
        """
        ov_dev, vl_dev = vb
        n_own = self.U * C // self.cols
        ov = ttnn.reshape(ttnn.to_layout(ov_dev, ttnn.TILE_LAYOUT), [1, 1, n_own, DIM])
        vl = ttnn.reshape(ttnn.to_layout(vl_dev, ttnn.TILE_LAYOUT), [1, 1, n_own, 1])
        return {"ov": ov, "vl": vl, "keep": ttnn.rsub(vl, 1.0)}

    def forward_device(self, bufs, S, s0, C, hook=None, profile=False, dyn=False):
        vb = bufs[2]
        if vb is None:
            return self._forward_device(bufs, S, s0, C, hook, profile, dyn)
        self._vctx = self._vision_ctx(vb, C)
        for _, pl in self.layers:
            pl.vl_rows = self._vctx["vl"]
        try:
            return self._forward_device(bufs, S, s0, C, hook, profile, dyn)
        finally:
            for _, pl in self.layers:
                pl.vl_rows = None
            for t in self._vctx.values():
                ttnn.deallocate(t)
            self._vctx = None

    def _forward_device(self, bufs, S, s0, C, hook=None, profile=False, dyn=False):
        """One prompt chunk (positions [s0, s0+C) of every user): embedding -> layers (+Engram) -> (last chunk only) head on the chunks holding the
        last real token of each user. Only enqueues device ops (traceable when ``hook`` / ``profile`` are off). Returns {local user: logits shard
        [1,1,1,vocab/cols] of its last token} for the last chunk, else None."""
        tok_dev, erows_dev, _ = bufs
        last = s0 + C >= S
        sink = getattr(self, "sink", None)
        taps = getattr(sink, "taps", None) if dyn else None  # (the traced / dynamic chunk of the hand-off model only)
        if (
            dyn
        ):  # traced-chunk mode: the masks are built on the device from the per-chunk tensors, the outputs stay in ``self.dyn_out``
            self.dyn.build_masks()

        def sync(name, t0):
            if profile:
                self.sync(name, t0)

        t0 = time.perf_counter()
        n32 = self.U * C // T // (self.cols if self.cs else 1)
        vc = self._vctx
        xs = [
            self.embedding.forward(
                ttnn.slice(tok_dev, [c * T, 0], [(c + 1) * T, 1]),
                *(
                    ()
                    if vc is None
                    else (
                        ttnn.slice(vc["ov"], [0, 0, c * T, 0], [1, 1, (c + 1) * T, DIM]),
                        ttnn.slice(vc["keep"], [0, 0, c * T, 0], [1, 1, (c + 1) * T, 1]),
                    )
                ),
            )[0]
            for c in range(n32)
        ]
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
                            pf_tune._env("DSV41_PFA_ENGRAM_BATCH", "0") == "1" and fe.__name__ == "forward_v2"
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
                        rs8 = engram_reduce_scatter(cat8, cc, outs8)
                        new.append(ttnn.multiply(rs8, 1.0 / self.cols))
                        for t_ in (xg, rs8):
                            ttnn.deallocate(t_)
                        for t_ in (cat8, *outs8):
                            if t_.is_allocated():
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
            if taps is not None and lid in TAP_LAYERS:
                # drafter taps (tt/prefill_taps.py): mean over the 4 streams at the input of layers 37 / 38 / 39 (after the Engram), last 128 positions of every user -> persistent stash
                taps.write(TAP_LAYERS.index(lid), xs, pl.L.mesh_config, pl.L.ccl, self.cs, self.sink.bufs[C]["tap"])
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
        self.dyn = DynCtx(self.md, C, S_pad, ratios, rope_src, per_row=_PS.per_row_enabled(self.U))
        # interleaved prefill (DSV41_PF_UMASK=1): per-user 0/1 mask [U,1,1,1] (per mesh row, persistent, refreshed before every replay) that gates every write of
        # the carried per-user prefill state (halo, latent / index-key FIFOs) INSIDE the trace, so users that are not part of a replay keep their state
        self.dyn.umask = None
        if os.environ.get("DSV41_PF_UMASK", "0") == "1":
            self.dyn.umask = ttnn.from_torch(
                torch.ones(self.rows * self.U, 1, 1, 1),
                device=self.md,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.shard,
            )
        for _, pl in self.layers:
            pl.pa.alloc_dyn(self.dyn)
        self.dyn_trace = None
        self.pre_replay_hooks = getattr(self, "pre_replay_hooks", [])
        self.post_replay_hooks = getattr(self, "post_replay_hooks", [])

    # ---- several S_pad contexts (small-prompt buckets): each owns its DynCtx, per-layer carried tables and chunk trace; one is ACTIVE (the attributes the forward reads), the others are stashed
    _PA_ATTRS = ("dyn", "halo", "lat_buf", "_lat_pre")
    _SP_ATTRS = ("ctx", "dyn_on", "kvt", "ids")
    _IX_ATTRS = ("keys", "keys_rm", "KL", "KL_full", "sh")
    _PM_ATTRS = ("dyn", "dyn_trace", "dyn_out", "head_out", "_dyn_compiled")
    _PA_FRESH = {"dyn": None, "halo": None, "lat_buf": None, "_lat_pre": None}
    _SP_FRESH = {"ctx": None, "dyn_on": False, "kvt": None, "ids": None}
    _IX_FRESH = {"keys": None, "keys_rm": None, "sh": None}

    def _dyn_objs(self):
        pas, sps, ixs, seen = [], [], [], set()
        for _, pl in self.layers:
            pa = pl.pa
            sp = pa.sparse
            ix = None if sp is None else sp.indexer
            for lst, o in ((pas, pa), (sps, sp), (ixs, ix)):
                if o is not None and id(o) not in seen:
                    seen.add(id(o))
                    lst.append(o)
        return pas, sps, ixs

    @staticmethod
    def _grab(o, names):
        return {n: o.__dict__[n] for n in names if n in o.__dict__}

    @staticmethod
    def _put(o, names, vals):
        for n in names:
            o.__dict__.pop(n, None)
        o.__dict__.update(vals)

    def _dyn_save(self):
        pas, sps, ixs = self._dyn_objs()
        return {
            "pm": self._grab(self, self._PM_ATTRS),
            "pa": [self._grab(o, self._PA_ATTRS) for o in pas],
            "sp": [self._grab(o, self._SP_ATTRS) for o in sps],
            "ix": [self._grab(o, self._IX_ATTRS) for o in ixs],
        }

    def _dyn_load(self, st):
        pas, sps, ixs = self._dyn_objs()
        self._put(self, self._PM_ATTRS, st["pm"])
        for o, v in zip(pas, st["pa"]):
            self._put(o, self._PA_ATTRS, v)
        for o, v in zip(sps, st["sp"]):
            self._put(o, self._SP_ATTRS, v)
        for o, v in zip(ixs, st["ix"]):
            self._put(o, self._IX_ATTRS, v)

    def _dyn_blank(self):
        """the active context becomes empty WITHOUT freeing anything (its owner keeps the references in a stash)"""
        pas, sps, ixs = self._dyn_objs()
        self._put(
            self,
            self._PM_ATTRS,
            {"dyn": None, "dyn_trace": None, "dyn_out": None, "head_out": [], "_dyn_compiled": None},
        )
        for o in pas:
            self._put(o, self._PA_ATTRS, dict(self._PA_FRESH))
        for o in sps:
            self._put(o, self._SP_ATTRS, dict(self._SP_FRESH))
        for o in ixs:
            self._put(o, ("keys", "keys_rm", "sh"), dict(self._IX_FRESH))

    @property
    def dyn_stash(self):
        return self.__dict__.setdefault("_dyn_stash", {})

    def stash_dyn(self):
        """move the active context aside (kept alive, not freed)"""
        if self.dyn is None:
            return
        self.dyn_stash[self.dyn.S_pad] = self._dyn_save()
        self._dyn_blank()

    def use_dyn(self, S_pad):
        """make the context of ``S_pad`` the active one (a no-op when it is)"""
        if self.dyn is not None and self.dyn.S_pad == S_pad:
            return
        st = self.dyn_stash.pop(S_pad)
        self.stash_dyn()
        self._dyn_load(st)

    def dyn_buckets(self):
        """S_pads of every context holding a CAPTURED chunk trace (active + stashed), ascending"""
        out = [] if self.dyn is None or self.dyn_trace is None else [self.dyn.S_pad]
        out += [k for k, st in self.dyn_stash.items() if st["pm"].get("dyn_trace") is not None]
        return sorted(out)

    def teardown_dyn(self):
        """Release the chunk trace(s) and the per-chunk buffers of EVERY context (a different chunk size / padded prompt length needs a new capture)."""
        while self.dyn_stash:
            self.stash_dyn()  # (the active one keeps living until its turn)
            k = next(iter(self.dyn_stash))
            self._dyn_load(self.dyn_stash.pop(k))
            self._teardown_active()
        self._teardown_active()

    def _teardown_active(self):
        if getattr(self, "dyn_trace", None) is not None:
            if isinstance(self.dyn_trace, moe_overlap.SegTrace):
                self.dyn_trace.release()
            else:
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
        self.dyn.cur_epoch = getattr(self, "_win_count", 0)
        act = getattr(self, "active_mask", None)
        self.dyn.active = (
            None if act is None else torch.as_tensor(act).bool()
        )  # (single-trace key append: users not in this replay write the scratch rows)
        self.dyn.update(s0)
        t1 = time.perf_counter()
        if getattr(self.dyn, "umask", None) is not None:
            act = getattr(self, "active_mask", None)
            m = torch.ones(self.rows * self.U) if act is None else torch.as_tensor(act).float()
            h2d(  # (recordable: see ``record_window``)
                ttnn.from_torch(
                    m.reshape(-1, 1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=self.shard
                ),
                self.dyn.umask,
            )
        for h in self.pre_replay_hooks:
            h(s0, C)
        t2 = time.perf_counter()
        self.timing["w_dynupd"] = self.timing.get("w_dynupd", 0.0) + t1 - t0
        self.timing["w_prehooks"] = self.timing.get("w_prehooks", 0.0) + t2 - t1

    # ---- interleaved prefill: capture on dummy inputs, then replay one window (s0) at a time -------------------------------------------------
    def dummy_inputs(self, C):
        out = {"tok": torch.zeros(self.rows * self.U * C, 1, dtype=torch.int32)}
        for lid, e in self.engram.items():
            out[lid] = torch.zeros(self.rows, 1, self.U * C, e.kin, dtype=torch.bfloat16)
        return out

    def capture_dyn(self, C, S_pad):
        """Compile + capture the chunk trace for (C, S_pad) on DUMMY inputs with every user masked (no real state is written), so that real windows are only ever
        replayed (``replay_window``). Returns True when a (re)capture happened (the caller must drop everything that depends on the carried state).
        """
        if not self.compile_dyn(C, S_pad):
            return False
        self.capture_dyn_trace(C)
        return True

    def compile_dyn(self, C, S_pad):
        """First half of ``capture_dyn``: allocate every per-chunk persistent buffer and run the eager compile pass (no trace is captured). Split from the capture so that a server can
        compile EVERY program (prefill chunk, all decode buckets) before capturing ANY trace. -> False when the (C, S_pad) trace is already captured.
        """
        assert C % 128 == 0
        if self.dyn is not None and self.dyn.C == C and self.dyn.S_pad == S_pad and self.dyn_trace is not None:
            return False
        self.teardown_dyn()
        self._compile_dyn_body(C, S_pad)
        return True

    def compile_dyn_extra(self, C, S_pad, after_setup=None):
        """Allocate + compile one ADDITIONAL (smaller) S_pad context next to the active one (kept: the active context is stashed meanwhile and active again afterwards). The new context is left
        stashed with its trace not captured: ``use_dyn(S_pad)`` + ``capture_dyn_trace(C)`` captures it (every persistent buffer of every context must exist before the first capture).
        """
        assert C % 128 == 0 and self.dyn is not None and S_pad not in self.dyn_stash and S_pad != self.dyn.S_pad
        cur = self.dyn.S_pad
        self.stash_dyn()
        try:
            self._compile_dyn_body(C, S_pad, after_setup)
            self.stash_dyn()
        finally:
            if self.dyn is None:
                self.use_dyn(cur)
        self.use_dyn(cur)

    def _compile_dyn_body(self, C, S_pad, after_setup=None):
        assert (
            type(self).last_logits is not DSV41PrefillModel.last_logits
        ), "interleaved prefill needs the ragged hand-off head (GenPrefillModel)"
        t0 = time.perf_counter()
        self.setup_dyn(C, S_pad)
        if after_setup is not None:
            after_setup()  # (the tables of this context exist: allocations that must not split their blocks come now)
        bufs = self.alloc_inputs(C)
        self.active_mask = torch.zeros(self.rows * self.U)
        self.upload_inputs(self.dummy_inputs(C), bufs)
        self.begin_chunk(0, C)
        self.forward_device(bufs, C, 0, C, dyn=True)  # compile pass (eager)
        ttnn.synchronize_device(self.md)
        xs0, pres0 = self.dyn_out
        for x in list(xs0) + list(pres0):
            ttnn.deallocate(x)
        for _, pl in self.layers:
            pl.pa.reset_dyn()  # before any capture only
        ttnn.synchronize_device(self.md)
        self._dyn_compiled = (C, bufs, t0)

    def _trace_region_mib(self):
        try:
            tv = ttnn.get_memory_view(self.md, ttnn.BufferType.TRACE)
            return tv.total_bytes_allocated_per_bank / 2**20
        except Exception:  # noqa: BLE001
            return float("nan")

    def _log_chunk_trace(self, C, before):
        """PF_TRACES line: how many prefill chunk traces are held and what this capture added to the trace region (MiB per DRAM bank; x 8 banks... = per chip)."""
        after = self._trace_region_mib()
        print(
            f"PF_TRACES prefill chunk trace captured: C={C} S_pad={self.dyn.S_pad} single={os.environ.get('DSV41_PF_SINGLE', '0')} -> chunk traces held {len(self.dyn_buckets())} "
            f"(S_pads {self.dyn_buckets()}); this capture +{after - before:.1f} MiB/bank, trace region now {after:.1f} MiB/bank",
            flush=True,
        )

    def _warm_gather_plans(self, C):
        """``DSV41PrefillIndexer.warm_gather`` for every distinct score width of the column-sharded indexers of the active context; right before a chunk-trace capture (DSV41_PF_WARM_GATHER=0 disables it)."""
        if os.environ.get("DSV41_PF_WARM_GATHER", "1") != "1":
            return
        seen = set()
        for ix in self._dyn_objs()[2]:
            if ix.sh is not None and ix.sh["Lc"] not in seen:
                seen.add(ix.sh["Lc"])
                ix.warm_gather(C)

    def capture_dyn_trace(self, C):
        """Second half of ``capture_dyn``: capture the chunk trace (everything is compiled and allocated)."""
        C_, bufs, t0 = self._dyn_compiled
        assert C_ == C
        from models.demos.blackhole.deepseek_v41_flash.tt.decode_buckets import corruptible

        if os.environ.get("DSV41_PF_PRECAPTURE_WARM", "1") == "1":
            # a server compiles the decode programs between ``compile_dyn`` and this capture: a program of the chunk whose hash changed meanwhile (seen with the column-sharded indexer at a 1M
            # context: "Cannot load new binaries during trace capture" in select_dyn_sharded) would be compiled INSIDE the capture. One more eager chunk right before it (same masked dummy
            # chunk as the compile pass, same allocation pattern) leaves nothing to compile.
            self.begin_chunk(0, C)
            self.forward_device(bufs, C, 0, C, dyn=True)
            ttnn.synchronize_device(self.md)
            xs0, pres0 = self.dyn_out
            for x in list(xs0) + list(pres0):
                ttnn.deallocate(x)
            for _, pl in self.layers:
                pl.pa.reset_dyn()
            ttnn.synchronize_device(self.md)
        tr0 = self._trace_region_mib()
        self._warm_gather_plans(C)
        self.dyn_trace = ttnn.begin_trace_capture(self.md, cq_id=0)
        _PS.CAPTURING[0] = True
        with corruptible(self.md):
            self.forward_device(bufs, C, 0, C, dyn=True)
        _PS.CAPTURING[0] = False
        ttnn.end_trace_capture(self.md, self.dyn_trace, cq_id=0)
        ttnn.synchronize_device(self.md)
        self._log_chunk_trace(C, tr0)
        self._dyn_compiled = None
        self.timing["compile_and_capture"] = time.perf_counter() - t0

    def record_window(self, prepped, s0, C, active):
        prof = self.__dict__.get(
            "build_prof"
        )  # DSV41_PF_PROFILE_BUILD=1 (diagnostics): cProfile of the host work of a window
        if prof is not None:
            prof.enable()
            try:
                return self._record_window(prepped, s0, C, active)
            finally:
                prof.disable()
        return self._record_window(prepped, s0, C, active)

    def _record_window(self, prepped, s0, C, active):
        """All the host work of the uploads of the window [s0, s0 + C) (input tensors, RoPE tables, masks, hand-off index tables: ``upload_inputs`` + ``begin_chunk``) with the device copies only RECORDED -> ops for
        ``replay_window(ops=...)``. Runs while the device replays the previous window (the demo's ``prep_rec``)."""
        self.active_mask = torch.as_tensor(active).bool()
        with recording() as ops:
            self.upload_inputs(prepped, self._bufs[C])
            self.begin_chunk(s0, C)
        return ops

    def record_upload(self, prepped, C):
        """The token-dependent half of ``record_window`` alone (Engram rows / tokens uploads): recorded ahead of the call that uses it (``lookahead``)."""
        with recording() as ops:
            self.upload_inputs(prepped, self._bufs[C])
        return ops

    def record_begin(self, s0, C, active):
        """The position / user dependent half of ``record_window`` alone (RoPE tables, masks, hand-off index tables: ``begin_chunk``)."""
        self.active_mask = torch.as_tensor(active).bool()
        with recording() as ops:
            self.begin_chunk(s0, C)
        return ops

    def replay_window(self, tokens, hashes, s0, C, active, vis=None, prepped=None, overlap=None, ops=None, sync=True):
        """One replay of the captured chunk trace for the window of positions [s0, s0 + C): tokens [B, C] (fillers 0), hashes of the active users only,
        ``active`` bool [B]; ``vis`` (DSV41_VISION=1): (image mask [B, C] bool, embedding override [B, C, dim] bf16) or None. Returns (host seconds, device seconds).
        ``sync=False`` (the demo's asynchronous loop, ``run_traced_chunks``): the copies of the window and its trace are only ENQUEUED (CQ 0 is in order: the copies land after the previous replay,
        the next window's copies and trace can be enqueued behind this one), no device synchronisation and no post hook; the host runs at most one replay ahead (the event of the replay before the previous
        one is awaited GIL-free before this window's copies are enqueued). The caller synchronises at the windows whose result it reads (``sync=True``: a user's last token, a snapshot boundary, the end of the call).
        """
        t0 = time.perf_counter()
        self._win_count = (
            getattr(self, "_win_count", 0) + 1
        )  # (epoch: a speculative upload is only valid while no other window was enqueued since)
        wl = self.__dict__.get(
            "winlog"
        )  # DSV41_PF_WINLOG=1 (diagnostics): per-window host timestamps + the time the device finished it (a watcher thread awaits its event)
        w = None if wl is None else {"s0": s0, "t_in": t0}
        hist = self.__dict__.setdefault("_ev_hist", [])
        if len(hist) >= 2:
            ttnn.event_synchronize(hist[-2])
        if ops is not None:  # recorded ahead (``record_window``): only the device copies are left
            replay(ops)
        else:
            prof = self.__dict__.get("build_prof")
            if prof is not None:
                prof.enable()
            self.active_mask = torch.as_tensor(active).bool()
            act = self.active_mask.nonzero().reshape(-1)
            self.upload_inputs(
                prepped if prepped is not None else self.prep_inputs(tokens, hashes, act, vis=vis), self._bufs[C]
            )
            self.begin_chunk(s0, C)
            if prof is not None:
                prof.disable()
        t1 = time.perf_counter()
        from models.demos.blackhole.deepseek_v41_flash.tt.decode_buckets import check_trace_allocations

        check_trace_allocations(self.md, self.dyn_trace, "prefill chunk")
        ttnn.execute_trace(self.md, self.dyn_trace, cq_id=0, blocking=False)
        hist.append(ttnn.record_event(self.md, 0))
        del hist[:-2]
        if w is not None:
            w["t_copied"], w["t_enq"] = t1, time.perf_counter()
            wl["windows"].append(w)
            wl["q"].put((w, hist[-1]))
        if (
            overlap is not None
        ):  # host work of the NEXT window (Engram hash + rows gather / dequant) runs while the device replays this one
            to0 = time.perf_counter()
            overlap()
            self.timing["host_overlapped"] = self.timing.get("host_overlapped", 0.0) + time.perf_counter() - to0
        t_en = time.perf_counter()
        c0 = self.__dict__.get("_call_t0")
        if c0 is not None:  # (call timeline for the server log: when the first trace of the call was enqueued)
            self.timing.setdefault("first_enq_rel", t_en - c0)
        if not sync:
            self.timing["host_per_chunk"] = self.timing.get("host_per_chunk", 0.0) + t1 - t0
            self.timing["async_windows"] = self.timing.get("async_windows", 0) + 1
            return t1 - t0, 0.0
        ttnn.synchronize_device(self.md)
        hist.clear()
        t2 = time.perf_counter()
        if c0 is not None:
            self.timing["sync_done_rel"] = t2 - c0  # (last synchronised window: the device finished it)
        t_h0 = time.perf_counter()
        if w is not None:
            w["sync"] = True
        for hk in self.post_replay_hooks:
            hk(s0, C)
        self.timing["post_hooks"] = self.timing.get("post_hooks", 0.0) + time.perf_counter() - t_h0
        self.timing["host_per_chunk"] = self.timing.get("host_per_chunk", 0.0) + t1 - t0
        self.timing["replay_per_chunk"] = self.timing.get("replay_per_chunk", 0.0) + t2 - t1
        return t1 - t0, t2 - t1

    def winlog_start(self):
        """Start the per-window timeline log (see ``replay_window``): a watcher thread stamps the completion time of every enqueued window."""
        import queue
        import threading

        q = queue.Queue()
        self.winlog = {"windows": [], "q": q, "calls": []}

        def watch():
            while True:
                item = q.get()
                if item is None:
                    q.task_done()
                    return
                w, ev = item
                ttnn.event_synchronize(ev)
                w["t_done"] = time.perf_counter()
                q.task_done()

        self._wl_thread = threading.Thread(target=watch, daemon=True)
        self._wl_thread.start()

    def winlog_stop(self):
        wl = self.__dict__.pop("winlog", None)
        if wl is not None:
            wl["q"].put(None)
            self._wl_thread.join()
        return wl

    def winlog_drain(self):
        """wait until every enqueued window has its completion stamp"""
        wl = self.__dict__.get("winlog")
        if wl is not None:
            wl["q"].join()

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
            self._warm_gather_plans(C)
            if (
                moe_overlap.enabled()
            ):  # the shared expert / dispatch overlap switches sub-device managers: segmented trace
                ov = moe_overlap.SDOverlap.get(self.md)
                self.dyn_trace = moe_overlap.SegTrace(self.md)
                ov.seg = self.dyn_trace
                self.dyn_trace.begin()
                self.forward_device(bufs, S, 0, C, dyn=True)
                self.dyn_trace.end()
                ov.seg = None
            else:
                tr0 = self._trace_region_mib()
                self.dyn_trace = ttnn.begin_trace_capture(self.md, cq_id=0)
                _PS.CAPTURING[0] = True
                self.forward_device(bufs, S, 0, C, dyn=True)
                _PS.CAPTURING[0] = False
                ttnn.end_trace_capture(self.md, self.dyn_trace, cq_id=0)
                ttnn.synchronize_device(self.md)
                self._log_chunk_trace(C, tr0)
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
            if isinstance(self.dyn_trace, moe_overlap.SegTrace):
                self.dyn_trace.replay()
            else:
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
