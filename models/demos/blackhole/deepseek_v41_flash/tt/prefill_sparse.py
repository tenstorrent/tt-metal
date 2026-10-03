# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PREFILL compressed-sparse attention (CSA) of DeepSeek-V4.1-Flash: lightning indexer + ``sparse_sdpa`` for the chunked prefill.

The checkpoint's attention of a compressed layer attends, per query, the 128-token window plus the ``index_topk`` = 512 compressed entries the
indexer of the layer's index source ranks highest (all of them while fewer than 512 are visible). ``DSV41PrefillAttention`` (dense SDPA over every
compressed entry) is exact only up to 512 visible entries; this module adds the selection for longer prompts. Per chunk of queries ``[s0, s0 + C)``:

  1. ``ingest``  : kv-source layers write the chunk's RoPE'd latents into the row-major kv table ``PrefillKV`` (entry j at row j) and index-key owners
                   (layers 2, 8, 14, 20) append the chunk's index keys (``k_norm(wk(latent_pre_rope))``, RoPE, fp4 simulation) to the key slab.
  2. ``select``  : index-source layers (2, 8, 14, 20, 24, 28, 32, 36) score the chunk's queries against ALL closed groups with ``indexer_score_dsa``
                   (multi-query, causal through ``chunk_start_idx``) and take the top 512 with ``topk_large_indices`` (-inf -> 0xFFFFFFFF); the other layers
                   reuse the ids of the last index source before them (like the checkpoint's ``shared_attn.topk_idxs``).
  3. ``attend``  : window rows (halo + chunk kv) are copied into the scratch rows of the kv table, index rows ``[window 128 | selected 512]`` are built
                   and ``ttnn.transformer.sparse_sdpa`` runs once per user (``cache_batch_idx``), 32 padded heads (8 real).

Ratio-2 layers: a query at token t sees entry j iff j < (t + 1) // 2, which ``indexer_score_dsa`` (key t visible iff t <= chunk_start + row) cannot express
in token units. The even / odd queries of the chunk are scored as two separate half-length query sets in ENTRY units (row m of the odd set sees j <= s0/2 + m
-> chunk_start = s0/2; the even set is shifted down one row by 31 zero rows in front and chunk_start = s0/2 - 32, which needs s0 >= 64).
"""

import os

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.attention import HEAD_DIM, LOCAL_HEADS, WINDOW
from models.demos.blackhole.deepseek_v41_flash.tt.indexer import DIM as IDIM
from models.demos.blackhole.deepseek_v41_flash.tt.indexer import HEADS as IHEADS
from models.demos.blackhole.deepseek_v41_flash.tt.indexer import TOPK, DSV41DecodeIndexer

SKIP = 0xFFFFFFFF
PAD_HEADS = 32
_CONST = {}


def clear_sparse_caches():
    for v in _CONST.values():
        for t in v if isinstance(v, tuple) else (v,):
            ttnn.deallocate(t)
    _CONST.clear()


def ceil32(n):
    return -(-n // 32) * 32


def _rep(md, t, dtype, layout):
    return ttnn.from_torch(
        t.contiguous(),
        device=md,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )


class PrefillKV:
    """Row-major kv table of one kv-source group (shared by the source layer and the layers that read its latents):
    ``[U, 1, LAT + WIN, 512]`` bf16, rows [0, LAT): compressed latents (entry j at row j), rows [LAT, LAT + WIN): window scratch (WIN = 128 + c_max rows:
    the 128 positions before the chunk, then the chunk), rewritten by every layer for every chunk."""

    def __init__(self, md, users, lat_cap, c_max):
        self.md, self.U = md, users
        self.LAT = ceil32(lat_cap)
        self.WIN = WINDOW + c_max
        self.t = ttnn.zeros(
            [users, 1, self.LAT + self.WIN, HEAD_DIM], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=md
        )

    def free(self):
        ttnn.deallocate(self.t)

    def write_latents(self, lat, j0):
        """lat [U,1,Cc,512] tile bf16 (RoPE'd)."""
        Cc = lat.shape[2]
        ttnn.experimental.slice_write(
            ttnn.to_layout(lat, ttnn.ROW_MAJOR_LAYOUT),
            self.t,
            [0, 0, j0, 0],
            [self.U, 1, j0 + Cc, HEAD_DIM],
            [1, 1, 1, 1],
        )

    def write_window(self, kv, halo, s0):
        """kv [U,1,C,512] chunk rows (post RoPE), halo [U,1,128,512] or None (first chunk): scratch row of position p is LAT + p - (s0 - 128)."""
        C = kv.shape[2]
        full = kv if halo is None else ttnn.concat([halo, kv], dim=2)
        base = self.LAT + (WINDOW if halo is None else 0)
        ttnn.experimental.slice_write(
            ttnn.to_layout(full, ttnn.ROW_MAJOR_LAYOUT),
            self.t,
            [0, 0, base, 0],
            [self.U, 1, base + full.shape[2], HEAD_DIM],
            [1, 1, 1, 1],
        )
        if full is not kv:
            ttnn.deallocate(full)


def _itabs(pa, start, n, step):
    """(cos, sin) [1,1,n,128] tables of the indexer (last 128 columns of the attention's 512-wide tables: 1 / 0 outside the rotated dims)."""
    key = ("itab", id(pa.md), start, n, step)
    if key not in _CONST:
        c, s, _ = pa.rope_tabs(start, n, step)
        sl = lambda t: ttnn.slice(t, [0, 0, 0, HEAD_DIM - IDIM], [1, 1, n, HEAD_DIM])
        _CONST[key] = (sl(c), sl(s))
    return _CONST[key]


class DSV41PrefillIndexer:
    """The indexer of ONE index-source layer for prefill chunks. Shares the device weights of a ``DSV41DecodeIndexer`` (``dec``; built here from ``w_idx``
    when not given). Index-key owners (``keys`` is None) allocate the key slab ``[U,1,KL,128]``; the other index sources score against ``key_owner.keys``.
    """

    def __init__(
        self, md, users, ratio, lmax, dec=None, w_idx=None, freqs_cis=None, key_owner=None, fp4_q=True, fp4_k=True
    ):
        self.md, self.U, self.ratio = md, users, ratio
        if dec is None:
            dec = DSV41DecodeIndexer(
                md, w_idx, freqs_cis, users_per_row=users, n_alloc=32, ratio=ratio, backend="matmul", fp4_q=fp4_q
            )
            if "wk" in w_idx:
                dec.set_key_weights(w_idx["wk"], w_idx["k_norm"])
        self.dec = dec
        self.fp4_q, self.fp4_k = fp4_q, fp4_k
        self.key_owner = key_owner
        self.KL = ceil32(lmax)
        self.keys = None
        if key_owner is None:
            self.keys = ttnn.zeros([users, 1, self.KL, IDIM], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md)
        self.cfg = ttnn.IndexerScoreProgramConfig(
            q_chunk_size=int(os.environ.get("DSV41_PIDX_QC", "32")),
            k_chunk_size=int(os.environ.get("DSV41_PIDX_KC", "128")),
            head_group_size=int(os.environ.get("DSV41_PIDX_HG", "0")),
        )
        self.ckc = dec.ckc

    @property
    def slab(self):
        return self.keys if self.key_owner is None else self.key_owner.keys

    # ---- numerics shared with the decode indexer -----------------------------------------------------------------------------
    def _fp4(self, x):
        """fp4 (e2m1 / per-32 e8m0) quantise-dequantise of a bf16 tile tensor [..., 128]."""
        shp = list(x.shape)
        n = 1
        for s in shp[:-1]:
            n *= s
        f = ttnn.reshape(ttnn.typecast(x, ttnn.float32), (1, 1, n * (IDIM // 32), 32))
        y = DSV41DecodeIndexer._fp4_blocks(f)
        return ttnn.typecast(ttnn.reshape(y, shp), ttnn.bfloat16)

    def _rope(self, x, tab):
        c, s = tab
        return ttnn.addcmul(ttnn.multiply(x, c), ttnn.matmul(x, self.dec.P, compute_kernel_config=self.ckc), s)

    # ---- keys --------------------------------------------------------------------------------------------------------------------
    def add_keys(self, pa, lat_pre, s0):
        """lat_pre [U,1,Cc,512] bf16 tile: the pooled latents of this chunk BEFORE RoPE -> index keys appended to the slab at entries [s0/r, s0/r + Cc)."""
        d, U, r = self.dec, self.U, self.ratio
        Cc = lat_pre.shape[2]
        j0 = s0 // r
        x = ttnn.reshape(lat_pre, [1, 1, U * Cc, HEAD_DIM])
        k = ttnn.linear(x, d.wk, compute_kernel_config=self.ckc)
        k = ttnn.rms_norm(ttnn.typecast(k, ttnn.bfloat16), weight=d.k_norm_w, epsilon=d.k_eps)
        k = ttnn.reshape(k, [U, 1, Cc, IDIM])
        k = self._rope(k, _itabs(pa, s0, Cc, r))
        if self.fp4_k:
            k = self._fp4(k)
        ttnn.experimental.slice_write(k, self.keys, [0, 0, j0, 0], [U, 1, j0 + Cc, IDIM], [1, 1, 1, 1])
        ttnn.deallocate(k)

    def export_keys(self, k_cache, n_entries):
        """Hand-off to decode: copy the keys of entries [0, ceil32(n_entries)) of the key slab into the decode key slab ``k_cache`` ([U,1,n_alloc,128] tile, bf16 / bfp8_b;
        ``DSV41DecodeIndexer.k_cache``). Entries >= the user's S // ratio are garbage of the padded tail: hidden by the decode valid length, rewritten by ``append_key``.
        """
        assert (
            self.key_owner is None
        ), "export from the key owner (layers 2, 8, 14, 20); layers 24-36 alias layer 20's slab"
        n = ceil32(n_entries)
        assert n <= k_cache.shape[2] and n <= self.KL
        src = ttnn.slice(self.keys, [0, 0, 0, 0], [self.U, 1, n, IDIM])
        if k_cache.dtype != ttnn.bfloat16:
            src = ttnn.typecast(src, k_cache.dtype)
        ttnn.experimental.slice_write(src, k_cache, [0, 0, 0, 0], [self.U, 1, n, IDIM], [1, 1, 1, 1])
        ttnn.deallocate(src)

    # ---- selection ---------------------------------------------------------------------------------------------------------------
    def select(self, pa, h, qr, s0, C):
        """-> ids [U,1,C,512] uint32 RM: the 512 entries (descending score, 0xFFFFFFFF where fewer are visible) every query of the chunk selects.
        h [1,1,U*C,5120] attention input, qr [1,1,U*C,1280] normed q-lora (bf16 tiles)."""
        d, U, r = self.dec, self.U, self.ratio
        R = U * C
        L_end = (s0 + C) // r
        assert L_end >= TOPK and L_end <= self.KL and L_end % 32 == 0
        w = ttnn.to_layout(ttnn.linear(h, d.wproj, compute_kernel_config=self.ckc), ttnn.ROW_MAJOR_LAYOUT)  # [1,1,R,32]
        q = ttnn.to_layout(
            ttnn.linear(qr, d.wq_b, compute_kernel_config=self.ckc), ttnn.ROW_MAJOR_LAYOUT
        )  # [1,1,R,4096]
        if r == 1:
            ids = self._part(pa, q, w, C, s0, 1, s0, 0, 0, L_end)
        else:
            n = C // 2
            qq = ttnn.reshape(q, [1, 1, R // 2, 2 * IHEADS * IDIM])
            ww = ttnn.reshape(w, [1, 1, R // 2, 2 * IHEADS])
            q_e = ttnn.slice(qq, [0, 0, 0, 0], [1, 1, R // 2, IHEADS * IDIM])
            q_o = ttnn.slice(qq, [0, 0, 0, IHEADS * IDIM], [1, 1, R // 2, 2 * IHEADS * IDIM])
            w_e = ttnn.slice(ww, [0, 0, 0, 0], [1, 1, R // 2, IHEADS])
            w_o = ttnn.slice(ww, [0, 0, 0, IHEADS], [1, 1, R // 2, 2 * IHEADS])
            assert s0 >= 64, "ratio-2 selection of the first chunk needs a first chunk of <= 1024 tokens (dense path)"
            ids_o = self._part(pa, q_o, w_o, n, s0 + 1, 2, s0 // 2, 0, 0, L_end)
            ids_e = self._part(pa, q_e, w_e, n, s0, 2, s0 // 2 - 32, 31, 1, L_end)
            # interleave: row 2m = even set m, row 2m+1 = odd set m
            e = ttnn.reshape(ids_e, [U * n, 1, TOPK])
            o = ttnn.reshape(ids_o, [U * n, 1, TOPK])
            ids = ttnn.reshape(ttnn.concat([e, o], dim=1), [U, 1, C, TOPK])
            for t in (
                q_e,
                q_o,
                w_e,
                w_o,
                ids_e,
                ids_o,
            ):  # e / o / qq / ww are reshape aliases of these: never free an alias twice
                ttnn.deallocate(t)
        for t in (q, w):
            ttnn.deallocate(t)
        return ids

    def _part(self, pa, q, w, n, rope_start, rope_step, cs, pad_front, pad_back, L_end):
        """q RM [1,1,U*n,4096], w RM [1,1,U*n,32]: n query rows per user -> ids [U,1,n,512] (the pad rows are cut)."""
        d, U = self.dec, self.U
        tab = _itabs(pa, rope_start, n, rope_step)
        qh = ttnn.permute(ttnn.reshape(q, [U, n, IHEADS, IDIM]), (0, 2, 1, 3))  # [U,32,n,128] RM
        # RoPE + fp4 on the real rows (tile layout), then pad
        qh = ttnn.to_layout(qh, ttnn.TILE_LAYOUT)
        qh = self._rope(qh, tab)
        if self.fp4_q:
            qh = self._fp4(qh)
        if pad_front or pad_back:
            qrm = ttnn.to_layout(qh, ttnn.ROW_MAJOR_LAYOUT)
            seq = (
                [
                    ttnn.zeros(
                        [U, IHEADS, pad_front, IDIM], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.md
                    )
                ]
                if pad_front
                else []
            ) + [qrm]
            seq += (
                [
                    ttnn.zeros(
                        [U, IHEADS, pad_back, IDIM], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.md
                    )
                ]
                if pad_back
                else []
            )
            qh = ttnn.to_layout(ttnn.concat(seq, dim=2), ttnn.TILE_LAYOUT)
        sq = n + pad_front + pad_back
        wt = ttnn.reshape(w, [U, 1, n, IHEADS])
        if pad_front or pad_back:
            seq = (
                [
                    ttnn.zeros(
                        [U, 1, pad_front, IHEADS], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.md
                    )
                ]
                if pad_front
                else []
            ) + [wt]
            seq += (
                [
                    ttnn.zeros(
                        [U, 1, pad_back, IHEADS], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.md
                    )
                ]
                if pad_back
                else []
            )
            wt = ttnn.concat(seq, dim=2)
        wt = ttnn.to_layout(wt, ttnn.TILE_LAYOUT)
        per_user = []
        for u in range(U):
            qu = ttnn.slice(qh, [u, 0, 0, 0], [u + 1, IHEADS, sq, IDIM])
            wu = ttnn.slice(wt, [u, 0, 0, 0], [u + 1, 1, sq, IHEADS])
            sc = ttnn.experimental.indexer_score_dsa(
                qu, self.slab, wu, chunk_start_idx=cs, kv_len=L_end, program_config=self.cfg, cache_batch_idx=u
            )
            ids = ttnn.experimental.topk_large_indices(sc, k=TOPK, valid_length=L_end)  # [1,1,sq,512]
            ttnn.deallocate(sc)
            if pad_front or pad_back:
                ids = ttnn.slice(ids, [0, 0, pad_front, 0], [1, 1, pad_front + n, TOPK])
            per_user.append(ids)
            ttnn.deallocate(qu)
            ttnn.deallocate(wu)
        out = per_user[0] if U == 1 else ttnn.concat(per_user, dim=0)
        ttnn.deallocate(qh)
        ttnn.deallocate(wt)
        return out


class DSV41PrefillSparse:
    """Attached to a ``DSV41PrefillAttention`` as ``pa.sparse``: kv table + (index sources) indexer + the sparse attention of its layer."""

    def __init__(self, pa, kvt, indexer=None, idx_src=None, force=False):
        """kvt: ``PrefillKV`` of the layer's kv source group; indexer: ``DSV41PrefillIndexer`` of index-source layers; idx_src: the ``DSV41PrefillSparse`` of
        the last index source before this layer (readers). force: use the sparse path also while <= 512 entries are visible.
        """
        self.pa, self.kvt, self.indexer, self.idx_src, self.force = pa, kvt, indexer, idx_src, force
        self.owner = pa.a.source is None  # kv-source layer: produces latents
        self.ids = None  # [U,1,C,512] ids of the current chunk (index sources)
        self.sink = None
        self.md, self.U = pa.md, pa.U

    def set_sink(self, attn_sink):
        """attn_sink [64] host tensor of the layer."""
        a, md = self.pa.a, self.md
        rows, cols = a.rows, a.cols
        s = torch.zeros(cols, PAD_HEADS)
        s[:, :LOCAL_HEADS] = (attn_sink.float() / a.scale).reshape(cols, LOCAL_HEADS)
        self.sink = ttnn.from_torch(
            s.reshape(1, 1, 1, cols * PAD_HEADS).to(torch.bfloat16),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 3), mesh_shape=(rows, cols)),
        )

    def active(self, s0, C):
        r = self.pa.ratio
        return self.force or (s0 + C) // r > TOPK

    # ---- constants ----------------------------------------------------------------------------------------------------------------
    def _win_idx(self, C, first):
        key = ("win", id(self.md), C, first, self.kvt.LAT)
        if key not in _CONST:
            LAT = self.kvt.LAT
            i = torch.arange(C).view(-1, 1)
            c = torch.arange(WINDOW).view(1, -1)
            if first:  # kv of position p at scratch row LAT + 128 + p; valid positions max(0, i - 127) .. i first
                p = (i - (WINDOW - 1)).clamp(min=0) + c
                rows = torch.where(p <= i, LAT + WINDOW + p, torch.full_like(p, SKIP))
            else:  # position p = s0 + i - 127 + c at scratch row LAT + p - (s0 - 128) = LAT + 1 + i + c
                rows = (LAT + 1 + i + c).expand(C, WINDOW)
            _CONST[key] = _rep(
                self.md,
                rows.reshape(1, 1, C, WINDOW).to(torch.int64).to(torch.int32),
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
            )
        return _CONST[key]

    def _small_idx(self, s0, C):
        """ids of all visible entries (<= 512 of them) per query, valid first."""
        r = self.pa.ratio
        key = ("small", id(self.md), s0, C, r)
        if key not in _CONST:
            t = s0 + torch.arange(C).view(-1, 1)
            j = torch.arange(TOPK).view(1, -1)
            ok = j < (t + 1) // r
            ids = torch.where(ok, j, torch.full_like(j.expand(C, TOPK), SKIP))
            _CONST[key] = _rep(
                self.md, ids.reshape(1, 1, C, TOPK).to(torch.int64).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
            )
        return _CONST[key]

    # ---- per chunk ----------------------------------------------------------------------------------------------------------------
    def ingest(self, lat, lat_pre, s0):
        """Owner layers: the chunk's latents into the kv table and (index-key owners) the keys into the slab. Called for EVERY chunk."""
        r = self.pa.ratio
        if lat is not None:
            self.kvt.write_latents(lat, s0 // r)
        if self.indexer is not None and self.indexer.key_owner is None:
            self.indexer.add_keys(self.pa, lat_pre, s0)

    def attend(self, qh, kv, halo, h, qr, s0, C):
        """qh [U,8,C,512] RoPE'd heads (tile), kv [U,1,C,512] chunk rows (post RoPE), halo [U,1,128,512] / None -> raw attention output [U,8,C,512] (tile)."""
        pa, U = self.pa, self.U
        r = pa.ratio
        L_end = (s0 + C) // r
        self.kvt.write_window(kv, halo, s0)
        if L_end <= TOPK:
            ids = None
            small = self._small_idx(s0, C)
        elif self.indexer is not None:
            if self.ids is not None:
                ttnn.deallocate(self.ids)
            self.ids = self.indexer.select(pa, h, qr, s0, C)
            ids = self.ids
        else:
            ids = self.idx_src.ids
            assert ids is not None, "index source of this layer did not select for this chunk"
        win = self._win_idx(C, s0 == 0)
        outs = []
        zero_heads = _CONST.get(("zh", id(self.md), C))
        if zero_heads is None:
            zero_heads = _CONST[("zh", id(self.md), C)] = ttnn.zeros(
                [1, PAD_HEADS - LOCAL_HEADS, C, HEAD_DIM],
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.md,
            )
        for u in range(U):
            if ids is None:
                idx = ttnn.concat([win, small], dim=3)
            else:
                idx = ttnn.concat([win, ttnn.slice(ids, [u, 0, 0, 0], [u + 1, 1, C, TOPK])], dim=3)
            qu = ttnn.to_layout(ttnn.slice(qh, [u, 0, 0, 0], [u + 1, LOCAL_HEADS, C, HEAD_DIM]), ttnn.ROW_MAJOR_LAYOUT)
            qu = ttnn.concat([qu, zero_heads], dim=1)  # [1,32,C,512]
            o = ttnn.transformer.sparse_sdpa(
                qu,
                self.kvt.t,
                idx,
                HEAD_DIM,
                kv_format=ttnn.transformer.SparseKVFormat.BF16,
                scale=pa.a.scale,
                k_chunk_size=128,
                compute_kernel_config=pa.ckc_sdpa,
                attention_sink=self.sink,
                cache_batch_idx=u,
            )
            ttnn.deallocate(qu)
            ttnn.deallocate(idx)
            o = ttnn.to_layout(ttnn.slice(o, [0, 0, 0, 0], [1, LOCAL_HEADS, C, HEAD_DIM]), ttnn.TILE_LAYOUT)
            outs.append(o)
        out = outs[0] if U == 1 else ttnn.concat(outs, dim=0)
        if getattr(self, "dbg", False):
            self._debug(qh, out, ids, win, s0, C)
        return out

    def _debug(self, qh, out, ids, win, s0, C):
        """host golden of the sparse attention of user 0 / device 0 from the device's own q, kv table and index rows."""
        from models.demos.blackhole.deepseek_v41_flash.reference.ref_layer import pcc

        d0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
        q, o, kvt, wi = d0(qh)[0], d0(out)[0], d0(self.kvt.t)[0, 0], d0(win).reshape(C, -1).long()
        ii = None if ids is None else ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).reshape(self.U, C, TOPK)[0].long()
        sink = ttnn.to_torch(ttnn.get_device_tensors(self.sink)[0]).float().reshape(-1)[:LOCAL_HEADS] * self.pa.a.scale
        res = []
        for i in (0, 5, C // 2, C - 1):
            rows = wi[i].tolist() + ([] if ii is None else ii[i].tolist())
            rows = [r for r in rows if r != 0xFFFFFFFF and r != -1 and 0 <= r < kvt.shape[0]]
            K = kvt[rows]
            sc = (q[:, i] @ K.T) * self.pa.a.scale
            p = torch.softmax(torch.cat([sc, sink.view(-1, 1)], -1), -1)[:, :-1]
            res.append(round(pcc(p @ K, o[:, i]), 5))
        torch.save(
            {"q": q.to(torch.bfloat16), "kvt": kvt.to(torch.bfloat16), "o": o.to(torch.bfloat16), "LAT": self.kvt.LAT},
            f"/mnt/tt-data/ssinghal/dsv4-logs/h46x_dbg_{id(self.pa) % 100000}_{s0}.pt",
        )
        print(
            f"PSDBG s0={s0} dbgfile h46x_dbg_{id(self.pa) % 100000}_{s0}.pt sparse_sdpa vs host golden from its own inputs (queries 0,5,mid,last): {res}; n_rows {len(rows)}",
            flush=True,
        )


def attach_prefill_sparse(
    pas, idx_weights, users, max_tokens, c_max, attn_sinks, force=False, fp4_q=True, fp4_k=True, decode_indexers=None
):
    """Attach the sparse path to the ``DSV41PrefillAttention`` objects ``pas`` {layer id: pa} of a model (compressed layers only).
    idx_weights {layer: {wq_b, weights_proj, [wk, k_norm]}} of the index-source layers (``loader.load_layer(..., with_indexer=True)['indexer']``);
    attn_sinks {layer: [64] host tensor}; max_tokens: longest prompt (padded); c_max: largest chunk. Returns {layer: DSV41PrefillSparse}.
    """
    if (
        os.environ.get("DSV41_PF_SPARSE", "0") != "1"
    ):  # env-gated, default OFF (dense prefill attention, exact up to 512 compressed entries)
        return {}
    out, kvts, idx_of = {}, {}, {}
    last_idx_src, last_key_owner = None, None
    for L in sorted(pas):
        pa = pas[L]
        if not pa.compressed:
            continue
        a = pa.a
        owner = a if a.source is None else a.source
        if id(owner) not in kvts:
            kvts[id(owner)] = PrefillKV(pa.md, users, max_tokens // pa.ratio + 32, c_max)
        indexer = None
        if L in idx_weights:
            w = idx_weights[L]
            dec = None if decode_indexers is None else decode_indexers.get(L)
            if "wk" in w:
                indexer = DSV41PrefillIndexer(
                    pa.md,
                    users,
                    pa.ratio,
                    max_tokens // pa.ratio + 32,
                    dec=dec,
                    w_idx=w,
                    freqs_cis=a.freqs_cis if hasattr(a, "freqs_cis") else None,
                    fp4_q=fp4_q,
                    fp4_k=fp4_k,
                )
                last_key_owner = indexer
            else:
                indexer = DSV41PrefillIndexer(
                    pa.md,
                    users,
                    pa.ratio,
                    max_tokens // pa.ratio + 32,
                    dec=dec,
                    w_idx=w,
                    freqs_cis=None,
                    key_owner=last_key_owner,
                    fp4_q=fp4_q,
                    fp4_k=fp4_k,
                )
        sp = DSV41PrefillSparse(pa, kvts[id(owner)], indexer=indexer, idx_src=last_idx_src, force=force)
        sp.set_sink(attn_sinks[L])
        if indexer is not None:
            last_idx_src = sp
        pa.sparse = sp
        out[L] = sp
    return out
