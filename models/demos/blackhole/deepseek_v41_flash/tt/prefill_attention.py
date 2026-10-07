# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1-Flash PREFILL attention (S-wide) that wraps an already built decode attention object.

Reuses the decode weights (wqkv, norms, wq_b / wo_a / wo_b column shards, rope matrix, compressor weights) so prefill costs no
extra weight DRAM, and leaves the decode state (window ring + compressed cache in ``attn.cache``, ratio-2 ``prev_cs``) exactly as
the existing decode loop expects. See docs/superpowers/specs/2026-10-02-dsv41-device-prefill-design.md.

Layout per mesh row: ``U`` users x ``Sp`` (padded prompt length, multiple of 64) tokens, user-major, as ``h [1,1,U*Sp,D]`` bf16;
heads are split over the 8 mesh columns exactly like decode. The padded positions (>= S) compute garbage that nothing real reads
(causal) and that the decode masks hide.
"""

import os

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import pf_tune
from models.demos.blackhole.deepseek_v41_flash.tt.attention import (
    HEAD_DIM,
    LOCAL_HEADS,
    Q_LORA,
    WINDOW,
    DSV41CompressedAttention,
)

NEG = -1e9
AR_ROWS = int(os.environ.get("DSV41_PF_AR_ROWS", "128"))
_MASKS = {}
_ZEROS = {}  # zero tensors (ttnn.zeros uploads from the host: not allowed inside a trace capture, so they are cached)
_TABS = {}  # shared by all layers of one rope kind: (id(md), compressed?, start, n, step) -> (C, S, -S) device tables


def clear_chunk_caches():
    """Free the per-chunk constants (masks, rope tables); the driver calls this when it moves on to the next chunk."""
    from models.demos.blackhole.deepseek_v41_flash.tt.prefill_sparse import clear_sparse_caches

    clear_sparse_caches()
    for d in (_MASKS, _TABS):  # _ZEROS stays: a captured trace may hold them
        for v in d.values():
            for t in v if isinstance(v, tuple) else (v,):
                ttnn.deallocate(t)
        d.clear()


def zeros(md, shape, dtype=ttnn.bfloat16):
    key = (id(md), tuple(shape), dtype)
    if key not in _ZEROS:
        _ZEROS[key] = ttnn.zeros(list(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=md)
    return _ZEROS[key]


def gate_user_state(ctx, new, old):
    """Interleaved prefill (``DynCtx.umask``, DSV41_PF_UMASK=1): the carried per-user state after this chunk = ``new`` for the users that are part of the replay and ``old`` (unchanged) for
    the others. ``new`` / ``old`` [U,1,...] with the same shape, tile or row-major; the mask [U,1,1,1] is a persistent per-mesh-row device tensor. Without a mask: ``new`` (unchanged path).
    """
    m = getattr(ctx, "umask", None) if ctx is not None else None
    if m is None:
        return new
    rm = new.layout == ttnn.ROW_MAJOR_LAYOUT
    n_, o_ = (ttnn.to_layout(new, ttnn.TILE_LAYOUT), ttnn.to_layout(old, ttnn.TILE_LAYOUT)) if rm else (new, old)
    out = ttnn.where(m, n_, o_)
    if rm:
        ttnn.deallocate(n_)
        ttnn.deallocate(o_)
        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
    return out


def pad_len(S, mult=64):
    return -(-S // mult) * mult


class DSV41PrefillAttention:
    def __init__(self, attn, attn_sink: torch.Tensor, q_chunk=None, k_chunk=None, users=None):
        """attn: built DSV41Attention / DSV41CompressedAttention (decode); attn_sink: [64] host tensor of the layer. ``users``: users per mesh row of the prefill
        (default: the decode's; fewer = interleaved prefill of a few users per row, ``DSV41_PREFILL_UP``)."""
        self.a = attn
        self.md = attn.mesh_device
        self.U = users or attn.T
        self.compressed = isinstance(attn, DSV41CompressedAttention)
        self.ratio = attn.ratio if self.compressed else 0
        pf_tune.p64(attn.mesh_device)  # constant created outside any trace capture
        md = self.md
        rows, cols = attn.rows, attn.cols
        # sink for the prefill SDPA: [1, 8 local heads, 1, 1] per column, pre-divided by the softmax scale (the kernel multiplies by it)
        sink = (attn_sink.float() / attn.scale).reshape(1, cols * LOCAL_HEADS, 1, 1)
        self.sink = ttnn.from_torch(
            sink,
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 1), mesh_shape=(rows, cols)),
        )
        self.last_lat = (
            None  # [U,1,Sc_pad,512] latents of the last prefill (owner layers; readers use ``source.pre.last_lat``)
        )
        self.rs_tokens = False  # column-split mode (DSV41PrefillLayer.forward_cols): the attention output is reduce-scattered over tokens
        self.halo = None  # [U,1,128,512] kv rows (post RoPE) of the 128 positions before the next chunk (None before the first chunk)
        self.lat_all = None  # owner layers: [U,1,L,512] latents of every group closed so far (RoPE'd)
        self.dyn = None  # DynCtx: traced-chunk mode (forward_dyn)
        self.lat_buf = None  # [U,1,Lmax,512] FIFO of latents (kv-source layers, traced-chunk mode)
        self.state_sink = None  # optional callable(self, kv, lat, cs, s0, C, h) called once per chunk; replaces the dense decode-cache write
        self.sparse = None  # tt/prefill_sparse.py ``DSV41PrefillSparse`` (env DSV41_PF_SPARSE=1): indexer top-512 + sparse_sdpa once more than 512 compressed entries are visible
        self._lat_pre = None  # pooled latents of the chunk before RoPE (the indexer keys derive from them)
        self.tap = None  # dict -> keeps intermediates (qh, kv, o_raw, c, part) for diagnostics

    # ---- constants ----------------------------------------------------------------------------------------
    def _up(self, t, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            t.contiguous(),
            device=self.md,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.md),
        )

    def rope_tabs(self, start, n, step=1):
        """Full-width tables for positions start + step * arange(n) -> (C, S, -S) device tensors [1,1,n,512] (shared by the layers of one rope kind)."""
        key = (id(self.md), self.compressed, start, n, step)
        if key not in _TABS:
            c, s = self.a._rope_inputs(start + step * torch.arange(n))
            _TABS[key] = tuple(self._up(t.reshape(1, 1, n, HEAD_DIM)) for t in (c, s, -s))
        return _TABS[key]

    def begin(self):
        """Forget the carried state of the previous prompt (halo, latents)."""
        self.halo = self.lat_all = None

    def mask(self, S, C, s0, Lc):
        """Additive mask [1,1,C,H+C+Lc] for the chunk of positions [s0, s0+C) over keys [halo (H=128 if s0>0) | chunk kv | latents 0..Lc)]
        of a compressed layer (cached per (S, C, s0, ratio))."""
        key = (C, s0, self.ratio, id(self.md))  # the mask of a real query never depends on the prompt length S
        if key not in _MASKS:
            r, H = self.ratio, (WINDOW if s0 > 0 else 0)
            t = s0 + torch.arange(C).view(-1, 1)  # absolute query positions
            kp = torch.cat([s0 - H + torch.arange(H), s0 + torch.arange(C)]).view(1, -1)  # absolute key positions
            m = torch.where((kp <= t) & (kp > t - WINDOW) & (kp >= 0), 0.0, NEG)
            jc = torch.arange(Lc).view(1, -1)
            mc = torch.where(jc < (t + 1) // r, 0.0, NEG)
            _MASKS[key] = self._up(torch.cat([m, mc], dim=1).reshape(1, 1, C, -1))
        return _MASKS[key]

    def window_mask(self, C, s0):
        """Band mask [1,1,C,128+C] for a window-only layer of a chunk that has a halo (s0 > 0): same for every such chunk."""
        key = ("w", C, id(self.md))
        if key not in _MASKS:
            t = torch.arange(C).view(-1, 1) + WINDOW
            kp = torch.arange(WINDOW + C).view(1, -1)
            _MASKS[key] = self._up(torch.where((kp <= t) & (kp > t - WINDOW), 0.0, NEG).reshape(1, 1, C, WINDOW + C))
        return _MASKS[key]

    # the tuning knobs (tt/pf_tune.py) are read when a trace is captured, not at construction
    @property
    def lckc(self):
        return pf_tune.lin_ckc(self.a)

    @property
    def cckc(self):
        return pf_tune.lin_ckc(self.a, "comp")

    @property
    def rckc(self):
        return pf_tune.lin_ckc(self.a, "rope")

    @property
    def ckc_sdpa(self):
        return pf_tune.sdpa_ckc(self.a)

    @property
    def q_chunk(self):
        return pf_tune.QC

    @property
    def k_chunk(self):
        return pf_tune.KC

    def _lin(self, x, w):
        return pf_tune.linear(x, w, self.lckc, self.md)

    def _sdpa_cfg(self):
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.md.compute_with_storage_grid_size(),
            q_chunk_size=self.q_chunk,
            k_chunk_size=self.k_chunk,
            exp_approx_mode=pf_tune.EXP_APPROX,
        )

    def _sdpa(self, qh, keys, **kw):
        q8, k8 = pf_tune.to8(qh), pf_tune.to8(keys)
        o = ttnn.transformer.scaled_dot_product_attention(
            q8,
            k8,
            k8,
            scale=self.a.scale,
            attention_sink=self.sink,
            compute_kernel_config=self.ckc_sdpa,
            program_config=self._sdpa_cfg(),
            **kw,
        )
        if q8 is not qh:
            ttnn.deallocate(q8)
        if k8 is not keys:
            ttnn.deallocate(k8)
        return o

    # ---- helpers ------------------------------------------------------------------------------------------
    def _rope(self, x, tabs):
        c, s, _ = tabs
        if pf_tune.ROPE_PE:
            return pf_tune.rope_pe(x, c, s, pf_tune.p64(self.md), self.rckc)
        return ttnn.add(
            ttnn.multiply(x, c), ttnn.multiply(ttnn.matmul(x, self.a.Pf, compute_kernel_config=self.rckc), s)
        )

    def _rope_inv(self, x, tabs):
        c, _, ns = tabs
        if pf_tune.ROPE_PE:
            return pf_tune.rope_pe(x, c, ns, pf_tune.p64(self.md), self.rckc)
        return ttnn.add(
            ttnn.multiply(x, c), ttnn.multiply(ttnn.matmul(x, self.a.Pf, compute_kernel_config=self.rckc), ns)
        )

    def _compress(self, h, s0, C):
        """-> (lat [U,1,C/ratio,512] bf16 RoPE'd latents of the groups of this chunk, cs [1,1,R,1024] fp32 or None)."""
        a, U, r = self.a, self.U, self.ratio
        R = U * C
        Cc = C // r
        cs = None
        if r == 1:
            lat = ttnn.rms_norm(
                ttnn.linear(h, a.c_wkv, compute_kernel_config=self.cckc), weight=a.c_norm, epsilon=a.eps
            )
        else:
            cs = ttnn.linear(
                h, a.c_wcat, compute_kernel_config=self.cckc, dtype=ttnn.float32
            )  # [1,1,R,1024] = [kv | score]
            pairs = ttnn.to_layout(
                ttnn.reshape(ttnn.to_layout(cs, ttnn.ROW_MAJOR_LAYOUT), [1, 1, R // 2, 2 * 2 * HEAD_DIM]),
                ttnn.TILE_LAYOUT,
            )
            ga = ttnn.slice(pairs, [0, 0, 0, 0], [1, 1, R // 2, 2 * HEAD_DIM])  # first token of each group
            gb = ttnn.slice(pairs, [0, 0, 0, 2 * HEAD_DIM], [1, 1, R // 2, 4 * HEAD_DIM])
            d = ttnn.subtract(ga, gb)
            sl = lambda t, lo: ttnn.slice(t, [0, 0, 0, lo], [1, 1, R // 2, lo + HEAD_DIM])
            pooled = ttnn.addcmul(sl(gb, 0), sl(d, 0), sl(ttnn.sigmoid(d), HEAD_DIM))
            lat = ttnn.rms_norm(ttnn.typecast(pooled, ttnn.bfloat16), weight=a.c_norm, epsilon=a.eps)
        lat = ttnn.reshape(lat, [U, 1, Cc, HEAD_DIM])
        self._lat_pre = lat if self.sparse is not None and self.sparse.indexer is not None else None
        lat = self._rope(lat, self.rope_tabs(s0, Cc, r))
        return lat, cs

    # ---- forward ------------------------------------------------------------------------------------------
    def forward(self, h, S, write_state=True, s0=0):
        """h [1,1,U*C,D] bf16 tile (normed attention input of the chunk of positions [s0, s0+C), user-major) -> [1,1,U*C,D] bf16 replicated
        over columns. S = real prompt length (the chunk holds real tokens for positions < S). The carried state (kv halo of the previous 128
        positions, latents of the closed groups) is kept on this object; the decode state of the layer is written when the chunk contains the
        last real token (``write_state``). Single-chunk prompts are s0 = 0 and C = Sp."""
        a, U = self.a, self.U
        R = h.shape[2]
        C = R // U
        assert C % 64 == 0 and s0 % 64 == 0 and 0 <= S - s0
        last = s0 + C >= S
        if s0 == 0:
            self.begin()
        tabs = self.rope_tabs(s0, C)
        y = self._lin(h, a.wqkv)  # [1,1,R,1792]
        qr = ttnn.rms_norm(ttnn.slice(y, [0, 0, 0, 0], [1, 1, R, Q_LORA]), weight=a.q_norm, epsilon=a.eps)
        kvn = ttnn.rms_norm(
            ttnn.slice(y, [0, 0, 0, Q_LORA], [1, 1, R, Q_LORA + HEAD_DIM]), weight=a.kv_norm, epsilon=a.eps
        )
        ttnn.deallocate(y)
        q = self._lin(qr, a.wq_b)  # [1,1,R,8*512]
        sp = self.sparse
        keep_qr = (
            sp is not None and sp.indexer is not None and sp.active(s0, C)
        )  # the indexer scores with the normed q-lora
        if not keep_qr:
            ttnn.deallocate(qr)
        q = ttnn.reshape(q, [U, 1, C, LOCAL_HEADS * HEAD_DIM])
        qh, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=LOCAL_HEADS, num_kv_heads=0, transpose_k_heads=False
        )
        ttnn.deallocate(q)
        qh = self._rope(qh, tabs)
        kv = self._rope(ttnn.reshape(kvn, [U, 1, C, HEAD_DIM]), tabs)  # [U,1,C,512]
        if self.tap is not None:
            self.tap.update(qh=qh, kv=kv)

        lat = cs = None
        use_sparse = sp is not None and sp.active(s0, C)
        if self.compressed:
            if a.source is None:
                lat, cs = self._compress(h, s0, C)
                if (
                    not use_sparse
                ):  # the sparse path keeps the latents in its kv table instead of a growing dense tensor
                    self.lat_all = lat if self.lat_all is None else ttnn.concat([self.lat_all, lat], dim=2)
            if sp is not None:
                sp.ingest(lat, self._lat_pre, s0)
            lat_all = None if use_sparse else (self.lat_all if a.source is None else a.source.prefill.lat_all)
        if use_sparse:
            o = sp.attend(qh, kv, self.halo, h, qr if keep_qr else None, s0, C)
            keys = kv
        else:
            keys = kv if s0 == 0 else ttnn.concat([self.halo, kv], dim=2)  # [U,1,H+C,512]
            if self.compressed:
                keys = ttnn.concat([keys, lat_all], dim=2)
                kw = dict(is_causal=False, attn_mask=self.mask(S, C, s0, lat_all.shape[2]))
            elif s0 == 0:
                kw = dict(is_causal=True, sliding_window_size=WINDOW)
            else:
                kw = dict(is_causal=False, attn_mask=self.window_mask(C, s0))
            o = self._sdpa(qh, keys, **kw)
        if keep_qr:
            ttnn.deallocate(qr)
        if self.tap is None:
            ttnn.deallocate(qh)
        else:
            self.tap["o_raw"] = o
        if keys is not kv:
            ttnn.deallocate(keys)
        if (
            self.state_sink is not None
        ):  # paged / external state writer: sees every chunk's kv, latents and compressor state
            lat_out = lat if (self.compressed and a.source is None) else None
            self.state_sink(self, kv, lat_out, cs, s0, C, h)
        elif write_state and last:
            self._write_state(kv, cs, h, S, s0, C)
        if not last:  # carry the last 128 kv rows into the next chunk
            assert C % WINDOW == 0, "chunks of a multi-chunk prompt must be multiples of 128"
            full = kv if s0 == 0 else ttnn.concat([self.halo, kv], dim=2)
            new_halo = ttnn.clone(
                ttnn.slice(full, [0, 0, full.shape[2] - WINDOW, 0], [U, 1, full.shape[2], HEAD_DIM])
            )  # clone: a full-range slice may alias kv
            if self.halo is not None:
                ttnn.deallocate(self.halo)
            if full is not kv:
                ttnn.deallocate(full)
            self.halo = new_halo
        if self.tap is None:
            ttnn.deallocate(kv)
        o = self._rope_inv(o, tabs)
        c = ttnn.experimental.nlp_concat_heads(o)  # [U,1,C,8*512]
        if self.tap is None:
            ttnn.deallocate(o)
        zero = zeros(self.md, [U, 1, C, HEAD_DIM])  # decode wo_a has 512 zero rows in front
        c = ttnn.reshape(ttnn.concat([zero, c], dim=3), [1, 1, R, (LOCAL_HEADS + 1) * HEAD_DIM])
        part = self._lin(self._lin(c, a.wo_a), a.wo_b)
        if self.tap is not None:
            self.tap.update(c=c, part=ttnn.clone(part))
        return self._allreduce(part)

    # ---- traced-chunk mode ---------------------------------------------------------------------------------
    def alloc_dyn(self, ctx):
        """Attach the per-chunk context and allocate this layer's persistent halo (+ latent FIFO for kv-source layers)."""
        self.dyn = ctx
        a, U = self.a, self.U
        z = lambda shape: ttnn.from_torch(
            torch.zeros(*shape),
            device=self.md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.md),
        )
        rows = tuple(self.md.shape)[0]
        # users differ per mesh row but the content starts as zeros: replicate then it is overwritten SPMD by the first chunk
        self.halo = z([U, 1, WINDOW, HEAD_DIM])
        del rows
        if self.sparse is not None:
            self.sparse.alloc_dyn(ctx)
        # the dense FIFO of latents is only read by the dense path: when the sparse path is active (dyn_on) its kv table holds the latents, so skip the
        # [U,1,L,512] bf16 buffer (~160 MiB/bank at 64k, U=8)
        sparse_on = self.sparse is not None and self.sparse.dyn_on
        self.lat_buf = (
            z([U, 1, ctx.L[self.ratio], HEAD_DIM]) if (self.compressed and a.source is None and not sparse_on) else None
        )

    def reset_dyn(self):
        """Zero the carried state before a new prompt (masked anyway, but NaN garbage must not sit in the buffers)."""
        for t in (self.halo, self.lat_buf):
            if t is not None:
                ttnn.copy(ttnn.zeros_like(t), t)

    def forward_dyn(self, h):
        """Same maths as ``forward`` for one chunk, but every per-chunk quantity is a persistent device tensor of ``self.dyn`` (RoPE tables,
        masks, latent offsets): the captured program is identical for every chunk. h [1,1,U*C,D] bf16 -> [1,1,U*C,D] bf16.
        """
        a, U, ctx = self.a, self.U, self.dyn
        R = h.shape[2]
        C = R // U
        assert C == ctx.C
        tabs = ctx.tabs[self.compressed]
        y = self._lin(h, a.wqkv)
        qr = ttnn.rms_norm(ttnn.slice(y, [0, 0, 0, 0], [1, 1, R, Q_LORA]), weight=a.q_norm, epsilon=a.eps)
        kvn = ttnn.rms_norm(
            ttnn.slice(y, [0, 0, 0, Q_LORA], [1, 1, R, Q_LORA + HEAD_DIM]), weight=a.kv_norm, epsilon=a.eps
        )
        ttnn.deallocate(y)
        q = self._lin(qr, a.wq_b)
        sp = self.sparse
        dyn_sp = sp is not None and sp.dyn_on
        if not (dyn_sp and sp.indexer is not None):
            ttnn.deallocate(qr)
        q = ttnn.reshape(q, [U, 1, C, LOCAL_HEADS * HEAD_DIM])
        qh, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=LOCAL_HEADS, num_kv_heads=0, transpose_k_heads=False
        )
        ttnn.deallocate(q)
        qh = self._rope(qh, tabs)
        kv = self._rope(ttnn.reshape(kvn, [U, 1, C, HEAD_DIM]), tabs)
        lat = cs = None
        if dyn_sp:  # indexer top-512 + sparse_sdpa (tt/prefill_sparse.py), all per-chunk values are device tensors
            if a.source is None:
                lat, cs = self._compress_dyn(h, C)
            o = sp.attend_dyn(qh, kv, self.halo, lat, self._lat_pre, h, qr)
            if sp.indexer is not None:
                ttnn.deallocate(qr)
            ttnn.deallocate(qh)
            return self._finish_dyn(o, kv, lat, cs, h, tabs, a, U, C, R)
        keys = ttnn.concat([self.halo, kv], dim=2)  # [U,1,128+C,512]
        if self.compressed:
            r = self.ratio
            if a.source is None:
                lat, cs = self._compress_dyn(h, C)
                Cc = C // r
                buf = self.lat_buf
                if (
                    buf.shape[2] == Cc
                ):  # single-chunk prompt (S_pad == C): the FIFO is just this chunk (a zero-length slice breaks the concat)
                    ttnn.copy(gate_user_state(ctx, lat, buf), buf)
                else:
                    new = ttnn.concat([ttnn.slice(buf, [0, 0, Cc, 0], [U, 1, buf.shape[2], HEAD_DIM]), lat], dim=2)
                    if getattr(ctx, "umask", None) is not None:
                        gated = gate_user_state(ctx, new, buf)
                        ttnn.deallocate(new)
                        new = gated
                    ttnn.copy(new, buf)
                    ttnn.deallocate(new)
                lat_buf = buf
            else:
                lat_buf = a.source.prefill.lat_buf
            full = ttnn.concat([keys, lat_buf], dim=2)
            ttnn.deallocate(keys)
            keys = full
        kw = dict(is_causal=False, attn_mask=ctx.masks[self.ratio] if self.compressed else ctx.win_mask)
        o = self._sdpa(qh, keys, **kw)
        ttnn.deallocate(qh)
        ttnn.deallocate(keys)
        return self._finish_dyn(o, kv, lat, cs, h, tabs, a, U, C, R)

    def _finish_dyn(self, o, kv, lat, cs, h, tabs, a, U, C, R):
        if self.state_sink is not None:
            self.state_sink(self, kv, lat, cs, 0, C, h)
        new_halo = ttnn.slice(ttnn.concat([self.halo, kv], dim=2), [0, 0, C, 0], [U, 1, WINDOW + C, HEAD_DIM])
        if getattr(self.dyn, "umask", None) is not None:
            gated = gate_user_state(self.dyn, new_halo, self.halo)
            ttnn.deallocate(new_halo)
            new_halo = gated
        ttnn.copy(new_halo, self.halo)
        ttnn.deallocate(new_halo)
        ttnn.deallocate(kv)
        o = self._rope_inv(o, tabs)
        c = ttnn.experimental.nlp_concat_heads(o)
        ttnn.deallocate(o)
        zero = zeros(self.md, [U, 1, C, HEAD_DIM])
        c = ttnn.reshape(ttnn.concat([zero, c], dim=3), [1, 1, R, (LOCAL_HEADS + 1) * HEAD_DIM])
        part = self._lin(self._lin(c, a.wo_a), a.wo_b)
        return self._allreduce(part)

    def _compress_dyn(self, h, C):
        """``_compress`` with the latent RoPE tables taken from the per-chunk context."""
        r = self.ratio
        tabs = self.dyn.lat_tabs[r] if r > 1 else self.dyn.tabs[True]
        saved = self.rope_tabs
        self.rope_tabs = lambda start, n, step=1: tabs  # noqa: E731  (``_compress`` asks for the latent positions)
        try:
            return self._compress(h, 0, C)
        finally:
            self.rope_tabs = saved

    def _reduce_scatter_tokens(self, part):
        """[1,1,R,D] partial sums (per column) -> [1,1,R/8,D]: chunk g of the result is this column's 32 rows of rows [256g, 256g+256) summed over
        the columns (the owner of 32-token chunk i of the row is column i % 8)."""
        a, R = self.a, part.shape[2]
        cc = a.ccl
        outs = []
        for i in range(0, R, 256):
            piece = ttnn.slice(part, [0, 0, i, 0], [1, 1, i + 256, part.shape[3]])
            outs.append(
                ttnn.experimental.reduce_scatter_minimal_async(
                    piece,
                    dim=2,
                    multi_device_global_semaphore=cc.get_rs_ping_pong_semaphore(),
                    num_links=cc.num_links,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=cc.topology,
                    cluster_axis=1,
                    barrier_semaphore=cc.get_barrier_semaphore(),
                )
            )
            ttnn.deallocate(piece)
        ttnn.deallocate(part)
        out = ttnn.concat(outs, dim=2) if len(outs) > 1 else outs[0]
        if len(outs) > 1:
            for o in outs:
                ttnn.deallocate(o)
        return out

    def _allreduce(self, part):
        """All-reduce over the 8 mesh columns in pieces of <= AR_ROWS rows. A single reduce_scatter + all_gather of [1,1,512,5120] bf16
        silently corrupts one tile-row block (sometimes NaN) on this build (tests/test_prefill_attn_debug.py); pieces of 256 rows or
        less are exact."""
        a, R = self.a, part.shape[2]
        if (
            self.rs_tokens
        ):  # column-split mode: reduce-scatter over the TOKEN dim in pieces of 256 rows -> own 32-row chunk of every group of 8
            return self._reduce_scatter_tokens(part)
        if R <= AR_ROWS:
            return a.mesh_config.allreduce(part, a.ccl, axis=1)
        pieces = [
            a.mesh_config.allreduce(
                ttnn.slice(part, [0, 0, i, 0], [1, 1, min(i + AR_ROWS, R), part.shape[3]]), a.ccl, axis=1
            )
            for i in range(0, R, AR_ROWS)
        ]
        ttnn.deallocate(part)
        out = ttnn.concat(pieces, dim=2)
        for p in pieces:
            ttnn.deallocate(p)
        return out

    # ---- decode state -------------------------------------------------------------------------------------
    def _ring(self, kv, S, s0):
        """Window ring [U,1,128,512] (position p at slot p % 128) after the prompt, from the last chunk's kv (+ the halo of the chunk before)."""
        U = self.U
        pre = zeros(self.md, [U, 1, WINDOW, HEAD_DIM], kv.dtype) if s0 == 0 else self.halo
        cat = ttnn.to_layout(ttnn.concat([pre, kv], dim=2), ttnn.ROW_MAJOR_LAYOUT)
        n = S - s0  # real tokens of this chunk
        tail = ttnn.slice(cat, [0, 0, n, 0], [U, 1, n + WINDOW, HEAD_DIM])  # positions S-128 .. S-1
        sft = S % WINDOW
        ring = (
            ttnn.concat(
                [
                    ttnn.slice(tail, [0, 0, WINDOW - sft, 0], [U, 1, WINDOW, HEAD_DIM]),
                    ttnn.slice(tail, [0, 0, 0, 0], [U, 1, WINDOW - sft, HEAD_DIM]),
                ],
                dim=2,
            )
            if sft
            else tail
        )
        return ttnn.to_layout(ring, ttnn.TILE_LAYOUT)

    def _write_state(self, kv, cs, h, S, s0, C):
        """Window ring (+ compressed latents, prev_cs) into the decode cache (the last chunk of the prompt calls this)."""
        a, U = self.a, self.U
        ring = self._ring(kv, S, s0)
        if not self.compressed:
            full = ring
            if a.cache.shape[2] > WINDOW:
                full = ttnn.pad(ring, [(0, 0), (0, 0), (0, a.cache.shape[2] - WINDOW), (0, 0)], 0.0)
        else:
            src = a if a.source is None else a.source
            comp = src.prefill.lat_all
            Scp = comp.shape[2]
            if Scp >= a.max_comp:
                comp = ttnn.slice(comp, [0, 0, 0, 0], [U, 1, a.max_comp, HEAD_DIM])
            else:
                comp = ttnn.pad(comp, [(0, 0), (0, 0), (0, a.max_comp - Scp), (0, 0)], 0.0)
            full = ttnn.concat([ring, comp], dim=2)
        ttnn.copy(ttnn.typecast(full, a.cache.dtype) if full.dtype != a.cache.dtype else full, a.cache)
        if cs is not None:  # ratio 2: the compressor's "previous token" state = [kv | score] of the last prompt token
            last = ttnn.slice(
                ttnn.reshape(cs, [U, 1, C, 2 * HEAD_DIM]), [0, 0, S - 1 - s0, 0], [U, 1, S - s0, 2 * HEAD_DIM]
            )
            last = ttnn.reshape(last, [1, 1, U, 2 * HEAD_DIM])
            if a.prev_cs is None:
                a.prev_cs = ttnn.clone(last)
            else:
                ttnn.copy(last, a.prev_cs)
