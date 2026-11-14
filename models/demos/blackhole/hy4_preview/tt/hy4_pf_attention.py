# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4AttentionPF: DIRECT-TO-POOL chunked prefill attention (task pool-prefill; program/pool-prefill/DESIGN.md).

``prefill_chunk`` has the contract of ``Hy4Attention.prefill_chunk`` (h [1,1,512,6144] -> (out [1,1,512,6144], ids)) but never builds a replicated cache: the chunk's rows are written into page
p = start / 512 of the sequence-sharded SCALED_FP8 pool (``slice_copy``), the indexer scores the 512 query rows against the column's own bfp8 key slices in the pool, and the attention of the chunk
runs on the column's own pool shard (q all-gathered over the columns, multi-row split kernel with (max, sum) stats, pre-scaled reduce-scatter merge). The ids returned for IndexCache reuse are this
column's LOCAL pool-row ids [1,1,512,2048] (sentinel tail), so shared layers skip the global -> local filter. The pool (``init_pool``) is the one of ``Hy4AttentionPool`` / ``Hy4PoolDecoder``.

Flags: HY4_PF_IDX_FID (hifi2 default | lofi | hifi4: fidelity of ``indexer_score_dsa``), HY4_PF_KLOC (local top-k width, default 512), HY4_PF_KC (split kernel k_chunk, default 128).
"""
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
from models.demos.blackhole.hy4_preview.tt import hy4_pool_prefill as PP
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.blackhole.hy4_preview.tt.hy4_idx_select import select_v2
from models.demos.blackhole.hy4_preview.tt.hy4_pf_opt import plinear
from models.demos.blackhole.hy4_preview.tt.hy4_pool_decode import Hy4AttentionPool

NC = 8
CH = 512
LINKS = int(os.environ.get("HY4_POOL_AG_LINKS", "2"))
_DENSE = {}


class Hy4AttentionPF(Hy4AttentionPool):
    # ------------------------------------------------------------------ one-time device constants
    def _pf_init(self):
        if getattr(self, "_pf_ready", False):
            return
        md = self.md
        so = self.host_w["sink_over_scale"].tensor.reshape(1, 1, 1, -1).float()  # [1,1,1,64] sink / scale per head
        self.pf_sink_logit = upload(
            md, so * self.scale, ttnn.float32, ttnn.TILE_LAYOUT
        )  # sink in the scaled-logit domain, fp32 (replicated)
        eye = torch.eye(NC).reshape(NC * NC, 1, 1, 1)  # device c holds eye[c] = [8,1,1,1]
        mapper = (
            ttnn.ShardTensor2dMesh(md, dims=(None, 0), mesh_shape=tuple(md.shape))
            if int(md.shape[0]) > 1
            else ttnn.ShardTensorToMesh(md, dim=0)
        )
        self.pf_onehot = ttnn.from_torch(
            eye.float(),
            device=md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fid = {"hifi4": ttnn.MathFidelity.HiFi4, "hifi2": ttnn.MathFidelity.HiFi2, "lofi": ttnn.MathFidelity.LoFi}[
            os.environ.get("HY4_PF_IDX_FID", "hifi2")
        ]
        self.pf_score_ckc = ttnn.init_device_compute_kernel_config(
            md.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
        )
        self.pf_score_cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=128, head_group_size=0)
        self._pf_ready = True

    # ------------------------------------------------------------------ CCL helpers (the layer's CCLManager, TILE tensors, links = HY4_POOL_AG_LINKS)
    def _ag(self, t, dim):
        return ttnn.experimental.all_gather_async(
            t,
            dim=dim,
            cluster_axis=1,
            mesh_device=self.md,
            topology=ttnn.Topology.Ring,
            multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
            num_links=LINKS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
        )

    def _rs(self, t, dim):
        return ttnn.experimental.reduce_scatter_minimal_async(
            t,
            dim=dim,
            multi_device_global_semaphore=self.ccl.get_rs_ping_pong_semaphore(),
            num_links=LINKS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            cluster_axis=1,
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
        )

    # ------------------------------------------------------------------ write path (C)
    def _pf_write_kv(self, lat, k_pe, p):
        """latent / rope rows of the chunk (replicated over the columns) -> SCALED_FP8 packed rows -> this column's 64-row slice of pool page p."""
        dp = ttnn.experimental.deepseek_prefill
        f8, sc = dp.per_token_cast_to_fp8(lat, round_scale_to_power_of_two=True)
        rope_rm = ttnn.to_layout(k_pe, ttnn.ROW_MAJOR_LAYOUT)
        packed = dp.pack_scaled_fp8_kv_cache(f8, sc, rope_rm)
        HP.slice_copy(packed, self.kv_pool, self.ctx.colid, 64, HP.ROWB, 1, dst_page0=p)
        for t in (f8, sc, rope_rm, packed):
            ttnn.deallocate(t)

    def _pf_write_key(self, h, icos, isin, p):
        """indexer key rows (layer_norm + rope) -> bfp8 -> this column's slice of key page p. Returns nothing (the score reads the pool)."""
        ix = self.indexer
        k = plinear(self.pf, h, ix.w_k, self.ckc_pf)
        k = ttnn.layer_norm(k, weight=ix.k_norm_w, bias=ix.k_norm_b, epsilon=ix.ln_eps)
        k = ix._rot(k, icos, isin)
        kb = ttnn.typecast(k, ttnn.bfloat8_b)
        ttnn.deallocate(k)
        HP.slice_copy(kb, self.key_view, self.ctx.colid, 8, HP.KEYTILE, 1, dst_page0=p)
        ttnn.deallocate(kb)

    # ------------------------------------------------------------------ indexer on the pool (B)
    def _pf_select(self, h, qa, start, p, icos, isin):
        """global top-2048 ids of the 512 rows -> this column's local ids [1,1,512,2048] (sentinel tail). Requires start >= 2048 (shorter contexts are dense: see prefill_chunk)."""
        ix, ctx = self.indexer, self.ctx
        n = h.shape[2]
        q = plinear(self.pf, qa, ix.w_q, self.ckc_pf)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=ix.nh, num_kv_heads=0, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        q = ix._rot(q, icos, isin)
        w = plinear(self.pf, h, ix.w_p, self.ckc_pf)
        return self._pf_select_core(q, w, p)

    def _pf_select_core(self, q, w, p, return_global=False):
        """q [1,32,512,128] (rope applied), w [1,1,512,32]: score against this column's keys of pages 0..p (page p = this chunk, already written), staircase, two-stage top-2048, local ids."""
        ctx = self.ctx
        n = q.shape[2]
        n_local = 64 * (p + 1)
        ksl = ttnn.slice(self.key_view, [0, 0, 0, 0], [1, 1, n_local, 128])
        score = ttnn.experimental.indexer_score_dsa(
            q,
            ksl,
            w,
            chunk_start_idx=n_local - 32,
            program_config=self.pf_score_cfg,
            compute_kernel_config=self.pf_score_ckc,
            kv_len=n_local,
        )  # [1,1,n,n_local] bf16 RM
        for t in (q, w):
            ttnn.deallocate(t)
        if n_local != int(self.key_view.shape[2]):  # a full-length slice is the pool view itself
            ttnn.deallocate(ksl)
        PP.pf_stair(score, ctx.colid)
        kloc = min(int(os.environ.get("HY4_PF_KLOC", "512")), n_local)
        gids = select_v2(
            ctx.ag, ctx.colid, ttnn.reshape(score, [1, 1, n, n_local]), n_local, 1, kloc, kloc, 2048, 1, False, None
        )  # [n,1,1,2048] global ids
        ttnn.deallocate(score)
        loc = PP.pf_id_filter(gids, ctx.colid)
        if return_global:
            return loc, gids
        ttnn.deallocate(gids)
        return loc

    def _pf_dense_ids(self, start, n):
        """chunks inside the first 2048 tokens: every causal key is selected. Local ids of this column (cached per chunk start; layer independent)."""
        key = (id(self.ctx), start)
        if key not in _DENSE:
            ix = torch.full((n, 1, 1, self.cfg.index_topk), -1, dtype=torch.int32)
            for i in range(n):
                ix[i, 0, 0, : start + i + 1] = torch.arange(start + i + 1, dtype=torch.int32)
            g = upload(self.md, ix, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            _DENSE[key] = PP.pf_id_filter(g, self.ctx.colid)
            ttnn.deallocate(g)
        return ttnn.clone(_DENSE[key])  # the chain deallocates the ids it carried: hand out a copy

    # ------------------------------------------------------------------ attention on the pool shard + merge (A)
    def _pf_merge_scale(self, stats):
        """stats [1,1,n,2*64] RM (raw max | sum of this column) -> scale [1,64,n,1] TILE bf16 = w_c / (sum_c w + sink term) of THIS column (w_c = l_c exp(m_c * scale - ref)).
        TILE stats gathered over dim 0 (the RM all-gather of the 256 B rows costs 230 us, TILE 48 us); the math is fp32 (bf16 exp of a ~40-wide argument loses 8 % in the weights).
        ref = the true max over the columns and the sink (default, HY4_PF_SAFE_MERGE=1, +3 ops) or the sink alone (=0: overflows when max - sink > 88).
        """
        n = stats.shape[2]
        st = ttnn.to_layout(stats, ttnn.TILE_LAYOUT)
        sg = self._ag(st, 0)  # [8,1,n,128] TILE (source-major)
        ttnn.deallocate(st)
        sg32 = ttnn.typecast(sg, ttnn.float32)
        ttnn.deallocate(sg)
        m = ttnn.slice(sg32, [0, 0, 0, 0], [NC, 1, n, 64])
        l = ttnn.slice(sg32, [0, 0, 0, 64], [NC, 1, n, 128])
        ttnn.deallocate(sg32)
        ms = ttnn.multiply(m, self.scale)  # scaled-logit max of every column (empty column: -inf)
        sl = self.pf_sink_logit  # [1,1,1,64] fp32, the sink in the scaled-logit domain
        if os.environ.get("HY4_PF_SAFE_MERGE", "1") == "1":
            mx = ttnn.maximum(ttnn.max(ms, dim=0, keepdim=True), sl)
            e = ttnn.exp(ttnn.subtract(ms, mx))
            w = ttnn.multiply(l, e)
            den = ttnn.add(ttnn.sum(w, dim=0, keepdim=True), ttnn.exp(ttnn.subtract(sl, mx)))
            ttnn.deallocate(mx)
        else:
            e = ttnn.exp(ttnn.subtract(ms, sl))
            w = ttnn.multiply(l, e)
            den = ttnn.add(ttnn.sum(w, dim=0, keepdim=True), 1.0)
        f = ttnn.multiply(ttnn.multiply(w, ttnn.reciprocal(den)), self.pf_onehot)
        own = ttnn.sum(f, dim=0, keepdim=True)  # [1,1,n,64] fp32
        for t in (m, l, ms, e, w, den, f):
            ttnn.deallocate(t)
        own16 = ttnn.typecast(own, ttnn.bfloat16)
        ttnn.deallocate(own)
        s_t = ttnn.permute(own16, (0, 3, 2, 1))  # [1,64,n,1] TILE
        ttnn.deallocate(own16)
        return s_t

    def _pf_attend(self, qfull, ids_loc):
        """qfull [1,8,n,576] TILE (own 8 heads) -> o_lat [1,8,n,512] TILE merged over the columns (sink included)."""
        qg = self._ag(qfull, 1)  # [1,64,n,576] TILE
        q_rm = ttnn.to_layout(qg, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(qg)
        out, stats = PP.sparse_sdpa_pf(
            q_rm, self.kv_pool, ids_loc, self.scale, int(os.environ.get("HY4_PF_KC", "128")), self.ckc_sdpa
        )
        ttnn.deallocate(q_rm)
        s_t = self._pf_merge_scale(stats)
        ttnn.deallocate(stats)
        o_t = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        ttnn.deallocate(out)
        b8 = "rs" in os.environ.get(
            "HY4_PF_BFP8", ""
        )  # bfp8 partials through the reduce-scatter (HY4_PF_BFP8=rs, default OFF)
        o_s = ttnn.multiply(o_t, s_t, dtype=ttnn.bfloat8_b) if b8 else ttnn.multiply(o_t, s_t)
        ttnn.deallocate(o_t)
        ttnn.deallocate(s_t)
        o_lat = self._rs(o_s, 1)  # [1,8,n,512]: this column's 8 heads, summed over the 8 pre-scaled partials
        ttnn.deallocate(o_s)
        if b8:
            o2 = ttnn.typecast(o_lat, ttnn.bfloat16)
            ttnn.deallocate(o_lat)
            o_lat = o2
        return o_lat

    # ------------------------------------------------------------------ the chunk
    def prefill_chunk(self, h, start, ids_in=None, ids_override=None, append=True, tabs=None, ids_are_global=False):
        """h [1,1,n,6144] TILE bf16 with n a multiple of 512 (PF_ROWS = 512 or 1024: a longer call is processed as n/512 consecutive pool pages) -> (out [1,1,n,6144], local ids [1,1,n,2048]);
        ids_in = local ids of the index-source layer (shared layers), or global natural ids [1,1,n,2048] when ids_are_global.
        """
        n = h.shape[2]
        if n == CH:
            return self._pf_chunk512(h, start, ids_in, append, tabs, ids_are_global)
        assert n % CH == 0 and tabs is None, (n, start)
        outs, idss = [], []
        for j in range(n // CH):
            hj = ttnn.slice(h, [0, 0, CH * j, 0], [1, 1, CH * (j + 1), h.shape[3]])
            ij = (
                None if ids_in is None else ttnn.slice(ids_in, [0, 0, CH * j, 0], [1, 1, CH * (j + 1), ids_in.shape[3]])
            )
            o, i_ = self._pf_chunk512(hj, start + CH * j, ij, append, None, ids_are_global)
            ttnn.deallocate(hj)
            if ij is not None:
                ttnn.deallocate(ij)
            outs.append(o)
            idss.append(i_)
        out = ttnn.concat(outs, dim=2)
        for t in outs:
            ttnn.deallocate(t)
        if not self.has_indexer:
            return out, ids_in
        ids = ttnn.concat(idss, dim=2)
        for t in idss:
            ttnn.deallocate(t)
        return out, ids

    def _pf_chunk512(self, h, start, ids_in, append, tabs, ids_are_global):
        cfg = self.cfg
        n = h.shape[2]
        assert n == CH and start % CH == 0, (n, start)
        self._pf_init()
        p = start // CH
        cos, sin, icos, isin = tabs if tabs is not None else self.rope_tabs(start, n)
        qa_c = plinear(self.pf, h, self.w_q_a, self.ckc_pf)
        qa_g = self.mesh_config.allgather(qa_c, self.ccl, axis=1, dim=3)
        ttnn.deallocate(qa_c)
        qa = self._rms(qa_g, self.q_a_norm)
        ttnn.deallocate(qa_g)
        self._tap("qa", qa)
        kv = plinear(self.pf, h, self.w_kv_a, self.ckc_pf)
        lat = self._rms(ttnn.slice(kv, [0, 0, 0, 0], [1, 1, n, cfg.kv_lora_rank]), self.kv_a_norm)
        k_pe = self.rope.apply(ttnn.slice(kv, [0, 0, 0, cfg.kv_lora_rank], [1, 1, n, cfg.kv_a_out]), cos, sin, self.ckc)
        ttnn.deallocate(kv)
        self._tap("lat", lat)
        self._tap("k_pe", k_pe)
        if append:
            self._pf_write_kv(lat, k_pe, p)
        self._free(lat, "lat")
        self._free(k_pe, "k_pe")
        dense = start + n <= cfg.index_topk
        if self.has_indexer:
            if append:
                self._pf_write_key(h, icos, isin, p)
            ids = self._pf_dense_ids(start, n) if dense else self._pf_select(h, qa, start, p, icos, isin)
        else:
            assert ids_in is not None, "shared-indexer layer needs the ids of its index source"
            if ids_are_global:
                ids = PP.pf_id_filter(ttnn.reshape(ids_in, [n, 1, 1, cfg.index_topk]), self.ctx.colid)
            else:
                ids = ids_in
        self._tap("ids", ids)
        q = plinear(self.pf, qa, self.w_q_b, self.ckc_pf)
        self._free(qa, "qa")
        q_nope, q_pe = ttnn.experimental.nlp_create_q_heads_split(
            q, num_heads=self.hc, split_head_dim=cfg.qk_nope_head_dim, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(q)
        q_abs = ttnn.matmul(q_nope, self.w_uk, compute_kernel_config=self.ckc_pf)
        ttnn.deallocate(q_nope)
        q_pe = self.rope.apply(q_pe, cos, sin, self.ckc)
        qfull = ttnn.concat([q_abs, q_pe], dim=3)
        ttnn.deallocate(q_abs)
        ttnn.deallocate(q_pe)
        self._tap("qfull", qfull)
        o_lat = self._pf_attend(qfull, ids)
        self._free(qfull, "qfull")
        self._tap("o_lat", o_lat)
        out = self._finish(o_lat, h)
        return out, ids
