# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-Preview attention layer on the 4x8 Blackhole mesh (rows = users, columns = TP8 head groups): gated MLA + DSA indexer / IndexCache + learnable sink.

Dataflow per column c (heads 8c..8c+7), input h [1,1,n,6144] bf16 replicated over the columns (post input_layernorm), output [1,1,n,6144] replicated (all-reduced):

    qa  = rms_norm(all_gather(h @ q_a[:, 256c : 256c+256]))          [n,2048]       (CPU model: tt/hy4_attn_ref.py, every name below is a tap)
    kv  = h @ kv_a -> lat = rms_norm(kv[:, :512]), k_pe = rope(kv[:, 512:])       cache row = [lat | k_pe] (576)   (written to the layer's KVCache)
    q   = qa @ q_b[c] -> [8 heads, 192 nope | 64 rope];  q_abs = q_nope @ w_uk[c] (512);  qfull = [q_abs | rope(q_pe)]  [8, n, 576]
    ids = indexer top-2048 (full layers) or the ids of the index-source layer (shared layers)
    o_lat = sparse_sdpa(qfull, cache, ids, v_dim=512, scale=256^-0.5, sink)         [8, n, 512]   (K row 576, V = its first 512)
        prefill: heads->sequence all_to_all over the 8 columns around the stock op (H=64, n/8 rows per chip); decode: 8 heads zero-padded to 32
    o   = o_lat @ w_uv_t[c] -> [n, 8*256];  o *= sigmoid(h @ gate[c]);  out = all_reduce(o @ o_proj[c])

Open conventions (default = the oracle's, flags for A/B): rope pairing ``rope_interleave`` (adjacent pairs), sink = extra softmax logit in the post-scale domain (the op takes sink/scale),
gate = sigmoid(linear(h)) with h the NORMED input. Everything bf16 here (weights bf16, ``wdtype`` switches to bfp8).
"""
import os

import ttnn
from models.demos.blackhole.hy4_preview.tt.hy4_attn_weights import build_attn_host_weights
from models.demos.blackhole.hy4_preview.tt.hy4_common import RopeDev, compute_cfg, upload
from models.demos.blackhole.hy4_preview.tt.hy4_indexer import Hy4Indexer
from models.demos.blackhole.hy4_preview.tt.hy4_kv_cache import KVCache
from models.demos.blackhole.hy4_preview.tt.hy4_pf_opt import PfOpt, plinear

_ROPE_TABS = (
    {}
)  # B49 (HY4_B49_ROPE_CACHE=1): key -> (cos, sin, idx_cos, idx_sin) of the current prefill chunk, shared by all layers


class Hy4Attention:
    def __init__(
        self,
        md,
        mesh_config,
        ccl,
        cfg,
        sd,
        layer,
        kv_fmt="bf16",
        rope_interleave=True,
        idx_rope_interleave=False,
        wdtype=ttnn.bfloat16,
        sink_over_scale_dtype=ttnn.bfloat16,
        wdtypes=None,
        pf_opt=None,
    ):
        """sd: checkpoint tensors of the layer keyed by the path after ``model.layers.<i>.`` (torch)."""
        self.md, self.mesh_config, self.ccl, self.cfg, self.layer, self.kv_fmt = (
            md,
            mesh_config,
            ccl,
            cfg,
            layer,
            kv_fmt,
        )
        self.rows, self.cols = int(md.shape[0]), int(md.shape[1])
        assert self.cols == cfg.mesh_cols
        self.hc = cfg.heads_per_col
        self.scale = cfg.qk_head_dim**-0.5  # 192 nope + 64 rope = 256
        self.spec = cfg.layer(layer)
        self.has_indexer = self.spec.has_indexer_weights
        hw = build_attn_host_weights(cfg, sd, self.cols)
        self.host_w = hw
        wdtypes = (
            wdtypes or {}
        )  # attn-bfp4 task: per-tensor dtype override {q_a,q_b,kv_a,w_uk,w_uv_t,gate,o_proj,idx_wq_b,idx_wk,idx_wp: ttnn dtype}; default OFF = unchanged
        T = lambda k, dt=wdtype, ly=ttnn.TILE_LAYOUT: upload(
            md, hw[k].tensor, wdtypes.get(k, dt), ly, shard_dim=hw[k].shard_dim
        )
        self.w_q_a, self.w_q_b, self.w_kv_a = T("q_a"), T("q_b"), T("kv_a")
        self.w_uk, self.w_uv_t = T("w_uk", ttnn.bfloat16), T("w_uv_t", ttnn.bfloat16)
        self.w_gate, self.w_o = T("gate"), T("o_proj")
        self.q_a_norm = T("q_a_norm", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.kv_a_norm = T("kv_a_norm", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.sink = T(
            "sink_over_scale", sink_over_scale_dtype, ttnn.ROW_MAJOR_LAYOUT
        )  # [1,1,1,64] replicated -> prefill uses all 64 heads (see _sink_full)
        self.sink_dec = T(
            "sink_dec", sink_over_scale_dtype, ttnn.ROW_MAJOR_LAYOUT
        )  # [1,1,1,32] per column: 8 heads + 24 zero pad
        self.sink_full = upload(md, hw["sink_over_scale"].tensor, sink_over_scale_dtype, ttnn.ROW_MAJOR_LAYOUT)
        self.rope = RopeDev(md, cfg.qk_rope_head_dim, cfg.rope_theta, rope_interleave)
        self.idx_rope = RopeDev(md, cfg.qk_rope_head_dim, cfg.rope_theta, idx_rope_interleave)
        self.indexer = (
            Hy4Indexer(md, cfg, hw, self.idx_rope, wdtype=wdtype, wdtypes=wdtypes) if self.has_indexer else None
        )
        self.ckc = compute_cfg(md, "hifi4", True)
        self.pf = pf_opt if pf_opt is not None else PfOpt.from_env()  # HY4_PF_OPT: prefill-only levers, default OFF
        self.ckc_pf = compute_cfg(md, "hifi2", True) if self.pf.hifi2 else self.ckc
        if self.indexer is not None:
            self.indexer.set_pf(self.pf, self.ckc_pf)
        self.ckc_sdpa = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.cache = KVCache(kv_fmt)
        self.taps = None  # dict -> filled with device tensors for dump-diff when set to {}

    # ------------------------------------------------------------------ helpers
    def reset(self):
        if self.cache.rows is not None:
            ttnn.deallocate(self.cache.rows)
        self.cache = KVCache(self.kv_fmt)
        if self.indexer is not None:
            self.indexer.reset()

    def rope_tabs(self, start, n):
        """(cos, sin, idx_cos, idx_sin) device tables for positions start .. start + n (host uploads: call outside a trace)."""
        key = None
        if n >= 256 and os.environ.get("HY4_B49_ROPE_CACHE", "0") == "1":
            # B49: the tables of a prefill chunk are the same for every layer (identical rope parameters): built / uploaded once per chunk (2.7 ms of host time per layer otherwise)
            key = (
                id(self.md),
                start,
                n,
                self.rope.dim,
                self.rope.theta,
                self.rope.interleave,
                self.idx_rope.interleave,
            )
            hit = _ROPE_TABS.get(key)
            if hit is not None:
                c, s, ic, is_ = hit
                if self.has_indexer and ic is None:
                    pos = list(range(start, start + n))
                    ic, is_ = self.idx_rope.tables(pos)
                    _ROPE_TABS[key] = (c, s, ic, is_)
                return (c, s, ic, is_) if self.has_indexer else (c, s, None, None)
        pos = list(range(start, start + n))
        c, s = self.rope.tables(pos)
        if self.has_indexer:
            ic, is_ = self.idx_rope.tables(pos)
        else:
            ic = is_ = None
        if key is not None:
            _ROPE_TABS.clear()  # keep only the current chunk's tables (the previous chunk's are freed with the dict entry)
            _ROPE_TABS[key] = (c, s, ic, is_)
        return c, s, ic, is_

    def _free(self, t, name):
        """deallocate unless the tensor is held as a tap (dump-diff mode)."""
        if self.taps is None or name not in self.taps:
            ttnn.deallocate(t)

    def _tap(self, name, t):
        if self.taps is not None:
            self.taps[name] = t

    def _rms(self, x, w):
        return ttnn.rms_norm(x, weight=w, epsilon=self.cfg.rms_norm_eps, compute_kernel_config=self.ckc)

    def _append_cache(self, lat, k_pe):
        """cache <- [lat | k_pe] rows (v0: concat-grown row-major tensor; the in-place paged write is the integration's job)."""
        n = lat.shape[2]
        if self.kv_fmt == "bf16":
            rows = ttnn.concat([lat, k_pe], dim=3)
            rows = ttnn.to_layout(rows, ttnn.ROW_MAJOR_LAYOUT)
        else:
            dp = ttnn.experimental.deepseek_prefill
            lat_rm = ttnn.to_layout(lat, ttnn.ROW_MAJOR_LAYOUT)
            lat_fp8, scales = dp.per_token_cast_to_fp8(lat, round_scale_to_power_of_two=True)
            rope_rm = ttnn.to_layout(k_pe, ttnn.ROW_MAJOR_LAYOUT)
            rows = dp.pack_scaled_fp8_kv_cache(lat_fp8, scales, rope_rm)
            for t in (lat_rm, lat_fp8, scales, rope_rm):
                ttnn.deallocate(t)
        c = self.cache
        c.rows = rows if c.rows is None else ttnn.concat([c.rows, rows], dim=2)
        c.length += n
        return c.rows

    def _kv_format(self):
        F = ttnn.transformer.SparseKVFormat
        return F.BF16 if self.kv_fmt == "bf16" else F.SCALED_FP8

    # ------------------------------------------------------------------ prefill (one chunk of n rows, positions start .. start+n)
    def prefill_chunk(self, h, start, ids_in=None, ids_override=None, append=True, tabs=None):
        """h [1,1,n,6144] TILE bf16 (n multiple of 512... 8 x 32 rows after the all_to_all); ids_in: top-k ids of the index-source layer (shared layers).
        append=False: the cache / key cache already hold this chunk's rows (re-run for tracing: static shapes, nothing grows); tabs = ``rope_tabs(start, n)`` (made outside a trace).
        Returns (out [1,1,n,6144], ids [1,1,n,2048])."""
        cfg, md = self.cfg, self.md
        n = h.shape[2]
        cos, sin, icos, isin = tabs if tabs is not None else self.rope_tabs(start, n)
        # q_a (N-sharded) -> gather -> norm
        qa_c = plinear(self.pf, h, self.w_q_a, self.ckc_pf)
        qa_g = self.mesh_config.allgather(qa_c, self.ccl, axis=1, dim=3)
        ttnn.deallocate(qa_c)
        qa = self._rms(qa_g, self.q_a_norm)
        ttnn.deallocate(qa_g)
        self._tap("qa", qa)
        # kv latent + rope key -> cache
        kv = plinear(self.pf, h, self.w_kv_a, self.ckc_pf)  # [1,1,n,576]
        lat = self._rms(ttnn.slice(kv, [0, 0, 0, 0], [1, 1, n, cfg.kv_lora_rank]), self.kv_a_norm)
        k_pe = self.rope.apply(ttnn.slice(kv, [0, 0, 0, cfg.kv_lora_rank], [1, 1, n, cfg.kv_a_out]), cos, sin, self.ckc)
        ttnn.deallocate(kv)
        self._tap("lat", lat)
        self._tap("k_pe", k_pe)
        kvc = self._append_cache(lat, k_pe) if append else self.cache.rows
        # indices
        if self.has_indexer:
            ids = self.indexer.select(h, qa, start, icos, isin, self.taps, append=append)
        else:
            assert ids_in is not None, "shared-indexer layer needs the ids of its index source"
            ids = ids_in
        self._tap("ids", ids)
        if ids_override is not None:  # experiments: attention uses these ids instead
            ids = ids_override
        # query
        q = plinear(self.pf, qa, self.w_q_b, self.ckc_pf)  # [1,1,n,2048]
        self._free(qa, "qa")
        q_nope, q_pe = ttnn.experimental.nlp_create_q_heads_split(
            q, num_heads=self.hc, split_head_dim=cfg.qk_nope_head_dim, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [1,8,n,192], [1,8,n,64]
        ttnn.deallocate(q)
        q_abs = ttnn.matmul(q_nope, self.w_uk, compute_kernel_config=self.ckc_pf)  # [1,8,n,512]
        ttnn.deallocate(q_nope)
        q_pe = self.rope.apply(q_pe, cos, sin, self.ckc)
        qfull = ttnn.concat([q_abs, q_pe], dim=3)  # [1,8,n,576]
        ttnn.deallocate(q_abs)
        ttnn.deallocate(q_pe)
        self._tap("qfull", qfull)
        o_lat = self._sparse_prefill(qfull, kvc, ids)  # [1,8,n,512]
        self._free(qfull, "qfull")
        self._tap("o_lat", o_lat)
        out = self._finish(o_lat, h)
        return out, ids

    def _sparse_prefill(self, qfull, kvc, ids):
        """heads->sequence all_to_all over the columns, stock sparse_sdpa on H=64 x n/8 rows, back."""
        n = qfull.shape[2]
        q_t = ttnn.experimental.all_to_all_async_generic(
            qfull,
            in_dim=1,
            out_dim=2,
            num_links=2 if self.pf.a2a2 else 1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=1,
        )  # [1,64,n/8,576] TILE (the op takes TILE only)
        q_seq = ttnn.to_layout(q_t, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(q_t)
        idx_c = ttnn.mesh_partition(ids, dim=2, cluster_axis=1)  # this column's n/8 rows
        out = ttnn.transformer.sparse_sdpa(
            q_seq,
            kvc,
            idx_c,
            self.cfg.kv_lora_rank,
            kv_format=self._kv_format(),
            scale=self.scale,
            k_chunk_size=64 if self.pf.kc64 else 128,
            compute_kernel_config=self.ckc_sdpa,
            attention_sink=self.sink_full,
        )  # [1,64,n/8,512] RM
        ttnn.deallocate(q_seq)
        ttnn.deallocate(idx_c)
        out_t = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        ttnn.deallocate(out)
        o_lat = ttnn.experimental.all_to_all_async_generic(
            out_t,
            in_dim=2,
            out_dim=1,
            num_links=2 if self.pf.a2a2 else 1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=1,
        )  # [1,8,n,512]
        ttnn.deallocate(out_t)
        return o_lat

    def _finish(self, o_lat, h):
        """o_lat [1,8,n,512] -> value up-projection, gate, o_proj, all-reduce -> [1,1,n,6144]."""
        pf = o_lat.shape[2] >= 256  # prefill rows: HY4_PF_OPT levers (decode keeps the stock path)
        ckc = self.ckc_pf if pf else self.ckc
        o = ttnn.matmul(o_lat, self.w_uv_t, compute_kernel_config=ckc)  # [1,8,n,256]
        o = ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,n,2048]
        self._tap("o", o)
        g = plinear(self.pf if pf else None, h, self.w_gate, ckc, act="sigmoid")
        og = ttnn.multiply(o, g)
        self._tap("gate", g)
        self._free(g, "gate")
        part = plinear(self.pf if pf else None, og, self.w_o, ckc)  # [1,1,n,6144] partial
        self._tap("og", og)
        self._free(og, "og")
        if self.taps is not None:
            self.taps["part"] = ttnn.clone(part)  # the all-reduce may consume its input
        res = self.mesh_config.allreduce(part, self.ccl, axis=1)
        return res

    # ------------------------------------------------------------------ decode (one token per user = per mesh row; all rows at the same position in v0)
    def dense_ids(self, pos):
        """ids [1,1,1,2048] for a context of pos + 1 <= 2048 keys: 0..pos then the sentinel (host-built constant: v0 / tests; an on-device iota is the production form)."""
        import torch

        ix = torch.full((1, 1, 1, self.cfg.index_topk), -1, dtype=torch.int32)
        ix[..., : pos + 1] = torch.arange(pos + 1, dtype=torch.int32)
        return upload(self.md, ix, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    def decode_step(self, h, pos, ids_in=None, v1=False, append=True, tabs=None, dense_ids=None):
        """h [1,1,1,6144] TILE bf16 (one user per mesh row; rows may differ). The cache / key cache hold the ``pos`` earlier rows. Returns (out [1,1,1,6144], ids)."""
        cfg = self.cfg
        cos, sin, icos, isin = tabs if tabs is not None else self.rope_tabs(pos, 1)
        qa_c = ttnn.linear(h, self.w_q_a, compute_kernel_config=self.ckc)
        qa_g = self.mesh_config.allgather(qa_c, self.ccl, axis=1, dim=3)
        ttnn.deallocate(qa_c)
        qa = self._rms(qa_g, self.q_a_norm)
        ttnn.deallocate(qa_g)
        kv = ttnn.linear(h, self.w_kv_a, compute_kernel_config=self.ckc)
        lat = self._rms(ttnn.slice(kv, [0, 0, 0, 0], [1, 1, 1, cfg.kv_lora_rank]), self.kv_a_norm)
        k_pe = self.rope.apply(ttnn.slice(kv, [0, 0, 0, cfg.kv_lora_rank], [1, 1, 1, cfg.kv_a_out]), cos, sin, self.ckc)
        ttnn.deallocate(kv)
        self._tap("lat", lat)
        self._tap("k_pe", k_pe)
        kvc = self._append_cache(lat, k_pe) if append else self.cache.rows
        if self.has_indexer:
            if pos + 1 <= cfg.index_topk:  # dense: every causal key is selected; the indexer only appends its key
                if append:
                    self.indexer.append_keys(h, icos, isin)
                ids = dense_ids if dense_ids is not None else self.dense_ids(pos)
            else:
                ids = self.indexer.select_decode(h, qa, pos, icos, isin, append=append)
        else:
            assert ids_in is not None
            ids = ids_in
        self._tap("ids", ids)
        q = ttnn.linear(qa, self.w_q_b, compute_kernel_config=self.ckc)  # [1,1,1,2048]
        ttnn.deallocate(qa)
        q_nope, q_pe = ttnn.experimental.nlp_create_q_heads_split(
            q, num_heads=self.hc, split_head_dim=cfg.qk_nope_head_dim, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(q)
        q_abs = ttnn.matmul(q_nope, self.w_uk, compute_kernel_config=self.ckc)
        ttnn.deallocate(q_nope)
        q_pe = self.rope.apply(q_pe, cos, sin, self.ckc)
        qfull = ttnn.concat([q_abs, q_pe], dim=3)  # [1,8,1,576]
        ttnn.deallocate(q_abs)
        ttnn.deallocate(q_pe)
        self._tap("qfull", qfull)
        o_lat = self._sparse_decode(qfull, kvc, ids, v1)  # [1,8,1,512]
        self._tap("o_lat", o_lat)
        out = self._finish(o_lat, h)
        return out, ids

    def _sparse_decode(self, qfull, kvc, ids, v1=False):
        """8 local heads zero-padded to the 32 the op needs (pad heads cost no latency: one query row per core), stock op (v0) or the DSV4 key-split kernel (v1, bf16 KV only)."""
        qp = ttnn.pad(qfull, [(0, 0), (0, 32 - self.hc), (0, 0), (0, 0)], 0)
        q_rm = ttnn.to_layout(qp, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(qp)
        if v1:
            from models.demos.blackhole.deepseek_v41_flash.tt import sparse_split as SS

            assert self.kv_fmt == "bf16", "the DSV4 split kernel reads bf16 / plain fp8 caches only (no scaled FP8)"
            out = SS.sparse_sdpa_split(
                q_rm, kvc, ids, self.cfg.kv_lora_rank, self.scale, 128, 8, self.ckc_sdpa, self.sink_dec
            )
        else:
            out = ttnn.transformer.sparse_sdpa(
                q_rm,
                kvc,
                ids,
                self.cfg.kv_lora_rank,
                kv_format=self._kv_format(),
                scale=self.scale,
                k_chunk_size=128,
                compute_kernel_config=self.ckc_sdpa,
                attention_sink=self.sink_dec,
            )  # [1,32,1,512] RM
        ttnn.deallocate(q_rm)
        o8 = ttnn.slice(out, [0, 0, 0, 0], [1, self.hc, 1, self.cfg.kv_lora_rank])
        ttnn.deallocate(out)
        o_t = ttnn.to_layout(o8, ttnn.TILE_LAYOUT)
        ttnn.deallocate(o8)
        return o_t
