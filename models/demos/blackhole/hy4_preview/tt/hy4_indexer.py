# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4 DSA lightning indexer of ONE full-indexer layer (32 heads x 128, top-2048), device, mesh 4x8 (all weights replicated: every column scores the whole row).

    q = rope_first64(wq_b(qa)) [32,128]      k = rope_first64(layer_norm(wk(h))) [128]      w = weights_proj(h) * 32^-.5 * 128^-.5 (folded into the weight)
    score[s,t] = sum_h relu(q[s,h] . k[t]) * w[s,h]   (t <= pos(s));  idx[s] = the 2048 best t (sentinel 0xFFFFFFFF where fewer exist)

Ops: ttnn.experimental.indexer_score_dsa (causal -inf fused) + ttnn.experimental.topk_large_indices (k in [16, 2048], valid_length >= k). For a context shorter than 2048
keys, k = the key count (multiple of 16) and the ids are padded to 2048 with the sentinel. Key cache: bf16 TILE [1,1,T,128] grown by concat (v0; paged / in-place later).
Shared layers do not run this: they reuse the ids of ``cfg.indexer_source(layer)`` (see ``Hy4Attention``).
The oracle omits the V3.2 Hadamard rotation of q and k (invariant for q.k in exact arithmetic); so does this.
"""
import os

import ttnn
from models.demos.blackhole.hy4_preview.tt.hy4_common import SENTINEL, compute_cfg, upload  # noqa: F401
from models.demos.blackhole.hy4_preview.tt.hy4_pf_opt import PfOpt, plinear

_C128 = {}  # B49 (HY4_B49_ROPE_CACHE=1): (id(cos), idx_rope_first) -> (cos, sin, cos128, sin128) of the current chunk


class Hy4Indexer:
    def __init__(self, md, cfg, host_w, rope, idx_rope_first=False, ln_eps=1e-6, wdtype=ttnn.bfloat16, wdtypes=None):
        wdtypes = wdtypes or {}
        self.md, self.cfg, self.rope, self.idx_rope_first, self.ln_eps = md, cfg, rope, idx_rope_first, ln_eps
        self.nh, self.d, self.rd, self.topk = (
            cfg.index_n_heads,
            cfg.index_head_dim,
            cfg.qk_rope_head_dim,
            cfg.index_topk,
        )
        w = lambda k, dt=wdtype, ly=ttnn.TILE_LAYOUT: upload(md, host_w[k].tensor, wdtypes.get(k, dt), ly)
        self.w_q = w("idx_wq_b")
        self.w_k = w("idx_wk")
        self.w_p = w("idx_wp", ttnn.bfloat16)
        self.k_norm_w = w("idx_k_norm_w", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.k_norm_b = w("idx_k_norm_b", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.ckc = compute_cfg(md, "hifi4", True)
        self.ckc_score = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        self._mask = {}
        self._mask_full = {}
        self._sent = {}
        self._c128 = {}
        self._tm = None
        self.pf, self.ckc_pf = PfOpt(), self.ckc  # set_pf() by Hy4Attention (HY4_PF_OPT); default OFF
        self.keys = None  # [1,1,T,128] TILE bf16
        self.length = 0

    def set_pf(self, pf, ckc_pf):
        self.pf, self.ckc_pf = pf, ckc_pf

    def _rot128(self, x, cos, sin):
        """rope1: ONE rotary_embedding_llama over the whole 128-wide head; cos = 1 / sin = 0 on the non-rope span, so that span passes through unchanged (adjacent-pair rope only)."""
        import torch

        n = x.shape[2]
        assert self.rope.interleave, "rope1 needs the adjacent-pair indexer rope"
        if self._tm is None:
            from models.tt_transformers.tt.common import get_rot_transformation_mat

            self._tm = upload(self.md, get_rot_transformation_mat())
        if n not in self._c128:
            self._c128[n] = (
                upload(self.md, torch.ones(1, 1, n, self.d - self.rd)),
                upload(self.md, torch.zeros(1, 1, n, self.d - self.rd)),
            )
        ones, zeros = self._c128[n]
        if os.environ.get("HY4_B49_ROPE_CACHE", "0") == "1":
            # B49: the 128-wide tables of a chunk are built once (append_keys + select of every full-indexer layer share them; the rope-table cache hands every layer the same cos / sin objects)
            key = (id(cos), self.idx_rope_first)
            ent = _C128.get(key)
            if ent is None:
                c = ttnn.concat([cos, ones] if self.idx_rope_first else [ones, cos], dim=3)
                s_ = ttnn.concat([sin, zeros] if self.idx_rope_first else [zeros, sin], dim=3)
                _C128.clear()
                ent = _C128[key] = (
                    cos,
                    sin,
                    c,
                    s_,
                )  # (cos, sin) kept alive so that id(cos) cannot be reused while cached
            return ttnn.experimental.rotary_embedding_llama(
                x, ent[2], ent[3], self._tm, is_decode_mode=False, compute_kernel_config=self.ckc
            )
        c = ttnn.concat([cos, ones] if self.idx_rope_first else [ones, cos], dim=3)
        s_ = ttnn.concat([sin, zeros] if self.idx_rope_first else [zeros, sin], dim=3)
        out = ttnn.experimental.rotary_embedding_llama(
            x, c, s_, self._tm, is_decode_mode=False, compute_kernel_config=self.ckc
        )
        ttnn.deallocate(c)
        ttnn.deallocate(s_)
        return out

    def _rot(self, x, cos, sin):
        return self._rot128(x, cos, sin) if (self.pf.rope1 and x.shape[2] >= 256) else self._rot_first(x, cos, sin)

    def _causal_mask1(self, score, start, n, T):
        """mask1: one add with a [n, T] mask (-inf above the diagonal of the last n columns), no slice / concat."""
        import torch

        key = (n, T, start)
        if key not in self._mask_full:
            for (
                old
            ) in (
                self._mask_full.values()
            ):  # one mask only: a per-(start, T) cache leaked n*T*2 bytes per chunk and layer (OOM in the 60k real-prefill needle run)
                ttnn.deallocate(old)
            self._mask_full.clear()
            m = torch.zeros(n, T)
            m[:, start : start + n] = torch.zeros(n, n).masked_fill(
                torch.ones(n, n, dtype=torch.bool).triu(1), float("-inf")
            )
            self._mask_full[key] = upload(self.md, m.reshape(1, 1, n, T), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.add(score, self._mask_full[key])
        ttnn.deallocate(score)
        return out

    def reset(self):
        if self.keys is not None:
            ttnn.deallocate(self.keys)
        self.keys, self.length = None, 0

    def _rot_first(self, x, cos, sin):
        """rope on the first (or last) 64 of the 128 dims of x [1,H,n,128]."""
        n = x.shape[2]
        h = x.shape[1]
        sl = (0, self.rd) if self.idx_rope_first else (self.d - self.rd, self.d)
        rest = (self.rd, self.d) if self.idx_rope_first else (0, self.d - self.rd)
        pe = ttnn.slice(x, [0, 0, 0, sl[0]], [1, h, n, sl[1]])
        pe = self.rope.apply(pe, cos, sin, self.ckc)
        nope = ttnn.slice(x, [0, 0, 0, rest[0]], [1, h, n, rest[1]])
        out = ttnn.concat([pe, nope] if self.idx_rope_first else [nope, pe], dim=3)
        ttnn.deallocate(pe)
        ttnn.deallocate(nope)
        return out

    def _causal_fix(self, score, start, n, T):
        """indexer_score_dsa derives each device's causal offset from its LINEAR mesh index (chunk_start + index * Sq) unless seq_shard_axes names an axis, so on the 4x8 mesh
        only device 0 masks exactly (probe found ids of columns 1..7 unmasked). Re-apply the in-chunk causal block mask ourselves: columns [start, start+n) get -inf above the
        diagonal (columns before it are always visible; columns past start+n are never written). v0: one slice + add + concat (a fused fix belongs in the op).
        """
        import torch

        if n not in self._mask:
            m = torch.zeros(n, n).masked_fill(torch.ones(n, n, dtype=torch.bool).triu(1), float("-inf"))
            self._mask[n] = upload(self.md, m.reshape(1, 1, n, n), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        tail = ttnn.slice(score, [0, 0, 0, start], [1, 1, n, start + n])
        tail = ttnn.add(tail, self._mask[n])
        if start == 0:
            out = tail
        else:
            head = ttnn.slice(score, [0, 0, 0, 0], [1, 1, n, start])
            out = ttnn.concat([head, tail], dim=3)
            ttnn.deallocate(head)
            ttnn.deallocate(tail)
        ttnn.deallocate(score)
        return out

    def append_keys(self, h, cos, sin):
        """h [1,1,n,6144] -> k rows [1,1,n,128] (layer_norm + rope) appended to the key cache; returns the new rows."""
        n = h.shape[2]
        pf = n >= 256
        k = plinear(self.pf if pf else None, h, self.w_k, self.ckc_pf if pf else self.ckc)
        k = ttnn.layer_norm(k, weight=self.k_norm_w, bias=self.k_norm_b, epsilon=self.ln_eps)
        k = self._rot(k, cos, sin)
        self.keys = k if self.keys is None else ttnn.concat([self.keys, k], dim=2)
        self.length += n
        return k

    def select_decode(self, h, qa, pos, cos, sin, append=True):
        """One query row per user (position ``pos``, this call's key appended). Requires pos + 1 >= 2048 (shorter contexts are dense: see ``Hy4Attention.decode_step``).
        Scores the key cache padded to a whole number of tiles; the top-k bound is valid_length = pos + 1 (padded / stale columns are never ranked).
        """
        if append:
            self.append_keys(h, cos, sin)
        T = self.length
        assert T == pos + 1 and T >= self.topk
        q = ttnn.linear(qa, self.w_q, compute_kernel_config=self.ckc)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=self.nh, num_kv_heads=0, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        q = self._rot_first(q, cos, sin)
        w = ttnn.linear(h, self.w_p, compute_kernel_config=self.ckc)
        Tp = -(-T // 32) * 32
        keys = _pad_rows(self.keys, Tp)
        q32, w32 = _pad_rows(q, 32), _pad_rows(w, 32)  # the op wants whole tiles of query rows: row 0 is the real one
        cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=min(Tp, 256), head_group_size=1)
        score32 = ttnn.experimental.indexer_score_dsa(
            q32, keys, w32, chunk_start_idx=pos, program_config=cfg, compute_kernel_config=self.ckc_score, kv_len=Tp
        )  # [1,1,32,Tp]; columns past pos are not ranked (valid_length below)
        score = ttnn.slice(score32, [0, 0, 0, 0], [1, 1, 1, Tp])
        ttnn.deallocate(score32)
        for t in (q32, w32):
            ttnn.deallocate(t)
        ids = ttnn.experimental.topk_large_indices(score, k=self.topk, valid_length=T)
        for t in (q, w, score):
            ttnn.deallocate(t)
        del keys  # NOT deallocate: ttnn.pad of a tile-padded tensor may return a view of self.keys (explicit deallocate freed the shared buffer)
        return ids

    def select(self, h, qa, start, cos, sin, taps=None, append=True):
        """Score the n query rows of ``h`` (positions start .. start+n) against the key cache (this call's keys are appended first) and return their top-k ids
        uint32 RM [1,1,n,2048]."""
        n = h.shape[2]
        if append:
            self.append_keys(h, cos, sin)
        T = self.length
        assert T % 16 == 0 and start + n == T, (start, n, T)
        q = plinear(self.pf, qa, self.w_q, self.ckc_pf)  # [1,1,n,4096]
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=self.nh, num_kv_heads=0, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [1,32,n,128]
        q = self._rot(q, cos, sin)
        w = plinear(self.pf, h, self.w_p, self.ckc_pf)  # [1,1,n,32]
        cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=min(T, 256), head_group_size=1)
        score = ttnn.experimental.indexer_score_dsa(
            q, self.keys, w, chunk_start_idx=start, program_config=cfg, compute_kernel_config=self.ckc_score, kv_len=T
        )  # [1,1,n,T] bf16 RM, future columns -inf
        score = self._causal_mask1(score, start, n, T) if self.pf.mask1 else self._causal_fix(score, start, n, T)
        k_eff = min(self.topk, T)
        ids = ttnn.experimental.topk_large_indices(score, k=k_eff, valid_length=T)
        if taps is not None:
            taps["idx_score"] = score
        else:
            ttnn.deallocate(score)
        for t in (q, w):
            ttnn.deallocate(t)
        if k_eff < self.topk:
            if (n, self.topk - k_eff) not in self._sent:
                self._sent[(n, self.topk - k_eff)] = upload_sentinel(self.md, n, self.topk - k_eff)
            pad = self._sent[(n, self.topk - k_eff)]
            ids2 = ttnn.concat([ids, pad], dim=3)
            ttnn.deallocate(ids)
            ids = ids2
        return ids


def upload_sentinel(md, n, width):
    import torch

    t = torch.full((1, 1, n, width), -1, dtype=torch.int32)  # -1 == 0xFFFFFFFF as uint32
    return upload(md, t, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)


def _pad_rows(t, to):
    n = t.shape[2]
    if n == to:
        return t
    return ttnn.pad(t, [(0, 0), (0, 0), (0, to - n), (0, 0)], 0)
