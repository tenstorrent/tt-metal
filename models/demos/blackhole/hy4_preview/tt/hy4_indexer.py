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
import ttnn
from models.demos.blackhole.hy4_preview.tt.hy4_common import SENTINEL, compute_cfg, upload  # noqa: F401


class Hy4Indexer:
    def __init__(self, md, cfg, host_w, rope, idx_rope_first=True, ln_eps=1e-6, wdtype=ttnn.bfloat16):
        self.md, self.cfg, self.rope, self.idx_rope_first, self.ln_eps = md, cfg, rope, idx_rope_first, ln_eps
        self.nh, self.d, self.rd, self.topk = (
            cfg.index_n_heads,
            cfg.index_head_dim,
            cfg.qk_rope_head_dim,
            cfg.index_topk,
        )
        w = lambda k, dt=wdtype, ly=ttnn.TILE_LAYOUT: upload(md, host_w[k].tensor, dt, ly)
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
        self.keys = None  # [1,1,T,128] TILE bf16
        self.length = 0

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

    def append_keys(self, h, cos, sin):
        """h [1,1,n,6144] -> k rows [1,1,n,128] (layer_norm + rope) appended to the key cache; returns the new rows."""
        n = h.shape[2]
        k = ttnn.linear(h, self.w_k, compute_kernel_config=self.ckc)
        k = ttnn.layer_norm(k, weight=self.k_norm_w, bias=self.k_norm_b, epsilon=self.ln_eps)
        k = self._rot_first(k, cos, sin)
        self.keys = k if self.keys is None else ttnn.concat([self.keys, k], dim=2)
        self.length += n
        return k

    def select(self, h, qa, start, cos, sin, taps=None):
        """Score the n query rows of ``h`` (positions start .. start+n) against the key cache (this call's keys are appended first) and return their top-k ids
        uint32 RM [1,1,n,2048]."""
        n = h.shape[2]
        self.append_keys(h, cos, sin)
        T = self.length
        assert T % 16 == 0 and start + n == T, (start, n, T)
        q = ttnn.linear(qa, self.w_q, compute_kernel_config=self.ckc)  # [1,1,n,4096]
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=self.nh, num_kv_heads=0, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [1,32,n,128]
        q = self._rot_first(q, cos, sin)
        w = ttnn.linear(h, self.w_p, compute_kernel_config=self.ckc)  # [1,1,n,32]
        cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=min(T, 256), head_group_size=1)
        score = ttnn.experimental.indexer_score_dsa(
            q, self.keys, w, chunk_start_idx=start, program_config=cfg, compute_kernel_config=self.ckc_score, kv_len=T
        )  # [1,1,n,T] bf16 RM, future columns -inf
        k_eff = min(self.topk, T)
        ids = ttnn.experimental.topk_large_indices(score, k=k_eff, valid_length=T)
        if taps is not None:
            taps["idx_score"] = score
        else:
            ttnn.deallocate(score)
        for t in (q, w):
            ttnn.deallocate(t)
        if k_eff < self.topk:
            pad = upload_sentinel(self.md, n, self.topk - k_eff)
            ids2 = ttnn.concat([ids, pad], dim=3)
            ttnn.deallocate(ids)
            ttnn.deallocate(pad)
            ids = ids2
        return ids


def upload_sentinel(md, n, width):
    import torch

    t = torch.full((1, 1, n, width), -1, dtype=torch.int32)  # -1 == 0xFFFFFFFF as uint32
    return upload(md, t, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
