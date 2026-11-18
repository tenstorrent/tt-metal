# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4 attention decode with flag-gated fusions (task layer-other phase 2). Every flag defaults OFF; ``Opt.all_on()`` is the recommended stack.

Flags (env ``HY4_OPT`` = comma list or ``all``; or pass an ``Opt``):
  tune      DSV4-style 1D-multicast program configs for the M = 1..4 row projections (q_a, kv_a, gate, q_b, o_proj, idx_wq_b, idx_wk, idx_wp)
  l1        decode activations / matmul outputs / CCL outputs L1-resident (the o_lat x w_uv batched matmul writes 4x faster to L1)
  rmssh     q_a rms_norm width-sharded over 16 cores (34 -> ~11 us incl. the reshard)
  rope1     every rope (q_pe, k_pe, indexer q and k) as ONE ``rotary_embedding_llama`` op instead of matmul+mul+mul+add (+ slices/concat for the indexer).
            The op is the interleaved-pair (Meta) rotation: the checkpoint's conv has rope_interleave = idx_rope_interleave = True, so NO weight permutation is needed here;
            ``half_to_interleave_perm`` is the offline permutation for a half-split (rotate_half) checkpoint (CPU-tested), asserted off otherwise.
  fuseproj  q_a | kv_a | gate [| idx_wk | idx_wp on full-indexer layers] as ONE linear on a column-concatenated weight [6144, 3040] (per column c: q_a[:, 256c:256c+256], kv_a (replicated),
            gate[:, 2048c:2048c+2048], idx_wk, idx_wp), sliced afterwards. One dtype for all parts (the layer's ``wdtype``): mixed per-tensor dtypes (abf4 ``wdtypes``) cannot fuse.
The module is a subclass of ``Hy4Attention`` (prefill and the unfused ``decode_step`` are unchanged and remain the reference path).
"""
import os
from dataclasses import dataclass, fields

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt.hy4_attention import Hy4Attention
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload

CAPS = {  # per-device (K, N, max cores) of the decode projections
    "q_a": (6144, 256, 64),
    "q_b": (2048, 2048, 32),
    "kv_a": (6144, 576, 32),
    "gate": (6144, 2048, 32),
    "o_proj": (2048, 6144, 32),
    "idx_wq_b": (2048, 4096, 32),
    "idx_wk": (6144, 128, 64),
    "idx_wp": (6144, 32, 32),
}


def half_to_interleave_perm(dim=128, rope_dim=64, first=True):
    """Index permutation p over ``dim`` columns (new[j] = old[p[j]]) taking a half-split (x[i], x[i+h]) rope pairing of the rope span to the adjacent-pair (x[2i], x[2i+1]) pairing.
    Apply to the output columns of the weights that feed the rope span (per head for q: wq_b; wk) and to the layer-norm weight / bias between wk and the rope; the nope span is untouched.
    """
    h = rope_dim // 2
    p = list(range(dim))
    base = 0 if first else dim - rope_dim
    for i in range(h):
        p[base + 2 * i] = base + i
        p[base + 2 * i + 1] = base + i + h
    return torch.tensor(p)


@dataclass
class Opt:
    tune: bool = False
    l1: bool = False
    rmssh: bool = False
    rope1: bool = False
    fuseproj: bool = False

    @classmethod
    def all_on(cls):
        return cls(*(True for _ in fields(cls)))

    @classmethod
    def parse(cls, s):
        s = (s or "").strip()
        if s == "all":
            return cls.all_on()
        names = {x for x in s.split(",") if x}
        bad = names - {f.name for f in fields(cls)}
        assert not bad, bad
        return cls(**{n: True for n in names})

    @classmethod
    def from_env(cls):
        return cls.parse(os.environ.get("HY4_OPT", ""))

    def tag(self):
        return ",".join(f.name for f in fields(self) if getattr(self, f.name)) or "base"


def cfg_1d(K, N, max_cores, act=None):
    Kt, Nt = K // 32, N // 32
    pcn = -(-Nt // max_cores)
    n = -(-Nt // pcn)
    gx, gy = min(n, 8), -(-n // 8)
    sw = max(d for d in range(1, min(pcn, 4) + 1) if pcn % d == 0)
    ibw = max(d for d in range(1, 9) if Kt % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=ibw,
        out_subblock_h=1,
        out_subblock_w=sw,
        per_core_M=1,
        per_core_N=pcn,
        fuse_batch=True,
        fused_activation=act,
        mcast_in0=True,
    )


class Hy4AttentionOpt(Hy4Attention):
    def __init__(self, *a, fuse_dtype=None, **kw):
        super().__init__(*a, **kw)
        md, cfg, hw = self.md, self.cfg, self.host_w
        wd = fuse_dtype or kw.get("wdtype", ttnn.bfloat16)
        self._fuse_dtype = wd
        self._nloc = 256 + 576 + 2048 + ((128 + 32) if self.has_indexer else 0)
        self._offs = [0, 256, 256 + 576, 256 + 576 + 2048] + (
            [256 + 576 + 2048 + 128, self._nloc] if self.has_indexer else [self._nloc]
        )
        self._wf_host_ = None  # host concat built lazily (HY4_OPT=fuseproj / HY4_SKINNY=hproj only): 300 MB per layer
        self._wf = None  # built lazily (device memory only when the flag is used)
        self._tm = None
        self._rms_cfg = None

    # ---------------------------------------------------------------- helpers
    @property
    def _wf_host(self):
        if self._wf_host_ is None:
            hw = self.host_w
            parts = []
            for c in range(self.cols):
                p_ = [
                    hw["q_a"].tensor[..., c * 256 : (c + 1) * 256],
                    hw["kv_a"].tensor,
                    hw["gate"].tensor[..., c * 2048 : (c + 1) * 2048],
                ]
                if self.has_indexer:
                    p_ += [hw["idx_wk"].tensor, hw["idx_wp"].tensor]
                parts.append(torch.cat(p_, dim=-1))
            self._wf_host_ = torch.cat(parts, dim=-1)
        return self._wf_host_

    def _skinny_o(self, og):
        """HY4_SKINNY=oproj (default OFF): o_proj on the bank-local streaming kernel (tt/hy4_skinny.SkinnyLinear, T <= 16). Builds a second (bank-layout) copy of the o_proj weights
        (13.4 MB bfp8 per chip per layer) on first use: call once outside a trace (warmup). Returns None when the flag is off or T > 16.
        """
        from models.demos.blackhole.hy4_preview.tt.hy4_skinny import SkinnyLinear, enabled

        T = int(og.shape[2])
        if not enabled("oproj") or T > 16:
            return None
        if getattr(self, "_sk_o", None) is None:
            self._sk_o = SkinnyLinear(self.md, self.host_w["o_proj"].tensor.reshape(-1, self.cfg.hidden_size))
        return self._sk_o(og, T)

    def _skinny_proj(self, h):
        """HY4_SKINNY=hproj (default OFF): the fused h-proj on the bank-local streaming kernel (tt/hy4_skinny.SkinnyProj, T <= 16); returns the part tensors [q_a, kv_a, gate(, idx_wk, idx_wp)] or None.
        Builds a second (bank-layout) copy of the fused weight (19.8 MB bfp8 per chip per layer) on first use: call once outside a trace (warmup).
        """
        from models.demos.blackhole.hy4_preview.tt.hy4_skinny import SkinnyProj, enabled

        S = int(h.shape[2])
        if not enabled("hproj") or S > 16:
            return None
        if getattr(self, "_sk_p", None) is None:
            of = self._offs
            self._sk_p = SkinnyProj(self.md, self._wf_host, [of[i + 1] - of[i] for i in range(len(of) - 1)])
        return self._sk_p(h, S)

    def _lazy_fused(self):
        if self._wf is None:
            self._wf = upload(self.md, self._wf_host, self._fuse_dtype, ttnn.TILE_LAYOUT, shard_dim=3)
            self._wf_pc = cfg_1d(6144, self._nloc, 32)
        return self._wf

    def _lazy_rope1(self):
        if self._tm is None:
            from models.tt_transformers.tt.common import get_rot_transformation_mat

            assert (
                self.rope.interleave
            ), "rope1 needs the adjacent-pair (interleaved) MLA rope; permute the weights (half_to_interleave_perm) for a half-split checkpoint"
            if self.has_indexer:
                assert (
                    self.idx_rope.interleave
                ), "rope1: indexer rope must be interleaved (permute idx weights with half_to_interleave_perm)"
            self._tm = upload(self.md, get_rot_transformation_mat())

    def _lazy_rms(self):
        if self._rms_cfg is None:
            mc = ttnn.create_sharded_memory_config(
                shape=(32, 128),
                core_grid=ttnn.CoreGrid(y=2, x=8),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(8, 2), subblock_w=4, block_h=1, block_w=4, inplace=False
            )
            self._rms_cfg = (mc, pc)
        return self._rms_cfg

    def rope_tabs_opt(self, start, n):
        """rope_tabs + (indexer cos/sin 128 wide, identity on the non-rope span) for rope1 (host uploads: outside a trace)."""
        cos, sin, icos, isin = self.rope_tabs(start, n)
        if not (self.has_indexer):
            return cos, sin, icos, isin, None, None
        ones = upload(self.md, torch.ones(1, 1, n, 64))
        zeros = upload(self.md, torch.zeros(1, 1, n, 64))
        first = self.indexer.idx_rope_first
        c128 = ttnn.concat([icos, ones] if first else [ones, icos], dim=3)
        s128 = ttnn.concat([isin, zeros] if first else [zeros, isin], dim=3)
        return cos, sin, icos, isin, c128, s128

    def _lin(self, o, x, w, name, act=None, ckc=None):
        ckc = ckc or self.ckc
        mc = ttnn.L1_MEMORY_CONFIG if o.l1 else None
        if o.tune and x.shape[2] <= 128:
            K, N, cap = CAPS[name]
            pc = cfg_1d(K, N, cap, ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID) if act == "sigmoid" else None)
            return ttnn.linear(x, w, compute_kernel_config=ckc, program_config=pc, memory_config=mc)
        return ttnn.linear(x, w, compute_kernel_config=ckc, activation=act, memory_config=mc)

    def _rope(self, o, x, cos, sin):
        if o.rope1:
            return ttnn.experimental.rotary_embedding_llama(
                x, cos, sin, self._tm, is_decode_mode=False, compute_kernel_config=self.ckc
            )
        return self.rope.apply(x, cos, sin, self.ckc)

    def _idx_rot(self, o, x, cos, sin, c128, s128):
        if o.rope1:
            return ttnn.experimental.rotary_embedding_llama(
                x, c128, s128, self._tm, is_decode_mode=False, compute_kernel_config=self.ckc
            )
        return self.indexer._rot_first(x, cos, sin)

    # ---------------------------------------------------------------- decode
    def front(self, h, o, tabs, append_cache=True, qrm=False, kvf=None, qpush=None):
        """h [1,1,S,6144] -> dict(qa, lat, k_pe, qfull, g, ik, iw). Everything of the decode layer before the sparse kernel (plus the gate of the o path)."""
        cfg, ix = self.cfg, self.indexer
        S = h.shape[2]
        cos, sin, icos, isin, c128, s128 = tabs
        mc = ttnn.L1_MEMORY_CONFIG if o.l1 else None
        d = {}
        if o.rope1:
            self._lazy_rope1()
        if o.fuseproj:
            sl = self._skinny_proj(
                h
            )  # HY4_SKINNY=hproj (default OFF): bank-local kernel, output already split (no slice ops)
            if sl is None:
                wf = self._lazy_fused()
                fp = ttnn.linear(
                    h,
                    wf,
                    compute_kernel_config=self.ckc,
                    program_config=self._wf_pc if S <= 4 else None,
                    memory_config=mc,
                )
                of = self._offs
                sl = [ttnn.slice(fp, [0, 0, 0, of[i]], [1, 1, S, of[i + 1]]) for i in range(len(of) - 1)]
                ttnn.deallocate(fp)
            qa_c, kv, g_raw = sl[0], sl[1], sl[2]
            if (
                os.environ.get("HY4_GATE_FOLD", "0") == "1"
            ):  # sigmoid applied inside the o-path multiply (back(..., gfold=True)): one op less per layer
                d["g"], d["gfold"] = g_raw, True
            else:
                d["g"] = ttnn.sigmoid(g_raw)
                ttnn.deallocate(g_raw)
            if self.has_indexer:
                d["ik"], d["iw"] = sl[3], sl[4]
        else:
            qa_c = self._lin(o, h, self.w_q_a, "q_a")
            kv = self._lin(o, h, self.w_kv_a, "kv_a")
            d["g"] = self._lin(o, h, self.w_gate, "gate", "sigmoid")
            if self.has_indexer:
                d["ik"] = self._lin(o, h, ix.w_k, "idx_wk", ckc=ix.ckc)
                d["iw"] = self._lin(o, h, ix.w_p, "idx_wp", ckc=ix.ckc)
        from models.demos.blackhole.hy4_preview.tt import hy4_fuse_attn as _FN

        use_qp = (
            os.environ.get("HY4_B29_QAPUSH", "0") == "1"
            and _FN.norm_enabled()
            and 1 <= S <= 8
            and qa_c.layout == ttnn.TILE_LAYOUT
            and kv.layout == ttnn.TILE_LAYOUT
            and qa_c.dtype == ttnn.bfloat16
            and kv.dtype == ttnn.bfloat16
            and not qa_c.memory_config().is_sharded()
            and not kv.memory_config().is_sharded()
            and tuple(qa_c.shape)[3] == 256
        )  # HY4_B29_QAPUSH=1 (needs HY4_NORM_FUSE=1): the q_a all-gather becomes a push program + flag wait in the fused norm readers
        if use_qp:
            qa_g = None
        else:
            qa_g = self.mesh_config.allgather(qa_c, self.ccl, axis=1, dim=3, **({"memory_config": mc} if mc else {}))
            ttnn.deallocate(qa_c)
        if use_qp:
            qa, lat, k_pe_raw = _FN.fused_norms(
                None, kv, self.q_a_norm, self.kv_a_norm, cfg.rms_norm_eps, mc, push=_FN.qa_push(self.md), qa_src=qa_c
            )
            ttnn.deallocate(qa_c)
            ttnn.deallocate(kv)
            d["qa"] = qa
            k_pe = self._rope(o, k_pe_raw, cos, sin)
        elif (
            _FN.norm_enabled()
            and 1 <= S <= 8
            and qa_g.layout == ttnn.TILE_LAYOUT
            and kv.layout == ttnn.TILE_LAYOUT
            and qa_g.dtype == ttnn.bfloat16
            and kv.dtype == ttnn.bfloat16
            and not qa_g.memory_config().is_sharded()
            and not kv.memory_config().is_sharded()
        ):  # HY4_NORM_FUSE=1 (B22): q_a rms + kv latent rms + k_pe split as ONE program (replaces 6 ops)
            qa, lat, k_pe_raw = _FN.fused_norms(qa_g, kv, self.q_a_norm, self.kv_a_norm, cfg.rms_norm_eps, mc)
            ttnn.deallocate(qa_g)
            ttnn.deallocate(kv)
            d["qa"] = qa
            k_pe = self._rope(o, k_pe_raw, cos, sin)
        else:
            if o.rmssh:
                smc, spc = self._lazy_rms()
                xs = ttnn.to_memory_config(qa_g, smc)
                y = ttnn.rms_norm(
                    xs,
                    weight=self.q_a_norm,
                    epsilon=cfg.rms_norm_eps,
                    program_config=spc,
                    memory_config=smc,
                    compute_kernel_config=self.ckc,
                )
                qa = ttnn.to_memory_config(y, ttnn.L1_MEMORY_CONFIG)
                for t in (xs, y):
                    ttnn.deallocate(t)
            else:
                qa = self._rms(qa_g, self.q_a_norm)
            ttnn.deallocate(qa_g)
            d["qa"] = qa
            lat = self._rms(ttnn.slice(kv, [0, 0, 0, 0], [1, 1, S, cfg.kv_lora_rank]), self.kv_a_norm)
            k_pe = self._rope(o, ttnn.slice(kv, [0, 0, 0, cfg.kv_lora_rank], [1, 1, S, cfg.kv_a_out]), cos, sin)
            ttnn.deallocate(kv)
        d["lat"], d["k_pe"] = lat, k_pe
        # q path
        q = self._lin(o, qa, self.w_q_b, "q_b")
        if (
            qrm
        ):  # HY4_ATTN_FUSE=1 (pool decode, S == 1): q_abs matmul + rope + concat + row-major layout in ONE program -> d["qfull"] RM [1,8,1,576]
            from models.demos.blackhole.hy4_preview.tt import hy4_fuse_attn as FA

            kvargs = None
            if (
                kvf is not None
                and FA.kv_in_q()
                and S <= 4
                and not lat.memory_config().is_sharded()
                and not k_pe.memory_config().is_sharded()
                and lat.layout == ttnn.TILE_LAYOUT
                and k_pe.layout == ttnn.TILE_LAYOUT
                and lat.dtype == ttnn.bfloat16
                and k_pe.dtype == ttnn.bfloat16
            ):  # HY4_KV_FUSE=2: KV append inside the fused_q program
                kvargs = (lat, k_pe, kvf[0], kvf[1])
                d["kv_done"] = True
            d["qfull"] = FA.fused_q(q, self.w_uk, cos, sin, self._tm, kv=kvargs, push=qpush)
            d["qrm"] = True
            d["qpushed"] = qpush is not None
            ttnn.deallocate(q)
            return d
        q_nope, q_pe = ttnn.experimental.nlp_create_q_heads_split(
            q, num_heads=self.hc, split_head_dim=cfg.qk_nope_head_dim, memory_config=mc or ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(q)
        q_abs = ttnn.matmul(q_nope, self.w_uk, compute_kernel_config=self.ckc, memory_config=mc)
        ttnn.deallocate(q_nope)
        q_pe = self._rope(o, q_pe, cos, sin)
        d["qfull"] = ttnn.concat([q_abs, q_pe], dim=3)
        ttnn.deallocate(q_abs)
        ttnn.deallocate(q_pe)
        return d

    def back(self, o, o_lat, g, gfold=False):
        """o_lat [1,8,S,512] -> value up-projection, gate, o_proj, all-reduce -> [1,1,S,6144]."""
        mc = ttnn.L1_MEMORY_CONFIG if o.l1 else None
        from models.demos.blackhole.hy4_preview.tt import hy4_fuse_attn as FA

        part = None
        if (
            FA.enabled("back")
            and int(o_lat.shape[2]) <= 4
            and o_lat.layout == ttnn.TILE_LAYOUT
            and self.w_uv_t.dtype == ttnn.bfloat16
        ):  # HY4_ATTN_FUSE=1 (S <= 4, head-major o_lat): value up-projection + head concat + gate in ONE program
            from models.demos.blackhole.hy4_preview.tt.hy4_skinny import SkinnyLinear
            from models.demos.blackhole.hy4_preview.tt.hy4_skinny import enabled as sk_enabled

            if (
                FA.back_fuse_enabled()
                and sk_enabled("oproj")
                and int(o_lat.shape[1]) == 8
                and self.w_uv_t.shape[3] == 256
                and int(self.w_uv_t.shape[2]) == 512
            ):  # HY4_BACK_FUSE=1 (B22): ... and the skinny o-proj in the same program
                if getattr(self, "_sk_o", None) is None:
                    self._sk_o = SkinnyLinear(self.md, self.host_w["o_proj"].tensor.reshape(-1, self.cfg.hidden_size))
                part = FA.fused_back_oproj(o_lat, self.w_uv_t, g, gfold, self._sk_o)
                ttnn.deallocate(g)
                og = None
            else:
                og = FA.fused_back(o_lat, self.w_uv_t, g, gfold)
                ttnn.deallocate(g)
        else:
            x = ttnn.matmul(o_lat, self.w_uv_t, compute_kernel_config=self.ckc, memory_config=mc)
            x = ttnn.experimental.nlp_concat_heads(x, memory_config=mc or ttnn.DRAM_MEMORY_CONFIG)
            if gfold:
                og = ttnn.multiply(x, g, input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)])
                ttnn.deallocate(g)
            else:
                og = ttnn.multiply(x, g)
        if part is None:
            sk = self._skinny_o(og)
            part = sk if sk is not None else self._lin(o, og, self.w_o, "o_proj")
            ttnn.deallocate(og)
        if (
            getattr(self, "far_defer", False)
            and not part.memory_config().is_sharded()
            and part.dtype == ttnn.bfloat16
            and part.layout == ttnn.TILE_LAYOUT
            and int(part.shape[2]) <= 16
        ):  # HY4_FUSE_AR=1: the layer fuses the all-reduce into the next iHC pre (hy4_fuse_ar)
            from models.demos.blackhole.hy4_preview.tt.hy4_fuse_ar import FusedPartial, applies

            if applies(int(part.shape[2])):
                return FusedPartial(part)
        return self.mesh_config.allreduce(part, self.ccl, axis=1, **({"memory_config": mc} if mc else {}))

    def _idx_append_opt(self, o, ik, tabs):
        ix = self.indexer
        cos, sin, icos, isin, c128, s128 = tabs
        k = ttnn.layer_norm(ik, weight=ix.k_norm_w, bias=ix.k_norm_b, epsilon=ix.ln_eps)
        k = self._idx_rot(o, k, icos, isin, c128, s128)
        ix.keys = k if ix.keys is None else ttnn.concat([ix.keys, k], dim=2)
        ix.length += ik.shape[2]

    def idx_q(self, o, d, tabs):
        """indexer query [1,32,S,128] (wq_b linear, head split, rope)."""
        ix = self.indexer
        cos, sin, icos, isin, c128, s128 = tabs
        q = self._lin(o, d["qa"], ix.w_q, "idx_wq_b", ckc=ix.ckc)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q,
            num_heads=ix.nh,
            num_kv_heads=0,
            transpose_k_heads=False,
            memory_config=ttnn.L1_MEMORY_CONFIG if o.l1 else ttnn.DRAM_MEMORY_CONFIG,
        )
        return self._idx_rot(o, q, icos, isin, c128, s128)

    def _idx_select_opt(self, o, d, pos, tabs):
        """Hy4Indexer.select_decode with the projections taken from ``front`` (keys already appended)."""
        from models.demos.blackhole.hy4_preview.tt.hy4_indexer import _pad_rows

        ix = self.indexer
        cos, sin, icos, isin, c128, s128 = tabs
        T = ix.length
        q = self.idx_q(o, d, tabs)
        w = d["iw"]
        Tp = -(-T // 32) * 32
        keys = _pad_rows(ix.keys, Tp)
        q32, w32 = _pad_rows(q, 32), _pad_rows(w, 32)
        pc = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=min(Tp, 256), head_group_size=1)
        score32 = ttnn.experimental.indexer_score_dsa(
            q32, keys, w32, chunk_start_idx=pos, program_config=pc, compute_kernel_config=ix.ckc_score, kv_len=Tp
        )
        score = ttnn.slice(score32, [0, 0, 0, 0], [1, 1, 1, Tp])
        ttnn.deallocate(score32)
        for t in (q32, w32):
            ttnn.deallocate(t)
        ids = ttnn.experimental.topk_large_indices(score, k=ix.topk, valid_length=T)
        for t in (q, score):
            ttnn.deallocate(t)
        return ids

    def decode_step_opt(self, h, pos, ids_in=None, opt=None, append=True, tabs=None, dense_ids=None):
        """Same contract as ``Hy4Attention.decode_step`` (S = 1 row per mesh row); returns (out [1,1,1,6144], ids)."""
        o = opt or Opt()
        cfg = self.cfg
        tabs = tabs if tabs is not None else self.rope_tabs_opt(pos, h.shape[2])
        d = self.front(h, o, tabs)
        self._tap("lat", d["lat"])
        self._tap("k_pe", d["k_pe"])
        self._tap("qfull", d["qfull"])
        if append:
            self._append_cache(d["lat"], d["k_pe"])
        kvc = self.cache.rows
        if self.has_indexer:
            if append:
                self._idx_append_opt(o, d["ik"], tabs)
            if pos + 1 <= cfg.index_topk:
                ids = dense_ids if dense_ids is not None else self.dense_ids(pos)
            elif (
                getattr(self, "ids_override", None) is not None
            ):  # gate experiments: the oracle's top-k ids replace the device selection (indexer projections still run)
                ids = self.ids_override
            else:
                ids = self._idx_select_opt(o, d, pos, tabs)
        else:
            assert ids_in is not None
            ids = ids_in
        if o.l1:  # the stock sparse_sdpa wants q / indices in DRAM
            ids = ttnn.to_memory_config(ids, ttnn.DRAM_MEMORY_CONFIG)
            d["qfull"] = ttnn.to_memory_config(d["qfull"], ttnn.DRAM_MEMORY_CONFIG)
        o_lat = self._sparse_decode(d["qfull"], kvc, ids, False)
        self._tap("o_lat", o_lat)
        out = self.back(o, o_lat, d["g"], d.get("gfold", False))
        return out, ids
