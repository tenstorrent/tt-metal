# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4PoolDecoder: traced B=4 decode of the 78-layer chain on the sequence-sharded SCALED_FP8 pool (task integ). See hy4_pool_decode.py for the per-layer dataflow and INTERFACE.md for the serving contract.

Build order (INTERFACE.md): mesh open (FABRIC_1D_RING) -> Hy4Chain with HY4_POOL=1 HY4_KV_FMT=scaled_fp8 (layers, head) -> Hy4PoolDecoder (PoolCtx incl. the ONE XMerge, pools, persistent inputs)
-> prefill / synthetic fill -> hand-off -> set_inputs -> capture (eager warm step, then trace) -> run.
"""
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.config import DENSE
from models.demos.blackhole.hy4_preview.tt import hy4_dense as D
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
from models.demos.blackhole.hy4_preview.tt.hy4_attn_opt import Opt
from models.demos.blackhole.hy4_preview.tt.hy4_chain import HID, NSTREAM
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.blackhole.hy4_preview.tt.hy4_decode import ROWS, Hy4Decoder, with_decode_cfg
from models.demos.blackhole.hy4_preview.tt.hy4_pool_decode import PoolCtx
from models.demos.blackhole.hy4_preview.tt.hy4_rope import rope_tables


class Hy4PoolDecoder(Hy4Decoder):
    def __init__(
        self,
        ch,
        pages,
        mode="topk",
        bucket_tokens=None,
        opt=None,
        key_dtype=ttnn.bfloat8_b,
        S=1,
        tok=None,
        extra_pages=0,
    ):
        """S > 1 (task mtp-device, verify rows): S query rows per user per step; ``tok`` = an existing persistent uint32 [ROWS, S] token tensor (the MTPRoundTail's) or None. S == 1: unchanged."""
        self.ch, self.md = ch, ch.md
        self.S = S
        md = self.md
        r, c = int(md.shape[0]), int(md.shape[1])
        assert r == ROWS
        self.pages, self.mode = pages, mode
        self.alloc_pages = pages + int(
            extra_pages
        )  # B45b HY4_BLOCKS: shadow pages appended to every pool tensor (logical pool = first ``pages``)
        self.opt = opt if opt is not None else Opt.from_env()
        self.mapper = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(r, c))
        z = lambda *shape, dt=torch.float32: torch.zeros(*shape, dtype=dt)
        if S == 1:
            self.tok = upload(md, z(ROWS, 1, 1, dt=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, row_dim=0)
        else:
            self.tok = (
                tok
                if tok is not None
                else upload(md, z(ROWS, S, dt=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, row_dim=0)
            )
        self.cos = upload(md, z(ROWS, 1, S, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT, row_dim=0)
        self.sin = upload(md, z(ROWS, 1, S, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT, row_dim=0)
        self.c128 = upload(md, z(ROWS, 1, S, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT, row_dim=0)
        self.s128 = upload(md, z(ROWS, 1, S, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT, row_dim=0)
        self.ctx = PoolCtx(md, ch.mc, ch.ccl, pages, rows=ROWS, S=S, mode=mode, bucket_tokens=bucket_tokens)
        self.kc = [None] * len(ch.layers)
        self.mlp = []
        for ly in ch.layers:
            self.mlp.append(
                with_decode_cfg(ly.dense, pn1=2, pn2=4, bw1=32, bw2=8)
                if ly.spec.mlp == DENSE
                else (ly.shared_dec if getattr(ly, "shared_dec", None) is not None else with_decode_cfg(ly.shared))
            )
        self.kv_pools, self.key_pools = [], []
        for ly in ch.layers:
            kv = ttnn.empty(
                [1, 1, self.alloc_pages * 64, 656],
                dtype=ttnn.fp8_e4m3,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=md,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            kp = (
                ttnn.empty(
                    [self.alloc_pages, 1, 64, 128],
                    dtype=key_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=md,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                if ly.attn.has_indexer
                else None
            )
            self.kv_pools.append(kv)
            self.key_pools.append(kp)
            ly.attn.init_pool(self.ctx, kv, kp)
        self.trace = None
        self.out = None
        self.far = None
        self.rp = None
        from models.demos.blackhole.hy4_preview.tt import hy4_fuse_ar as FAR

        if (
            FAR.enabled()
        ):  # HY4_FUSE_AR=1 (needs HY4_IHC_FUSED=1): one-shot all-reduce of the attention o-path and MoE partials inside the next iHC pre; built last (after the CCLManager / XMerge / pools)
            assert os.environ.get("HY4_IHC_FUSED", "0") == "1", "HY4_FUSE_AR needs HY4_IHC_FUSED=1"
            FAR.patch_dsv4_tail()
            self.far = FAR.shared(md)
            self.rp = (
                FAR.shared_rp(md, int(os.environ.get("HY4_MOE_TP32_T", "1")))
                if os.environ.get("HY4_TP32_TAIL", "") == "oneshot"
                else None
            )
            for ly in ch.layers:
                ly.attn.far_defer = True

    # ------------------------------------------------------------------ host side
    def set_inputs(self, tokens, pos):
        pos = [int(p) for p in pos]
        rope = self.ch.layers[0].attn.rope
        cs = [rope_tables(torch.tensor([p]), rope.dim, rope.theta, rope.interleave) for p in pos]
        cos = torch.stack([x[0].reshape(1, 1, 64) for x in cs]).reshape(ROWS, 1, 1, 64)
        sin = torch.stack([x[1].reshape(1, 1, 64) for x in cs]).reshape(ROWS, 1, 1, 64)
        cp = lambda t, dt, lay, tgt: ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(t, dtype=dt, layout=lay, mesh_mapper=self.mapper), tgt
        )
        cp(torch.tensor(tokens, dtype=torch.int32).reshape(ROWS, 1, 1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.tok)
        cp(cos, ttnn.bfloat16, ttnn.TILE_LAYOUT, self.cos)
        cp(sin, ttnn.bfloat16, ttnn.TILE_LAYOUT, self.sin)
        first = self.ch.layers[0].attn.indexer.idx_rope_first
        ones, zeros = torch.ones(ROWS, 1, 1, 64), torch.zeros(ROWS, 1, 1, 64)
        cp(torch.cat([cos, ones] if first else [ones, cos], dim=3), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.c128)
        cp(torch.cat([sin, zeros] if first else [zeros, sin], dim=3), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.s128)
        self.ctx.set_step(pos)

    def handoff(self, nrows):
        """prefill hand-off (outside traces): the chain's replicated per-layer caches [1,1,N,656] (scaled_fp8) + indexer keys [1,1,N,128] bf16 -> this column's pool slices. nrows = N (multiple of 512)."""
        assert nrows % HP.CHUNK == 0 and nrows // HP.CHUNK <= self.pages
        npg = nrows // HP.CHUNK
        for ly, kv, kp in zip(self.ch.layers, self.kv_pools, self.key_pools):
            rows = ly.attn.cache.rows
            assert int(rows.shape[2]) >= nrows and rows.dtype == ttnn.fp8_e4m3, (rows.shape, rows.dtype)
            HP.slice_copy(rows, kv, self.ctx.colid, 64, HP.ROWB, npg)
            if kp is not None:
                k = ly.attn.indexer.keys
                kb = ttnn.typecast(k, ttnn.bfloat8_b) if k.dtype != ttnn.bfloat8_b else k
                kv4 = ttnn.reshape(kp, [1, 1, self.alloc_pages * 64, 128])
                HP.slice_copy(kb, kv4, self.ctx.colid, 8, HP.KEYTILE, npg)
                if kb is not k:
                    ttnn.deallocate(kb)

    # ------------------------------------------------------------------ device step
    def _attn(self, ly, kc, h):
        tabs = (self.cos, self.sin, self.cos, self.sin, self.c128, self.s128)
        return ly.attn.decode_step_pool(h, self.opt, tabs, self._dense8)

    def step(self):
        if os.environ.get("HY4_IHC_FUSED", "0") == "1":
            return self.step_fused()
        ch = self.ch
        self._dense8 = self.ctx.dense_ids8() if self.mode == "dense" else None
        e = ttnn.embedding(ttnn.reshape(self.tok, [1, 1]), ch.embed, layout=ttnn.TILE_LAYOUT)
        e = ttnn.reshape(e, [1, 1, 1, HID // 8])
        e = ch.mc.allgather(e, ch.ccl, axis=1, dim=3)
        e32 = ttnn.typecast(e, ttnn.float32)
        ttnn.deallocate(e)
        x4 = ttnn.concat([e32] * NSTREAM, dim=1)
        ttnn.deallocate(e32)
        for i, ly in enumerate(ch.layers):
            pre, post = ly.ihc_a.pre_post(x4)
            raw = ly.ihc_a.collapse(x4, pre)
            ttnn.deallocate(pre)
            h = ly._norm(raw, ly.w_in)
            ttnn.deallocate(raw)
            a = self._attn(ly, None, h)
            ttnn.deallocate(h)
            x4 = ly.ihc_a.expand(x4, a, post)
            ttnn.deallocate(a)
            pre, post = ly.ihc_m.pre_post(x4)
            raw = ly.ihc_m.collapse(x4, pre)
            ttnn.deallocate(pre)
            h = ly._norm(raw, ly.w_post)
            ttnn.deallocate(raw)
            y = self._mlp(i, ly, h)
            ttnn.deallocate(h)
            x4 = ly.ihc_m.expand(x4, y, post)
            ttnn.deallocate(y)
        hn, logits = ch.head(x4)
        ttnn.deallocate(hn)
        ttnn.deallocate(x4)
        return logits

    def release(self):
        if self.trace is not None:
            ttnn.release_trace(self.md, self.trace)
            self.trace = None
            self.out = None

    def fill_synthetic(self, tokens, gen, chunk=65536, same_all_layers=True):
        """FAST SYNTHETIC FILL (long-context perf / capacity tests; NOT a real context): random SCALED_FP8 rows and indexer keys for the first ``tokens`` tokens of every user, written by slice_copy.
        One random chunk is generated on the host per call and reused for every layer (same_all_layers), so a 1M fill takes seconds.
        """
        dp = ttnn.experimental.deepseek_prefill
        md = self.md
        rep = lambda t, dt, lay: ttnn.from_torch(
            t,
            device=md,
            dtype=dt,
            layout=lay,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for c0 in range(0, tokens, chunk):
            n = min(chunk, tokens - c0)
            assert n % HP.CHUNK == 0
            lat = rep(torch.randn(1, 1, n, 512, generator=gen).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            rope = rep(torch.randn(1, 1, n, 64, generator=gen).bfloat16(), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
            f8, sc = dp.per_token_cast_to_fp8(lat, round_scale_to_power_of_two=True)
            nat = dp.pack_scaled_fp8_kv_cache(f8, sc, rope)
            kd = rep(torch.randn(1, 1, n, 128, generator=gen).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            kb = ttnn.typecast(kd, ttnn.bfloat8_b)
            for kv, kp in zip(self.kv_pools, self.key_pools):
                HP.slice_copy(nat, kv, self.ctx.colid, 64, HP.ROWB, n // 512, dst_page0=c0 // 512)
                if kp is not None:
                    HP.slice_copy(
                        kb,
                        ttnn.reshape(kp, [1, 1, self.alloc_pages * 64, 128]),
                        self.ctx.colid,
                        8,
                        HP.KEYTILE,
                        n // 512,
                        dst_page0=c0 // 512,
                    )
            ttnn.synchronize_device(md)
            for t in (lat, rope, f8, sc, nat, kd, kb):
                ttnn.deallocate(t)

    def step_fused(self):
        """HY4_IHC_FUSED=1: iHC as the fused JIT kernels on packed streams [1,1,4,D] fp32 (one call per sub-block: expand of the previous sub-block + mixes + collapse + norm; the last expand is one extra call)."""
        ch = self.ch
        self._dense8 = self.ctx.dense_ids8() if self.mode == "dense" else None
        e = ttnn.embedding(ttnn.reshape(self.tok, [1, 1]), ch.embed, layout=ttnn.TILE_LAYOUT)
        e = ttnn.reshape(e, [1, 1, 1, HID // 8])
        e = ch.mc.allgather(e, ch.ccl, axis=1, dim=3)
        e32 = ttnn.typecast(e, ttnn.float32)
        ttnn.deallocate(e)
        x = ttnn.concat([e32] * NSTREAM, dim=2)  # packed [1,1,4,D]
        ttnn.deallocate(e32)
        y = pp_m = None
        from models.demos.blackhole.hy4_preview.tt import hy4_fuse_ar as FAR

        pre = (
            (lambda ihc, x, w, y, pp, xo: FAR.pre_with(ihc, self.far, x, w, y, pp, xo))
            if self.far is not None
            else (lambda ihc, x, w, y, pp, xo: ihc.pre(x, w, y=y, pp_prev=pp, x_out=xo))
        )
        for i, ly in enumerate(ch.layers):
            wa, wm = ly.ihcf_a.norm_weight(ly.hw_in, 1), ly.ihcf_m.norm_weight(ly.hw_post, 1)
            if y is None:
                h, _, pp_a = ly.ihcf_a.pre(x, wa)
            else:
                h, _, pp_a, x = pre(ly.ihcf_a, x, wa, y, pp_m, x)
                if getattr(y, "part", None) is None:
                    ttnn.deallocate(y)
            a = self._attn(ly, None, h)
            ttnn.deallocate(h)
            h2, _, pp_m, x = pre(ly.ihcf_m, x, wm, a, pp_a, x)
            if getattr(a, "part", None) is None:
                ttnn.deallocate(a)
            y = self._mlp(i, ly, h2)
            ttnn.deallocate(h2)
        last = ch.layers[-1]
        _, _, _, x = pre(last.ihcf_m, x, last.ihcf_m.norm_weight(last.hw_post, 1), y, pp_m, x)
        if getattr(y, "part", None) is None:
            ttnn.deallocate(y)
        x4 = ttnn.to_layout(
            ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), [1, NSTREAM, 1, HID]), ttnn.TILE_LAYOUT
        )
        hn, logits = ch.head(x4)
        ttnn.deallocate(hn)
        ttnn.deallocate(x4)
        return logits

    # ------------------------------------------------------------------ verify rows (task mtp-device): S = k + 1 query rows per user, fused iHC path only
    def attach_derive(self, sd):
        """use the rope tables / write rows of a SpecDerive (device-derived from the resident position) instead of host set_inputs."""
        self.cos, self.sin, self.c128, self.s128 = sd.cos, sd.sin, sd.c128, sd.s128

    def _dump(self, name, i, t):
        """diagnostic (verify-round track): self.dump = list -> host copies [32 devices, ...] of intermediate tensors (eager runs only)."""
        d = getattr(self, "dump", None)
        if (
            d is not None and getattr(t, "part", None) is None
        ):  # a FusedPartial (all-reduce deferred into the next pre) has no host copy
            ttnn.synchronize_device(self.md)
            d.append((name, i, ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.md, dim=0)).float()))

    def step_verify(self):
        """one verify step on S rows per user (tokens self.tok [1,S] per device; positions P..P+S-1 via ctx / tabs filled by SpecDerive.derive) -> (hn bf16 [1,1,S,D] post model.norm rows, logits [1,1,S,V/8])."""
        assert (
            os.environ.get("HY4_IHC_FUSED", "0") == "1"
        ), "verify rows need the fused-iHC path (packed streams, T <= 16)"
        ch, S = self.ch, self.S
        self._dense8 = self.ctx.dense_ids8() if self.mode == "dense" else None
        e = ttnn.embedding(self.tok, ch.embed, layout=ttnn.TILE_LAYOUT)  # [1,S,D/8]
        e = ttnn.reshape(e, [1, 1, S, HID // 8])
        e = ch.mc.allgather(e, ch.ccl, axis=1, dim=3)
        e32 = ttnn.typecast(e, ttnn.float32)
        ttnn.deallocate(e)
        e_rm = ttnn.reshape(ttnn.to_layout(e32, ttnn.ROW_MAJOR_LAYOUT), [1, S, 1, HID])
        ttnn.deallocate(e32)
        x = ttnn.to_layout(
            ttnn.reshape(ttnn.concat([e_rm] * NSTREAM, dim=2), [1, 1, NSTREAM * S, HID]), ttnn.TILE_LAYOUT
        )  # packed: row 4t + i = stream i of token t
        ttnn.deallocate(e_rm)
        y = pp_m = None
        from models.demos.blackhole.hy4_preview.tt import hy4_fuse_ar as FAR

        # HY4_FUSE_AR=1 (+ HY4_FAR_T16=1 for S*... rows > 8): the attention o-proj partial's all-reduce runs inside the next iHC pre (FusedPartial), as in the plain step_fused (verify-opt track)
        pre = (
            (lambda ihc, x_, w, y_, pp, xo: FAR.pre_with(ihc, self.far, x_, w, y_, pp, xo))
            if self.far is not None
            else (lambda ihc, x_, w, y_, pp, xo: ihc.pre(x_, w, y=y_, pp_prev=pp, x_out=xo))
        )
        for i, ly in enumerate(ch.layers):
            wa, wm = ly.ihcf_a.norm_weight(ly.hw_in, S), ly.ihcf_m.norm_weight(ly.hw_post, S)
            if y is None:
                h, _, pp_a = ly.ihcf_a.pre(x, wa)
            else:
                h, _, pp_a, x = pre(ly.ihcf_a, x, wa, y, pp_m, x)
                if getattr(y, "part", None) is None:
                    ttnn.deallocate(y)
            a = self._attn(ly, None, h)
            if getattr(a, "part", None) is None:
                self._dump("h_in", i, h)
                self._dump("attn", i, a)
            ttnn.deallocate(h)
            h2, _, pp_m, x = pre(ly.ihcf_m, x, wm, a, pp_a, x)
            if getattr(a, "part", None) is None:
                ttnn.deallocate(a)
            self._dump("h2", i, h2)
            y = self._mlp_rows(i, ly, h2)
            self._dump("mlp", i, y)
            ttnn.deallocate(h2)
        last = ch.layers[-1]
        _, _, _, x = pre(last.ihcf_m, x, last.ihcf_m.norm_weight(last.hw_post, S), y, pp_m, x)
        if getattr(y, "part", None) is None:
            ttnn.deallocate(y)
        if (
            os.environ.get("HY4_HEAD_FAST", "0") == "1"
        ):  # verify-opt: head on the packed streams (no permute churn, split-K reductions)
            hn, logits = ch.head_packed(x)
            ttnn.deallocate(x)
            return hn, logits
        x4 = ttnn.to_layout(
            ttnn.permute(ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), [1, S, NSTREAM, HID]), (0, 2, 1, 3)),
            ttnn.TILE_LAYOUT,
        )  # [1,4,S,D]
        ttnn.deallocate(x)
        hn, logits = ch.head(x4)
        ttnn.deallocate(x4)
        return hn, logits

    def _mlp_rows(self, i, ly, h):
        """MLP of S rows per user: dense layer 0 as in the decoder, MoE blocks built (``for_batch``) at batch_per_device = S."""
        ch, S = self.ch, self.S
        if ly.spec.mlp == DENSE:
            return D.allreduce_cols(self.mlp[i].forward(h))
        part = None if getattr(self._blks[i], "shared_fuse", None) is not None else self.mlp[i].forward(h)
        if (
            os.environ.get("HY4_VERIFY_MOE_SPLIT", "0") == "1"
        ):  # diagnostic: S separate T=1-shaped MoE calls (the plain decode shape) instead of one T=S call
            outs = []
            for j in range(S):
                hj = ttnn.slice(h, [0, 0, j, 0], [1, 1, j + 1, HID])
                pj = ttnn.slice(part, [0, 0, j, 0], [1, 1, j + 1, HID])
                tj = ttnn.reshape(ttnn.to_layout(hj, ttnn.ROW_MAJOR_LAYOUT), [1, 1, 1, HID])
                oj = ttnn.to_memory_config(ly.blk.forward(hj, tj, fold=pj), ttnn.DRAM_MEMORY_CONFIG)
                outs.append(ttnn.to_layout(oj, ttnn.ROW_MAJOR_LAYOUT))
            out = ttnn.to_layout(ttnn.concat(outs, dim=2), ttnn.TILE_LAYOUT)
            ttnn.deallocate(part)
            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
            y = ch.mc.allgather(out, ch.ccl, axis=1, dim=3)
            ttnn.deallocate(out)
            return y
        tok = ttnn.reshape(ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT), [S, 1, 1, HID])
        blk = self._blks[i]
        if getattr(self, "rp_v", None) is not None and hasattr(blk, "forward_partial"):
            # HY4_TP32_TAIL=oneshot at the verify shape (verify-opt track): rows-phase reduce (packed T = S rows, nrp 24) + column all-reduce + the next iHC pre in ONE mesh program
            from models.demos.blackhole.hy4_preview.tt.hy4_decode import bfuse
            from models.demos.blackhole.hy4_preview.tt.hy4_fuse_ar import FusedPartial

            if bfuse("fold"):
                pt = blk.forward_partial(h, tok, fold=part, defer_fold=True)
                return FusedPartial(pt, rp=self.rp_v, fold=part)
            pt = blk.forward_partial(h, tok, fold=part)
            if part is not None:
                ttnn.deallocate(part)
            return FusedPartial(pt, rp=self.rp_v)
        out = blk.forward(h, tok, fold=part)
        if part is not None:
            ttnn.deallocate(part)
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        y = ch.mc.allgather(out, ch.ccl, axis=1, dim=3)
        ttnn.deallocate(out)
        return y

    def prepare_blocks(self):
        """per-layer MoE blocks at batch_per_device = S (weights shared with the S = 1 blocks: ``for_batch``); call BEFORE any trace capture."""
        self._blks, self.blk_buffers = [], None
        self.rp_v = None
        if (
            getattr(self, "far", None) is not None
            and os.environ.get("HY4_TP32_TAIL", "") == "oneshot"
            and os.environ.get("HY4_MOE_TP32", "0") == "1"
        ):
            from models.demos.blackhole.hy4_preview.tt import hy4_fuse_ar as FAR

            self.rp_v = FAR.shared_rp(self.md, self.S)  # built outside any trace, after the CCLManager / Hy4FuseAR
        for ly in self.ch.layers:
            if ly.spec.mlp == DENSE:
                self._blks.append(None)
                continue
            nb = ly.blk.for_batch(
                self.S, buffers=self.blk_buffers
            )  # ONE dispatch / combine scratch set at the new size shared by all layers (``for_batch`` allocates a new set when ``buffers`` is None)
            if self.blk_buffers is None:
                self.blk_buffers = nb.decode.buffers
            if getattr(ly, "blk_tp", None) is not None:
                # HY4_MOE_TP32=1 (+ HY4_TP32_CACHE_DIR): TP32 (tp24 kernel) routed experts for the S verify rows too: same expert copy as the S = 1 block, a block at T = S tokens per mesh row
                # (tp24: T * 4 <= 16 tokens on the mesh), the router of the batch-S block. Untested at S > 1 on the model path (product-integ): default OFF with the flag.
                from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp32_block import Hy4MoeTp32Block

                assert self.S * int(self.md.shape[0]) <= 16, "TP32 verify block: S * 4 <= 16 tokens"
                _b = Hy4MoeTp32Block(self.md, self.ch.ccl, nb.gate, ly.blk_tp.tpe, self.S)
                _b.shared_fuse = ly.blk_tp.shared_fuse
                self._blks.append(_b)
                continue
            self._blks.append(nb)
