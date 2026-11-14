# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4PoolDecoder: traced B=4 decode of the 78-layer chain on the sequence-sharded SCALED_FP8 pool (task integ). See hy4_pool_decode.py for the per-layer dataflow and INTERFACE.md for the serving contract.

Build order (INTERFACE.md): mesh open (FABRIC_1D_RING) -> Hy4Chain with HY4_POOL=1 HY4_KV_FMT=scaled_fp8 (layers, head) -> Hy4PoolDecoder (PoolCtx incl. the ONE XMerge, pools, persistent inputs)
-> prefill / synthetic fill -> hand-off -> set_inputs -> capture (eager warm step, then trace) -> run.
"""
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
    def __init__(self, ch, pages, mode="topk", bucket_tokens=None, opt=None, key_dtype=ttnn.bfloat8_b):
        self.ch, self.md = ch, ch.md
        md = self.md
        r, c = int(md.shape[0]), int(md.shape[1])
        assert r == ROWS
        self.pages, self.mode = pages, mode
        self.opt = opt if opt is not None else Opt.from_env()
        self.mapper = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(r, c))
        z = lambda *shape, dt=torch.float32: torch.zeros(*shape, dtype=dt)
        self.tok = upload(md, z(ROWS, 1, 1, dt=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, row_dim=0)
        self.cos = upload(md, z(ROWS, 1, 1, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT, row_dim=0)
        self.sin = upload(md, z(ROWS, 1, 1, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT, row_dim=0)
        self.c128 = upload(md, z(ROWS, 1, 1, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT, row_dim=0)
        self.s128 = upload(md, z(ROWS, 1, 1, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT, row_dim=0)
        self.ctx = PoolCtx(md, ch.mc, ch.ccl, pages, rows=ROWS, S=1, mode=mode, bucket_tokens=bucket_tokens)
        self.kc = [None] * len(ch.layers)
        self.mlp = []
        for ly in ch.layers:
            self.mlp.append(
                with_decode_cfg(ly.dense, pn1=2, pn2=4, bw1=32, bw2=8) if ly.spec.mlp == DENSE else with_decode_cfg(ly.shared)
            )
        self.kv_pools, self.key_pools = [], []
        for ly in ch.layers:
            kv = ttnn.empty([1, 1, pages * 64, 656], dtype=ttnn.fp8_e4m3, layout=ttnn.ROW_MAJOR_LAYOUT, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            kp = (
                ttnn.empty([pages, 1, 64, 128], dtype=key_dtype, layout=ttnn.TILE_LAYOUT, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if ly.attn.has_indexer
                else None
            )
            self.kv_pools.append(kv)
            self.key_pools.append(kp)
            ly.attn.init_pool(self.ctx, kv, kp)
        self.trace = None
        self.out = None

    # ------------------------------------------------------------------ host side
    def set_inputs(self, tokens, pos):
        pos = [int(p) for p in pos]
        rope = self.ch.layers[0].attn.rope
        cs = [rope_tables(torch.tensor([p]), rope.dim, rope.theta, rope.interleave) for p in pos]
        cos = torch.stack([x[0].reshape(1, 1, 64) for x in cs]).reshape(ROWS, 1, 1, 64)
        sin = torch.stack([x[1].reshape(1, 1, 64) for x in cs]).reshape(ROWS, 1, 1, 64)
        cp = lambda t, dt, lay, tgt: ttnn.copy_host_to_device_tensor(ttnn.from_torch(t, dtype=dt, layout=lay, mesh_mapper=self.mapper), tgt)
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
                kv4 = ttnn.reshape(kp, [1, 1, self.pages * 64, 128])
                HP.slice_copy(kb, kv4, self.ctx.colid, 8, HP.KEYTILE, npg)
                if kb is not k:
                    ttnn.deallocate(kb)

    # ------------------------------------------------------------------ device step
    def _attn(self, ly, kc, h):
        tabs = (self.cos, self.sin, self.cos, self.sin, self.c128, self.s128)
        return ly.attn.decode_step_pool(h, self.opt, tabs, self._dense8)

    def step(self):
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
