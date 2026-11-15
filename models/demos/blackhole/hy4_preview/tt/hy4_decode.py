# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced decode of the Hy4 chain (B=4: one user per mesh row, T=1 token per device), static shapes, persistent inputs.

v0 scope (stage B of the chain task): contexts <= 2048 keys (dense attention: every causal key selected, the indexer is not needed; ids = arange(pos + 1) built on the host per
step), KV cache = persistent TILE bf16 [1,1,Tcap,576] per layer written IN PLACE with ``paged_update_cache`` at a per-row position tensor, read through a row-major copy by the stock
``sparse_sdpa`` (v0 decode form: 8 local heads padded to 32). Not included: the decode indexer key cache / top-k (v2 pipeline of idx-1m), sequence-sharded pool, SCALED_FP8 rows.
Inputs that change per step (token ids, position, rope tables, ids) are persistent device tensors overwritten with ``copy_host_to_device_tensor``; the whole step is one trace.
"""
import copy

import torch

import ttnn
from models.demos.blackhole.hy4_preview.config import DENSE
from models.demos.blackhole.hy4_preview.tt import hy4_dense as D
from models.demos.blackhole.hy4_preview.tt.hy4_chain import HID, NSTREAM
from models.demos.blackhole.hy4_preview.tt.hy4_common import SENTINEL, upload  # noqa: F401
from models.demos.blackhole.hy4_preview.tt.hy4_rope import rope_tables

ROWS = 4


def with_decode_cfg(sh, g1=(12, 8), g2=(12, 8), pn1=1, pn2=4, bw1=32, bw2=4):
    """A decode (T <= 32 rows, 1D mcast program configs) view of a DSV41SharedExpertColTP built with cfg=False: shares the weights."""
    from models.demos.blackhole.deepseek_v41_flash.tt.shared_expert_v2 import TILE, _cfg1d

    n = copy.copy(sh)
    Kt, ict = sh.dim // TILE, sh.ic // TILE
    n.pc1 = _cfg1d(g1, 2 * ict, pn1, Kt, bw1)
    n.pc2 = _cfg1d(g2, sh.dim // TILE, pn2, ict, bw2)
    return n


class Hy4Decoder:
    def __init__(self, ch, tcap=1024):
        assert tcap % 32 == 0 and tcap <= 2048
        self.ch, self.md, self.tcap = ch, ch.md, tcap
        md = self.md
        r, c = int(md.shape[0]), int(md.shape[1])
        assert r == ROWS
        self.mapper = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(r, c))
        z = lambda *shape, dt=torch.float32: torch.zeros(*shape, dtype=dt)
        self.tok = upload(md, z(ROWS, 1, 1, dt=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, row_dim=0)
        self.upd = upload(md, z(ROWS, dt=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, row_dim=0)
        self.cos = upload(md, z(ROWS, 1, 1, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT, row_dim=0)
        self.sin = upload(md, z(ROWS, 1, 1, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT, row_dim=0)
        self.ids = upload(md, z(ROWS, 1, 1, 2048, dt=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, row_dim=0)
        self.kc = []
        self.mlp = []
        for ly in ch.layers:
            self.kc.append(upload(md, z(ROWS, 1, tcap, 576), ttnn.bfloat16, ttnn.TILE_LAYOUT, row_dim=0))
            if ly.spec.mlp == DENSE:
                self.mlp.append(with_decode_cfg(ly.dense, pn1=2, pn2=4, bw1=32, bw2=8))
            else:
                self.mlp.append(with_decode_cfg(ly.shared))
        self.kcfg = ttnn.create_sharded_memory_config(
            shape=(32, 576),
            core_grid=ttnn.num_cores_to_corerangeset(1, ttnn.CoreCoord(8, 8), row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.trace = None
        self.out = None

    # ------------------------------------------------------------------ host side
    def _host(self, t, dtype, layout):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=self.mapper)

    def set_inputs(self, tokens, pos):
        """tokens [4] int, pos [4] int (position of the token being fed = number of keys before it)."""
        assert max(pos) < self.tcap
        rope = self.ch.layers[0].attn.rope
        cs = [rope_tables(torch.tensor([int(p)]), rope.dim, rope.theta, rope.interleave) for p in pos]
        cos = torch.stack([x[0].reshape(1, 1, 64) for x in cs]).reshape(ROWS, 1, 1, 64)
        sin = torch.stack([x[1].reshape(1, 1, 64) for x in cs]).reshape(ROWS, 1, 1, 64)
        ids = torch.full((ROWS, 1, 1, 2048), -1, dtype=torch.int32)
        for r, p in enumerate(pos):
            ids[r, 0, 0, : int(p) + 1] = torch.arange(int(p) + 1, dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(
            self._host(torch.tensor(tokens, dtype=torch.int32).reshape(ROWS, 1, 1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            self.tok,
        )
        ttnn.copy_host_to_device_tensor(
            self._host(torch.tensor(pos, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), self.upd
        )
        ttnn.copy_host_to_device_tensor(self._host(cos, ttnn.bfloat16, ttnn.TILE_LAYOUT), self.cos)
        ttnn.copy_host_to_device_tensor(self._host(sin, ttnn.bfloat16, ttnn.TILE_LAYOUT), self.sin)
        ttnn.copy_host_to_device_tensor(self._host(ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), self.ids)

    def load_prefill_cache(self):
        """prefill hand-off (outside any trace): the concat-grown prefill caches -> the persistent decode caches (rows beyond the prompt hold padding keys: never selected)."""
        for ly, kc in zip(self.ch.layers, self.kc):
            rows = ly.attn.cache.rows  # [1,1,N,576] RM bf16
            N = rows.shape[2]
            assert N <= self.tcap, (N, self.tcap)
            p = ttnn.pad(rows, [(0, 0), (0, 0), (0, self.tcap - N), (0, 0)], 0) if N < self.tcap else rows
            t = ttnn.to_layout(p, ttnn.TILE_LAYOUT)
            ttnn.copy(t, kc)
            ttnn.deallocate(t)

    # ------------------------------------------------------------------ device step
    def _attn(self, ly, kc, h):
        a, ch = ly.attn, self.ch
        cfg = ch.cfg
        qa_c = ttnn.linear(h, a.w_q_a, compute_kernel_config=a.ckc)
        qa_g = ch.mc.allgather(qa_c, ch.ccl, axis=1, dim=3)
        ttnn.deallocate(qa_c)
        qa = a._rms(qa_g, a.q_a_norm)
        ttnn.deallocate(qa_g)
        kv = ttnn.linear(h, a.w_kv_a, compute_kernel_config=a.ckc)
        lat = a._rms(ttnn.slice(kv, [0, 0, 0, 0], [1, 1, 1, cfg.kv_lora_rank]), a.kv_a_norm)
        k_pe = a.rope.apply(
            ttnn.slice(kv, [0, 0, 0, cfg.kv_lora_rank], [1, 1, 1, cfg.kv_a_out]), self.cos, self.sin, a.ckc
        )
        ttnn.deallocate(kv)
        row = ttnn.concat([lat, k_pe], dim=3)  # [1,1,1,576] TILE
        ttnn.deallocate(lat)
        ttnn.deallocate(k_pe)
        row = ttnn.to_memory_config(row, self.kcfg)
        ttnn.experimental.paged_update_cache(kc, row, update_idxs_tensor=self.upd, page_table=None)
        ttnn.deallocate(row)
        kv_rm = ttnn.to_layout(kc, ttnn.ROW_MAJOR_LAYOUT)
        q = ttnn.linear(qa, a.w_q_b, compute_kernel_config=a.ckc)
        ttnn.deallocate(qa)
        q_nope, q_pe = ttnn.experimental.nlp_create_q_heads_split(
            q, num_heads=a.hc, split_head_dim=cfg.qk_nope_head_dim, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(q)
        q_abs = ttnn.matmul(q_nope, a.w_uk, compute_kernel_config=a.ckc)
        ttnn.deallocate(q_nope)
        q_pe = a.rope.apply(q_pe, self.cos, self.sin, a.ckc)
        qfull = ttnn.concat([q_abs, q_pe], dim=3)
        ttnn.deallocate(q_abs)
        ttnn.deallocate(q_pe)
        o_lat = a._sparse_decode(qfull, kv_rm, self.ids)
        ttnn.deallocate(qfull)
        ttnn.deallocate(kv_rm)
        return a._finish(o_lat, h)

    def _mlp(self, i, ly, h):
        ch = self.ch
        if ly.spec.mlp == DENSE:
            return D.allreduce_cols(self.mlp[i].forward(h))
        part = self.mlp[i].forward(h)
        tok = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT)
        tok = ttnn.reshape(tok, [1, 1, 1, HID])
        if (
            getattr(ly, "blk_tp", None) is not None
        ):  # HY4_MOE_TP32=1: TP32 routed experts (tt/hy4_moe_tp32_block); no deferred all-reduce on this path
            if (
                getattr(self, "rp", None) is not None
            ):  # HY4_TP32_TAIL=oneshot: partial -> Hy4RowsReduce + Hy4FuseAR + iHC pre in one program (next pre call)
                from models.demos.blackhole.hy4_preview.tt.hy4_fuse_ar import FusedPartial

                pt = ly.blk_tp.forward_partial(h, tok, fold=part)
                ttnn.deallocate(part)
                return FusedPartial(pt, rp=self.rp)
            out = ly.blk_tp.forward(h, tok, fold=part)
            ttnn.deallocate(part)
            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
            y = ch.mc.allgather(out, ch.ccl, axis=1, dim=3)
            ttnn.deallocate(out)
            return y
        defer = getattr(self, "far", None) is not None
        out = ly.blk.forward(
            h, tok, fold=part, ft_defer=defer
        )  # [1,1,1,HID/8]  (HY4_FUSE_AR: the un-reduced partial as a FusedPartial: its all-reduce runs inside the next iHC pre)
        ttnn.deallocate(part)
        if defer and getattr(out, "part", None) is not None:
            return out
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        y = ch.mc.allgather(out, ch.ccl, axis=1, dim=3)
        ttnn.deallocate(out)
        return y

    def step(self):
        """one decode step on the persistent inputs -> logits [1,1,1,V/8] bf16 per device (vocab sharded over the columns)."""
        ch = self.ch
        e = ttnn.embedding(ttnn.reshape(self.tok, [1, 1]), ch.embed, layout=ttnn.TILE_LAYOUT)  # [1,1,768]
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
            a = self._attn(ly, self.kc[i], h)
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

    def capture(self):
        """compile (eager step) then capture the trace; call after set_inputs."""
        lg = self.step()
        ttnn.synchronize_device(self.md)
        ttnn.deallocate(lg)
        self.trace = ttnn.begin_trace_capture(self.md, cq_id=0)
        self.out = self.step()
        ttnn.end_trace_capture(self.md, self.trace, cq_id=0)
        ttnn.synchronize_device(self.md)

    def run(self):
        ttnn.execute_trace(self.md, self.trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.md)
        return self.out

    def read_argmax(self, logits):
        lg = ttnn.to_torch(
            logits, mesh_composer=ttnn.ConcatMesh2dToTensor(self.md, dims=(0, 3), mesh_shape=tuple(self.md.shape))
        )
        lg = lg.reshape(ROWS, -1).float()
        return lg.argmax(-1).tolist(), lg
