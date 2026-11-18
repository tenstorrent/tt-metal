# SPDX-License-Identifier: Apache-2.0
"""Hy4 MoE PREFILL block on TP32 experts (tp32-prefill2; HY4_TP32_PF=1, default OFF; JIT kernels tt/tp32pf_kernels).

forward(h [1,1,n,D] tile bf16, replicated over the 8 columns, row r = user r) -> y [1,1,n,D/8] (column-sharded routed partial summed over all 32 chips,
rows keep their own n tokens): router on own rows, all_gather over the 4 mesh rows (tokens + routes) -> Tt = 4 n tokens on every chip ->
plan (host in v0) -> K1b/K2b (single weight copy, the decode W01/W2 v1 layout) -> partial [Tt, D] TILE -> reduce_scatter over the 8 columns (hidden)
-> reduce_scatter over the 4 rows (tokens).  The shared expert and the final column all_gather stay with the caller.
"""
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt.hy4_tp32_prefill import PlanTensors, Tp32PlanBuilder, Tp32PrefillExperts
from models.demos.blackhole.hy4_preview.tt.hy4_tp32pf_plan import build_plan

HID = 6144
TOPK = 8


TAP = None  # product-race debugging: a list -> forward appends host copies (chip (0,0)) of the gathered tokens, routes and the pre-reduce partial `out`


class Hy4MoeTp32PrefillBlock:
    def __init__(self, md, ccl, gate, tpe, n, nc=110):
        self.md, self.ccl, self.gate, self.n = md, ccl, gate, n
        self.rows, self.cols = tuple(md.shape)
        self.tt = n * self.rows
        self.ex = Tp32PrefillExperts(md, tpe.w01, tpe.w2, nc=nc)
        self.nc = nc
        self.nl = int(os.environ.get("HY4_TP32PF_LINKS", "2"))
        self.async_ccl = os.environ.get("HY4_TP32PF_CCL", "stock") == "async"
        self.repl = ttnn.ReplicateTensorToMesh(md)
        self.t_plan = 0.0
        self.last = None
        self.last_route = None
        self.keep_route = False
        self.dram_log = None  # tests: list -> (stage, allocated bytes per bank) samples
        self.rs_fp32 = os.environ.get("HY4_TP32PF_RS_FP32", "0") == "1"
        # EXACT row-independent reduction (product-race): gather the 4 row partials, add them in the FIXED order 0..3 on every chip (same bits whatever the destination row), take the own n tokens with an embedding row-select.
        self.rs_exact = os.environ.get("HY4_TP32PF_RS_EXACT", "0") == "1"
        self.sel_idx = None
        if self.rs_exact:
            from models.demos.blackhole.hy4_preview.tt.hy4_common import upload

            ids = (
                torch.arange(self.rows, dtype=torch.int32).reshape(self.rows, 1) * n
                + torch.arange(n, dtype=torch.int32).reshape(1, n)
            ).reshape(self.rows, 1, n)
            self.sel_idx = ttnn.reshape(upload(md, ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, row_dim=0), [1, n])
        self.rowbase = None
        if self.rs_exact:
            from models.demos.blackhole.hy4_preview.tt import hy4_rs_tail

            if hy4_rs_tail.enabled():
                self.rowbase = hy4_rs_tail.make_rowbase(md, n)
        self.plan_mode = os.environ.get("HY4_TP32PF_PLAN", "dev")
        self.pb = Tp32PlanBuilder(md, nc=nc)

    def _ag_rows(self, t, dim):
        if self.async_ccl:
            return ttnn.experimental.all_gather_async(
                t,
                dim=dim,
                cluster_axis=0,
                mesh_device=self.md,
                topology=ttnn.Topology.Linear,
                multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
                num_links=self.nl,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                barrier_semaphore=self.ccl.get_barrier_semaphore(),
            )
        return ttnn.all_gather(
            t,
            dim=dim,
            cluster_axis=0,
            num_links=self.nl,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _rs(self, t, dim, axis, topology):
        if self.async_ccl:
            return ttnn.experimental.reduce_scatter_minimal_async(
                t,
                dim=dim,
                multi_device_global_semaphore=self.ccl.get_rs_ping_pong_semaphore(),
                num_links=self.nl,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=topology,
                cluster_axis=axis,
                barrier_semaphore=self.ccl.get_barrier_semaphore(),
            )
        return ttnn.reduce_scatter(
            t,
            dim=dim,
            cluster_axis=axis,
            num_links=self.nl,
            topology=topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def route(self, h):
        """router on the own rows -> (ids uint16 RM [n,1,1,8], scores bf16 RM [n,1,1,8]) per chip"""
        n = self.n
        gy = int(os.environ.get("HY4_PF_RT_GY", "0")) or min(4, max(1, n // 128))
        w, idx = self.gate.forward(h, grid=(gy, 8))
        return w, idx

    def _pack_route(self, w, idx):
        """w bf16 RM [n,1,1,8], idx uint16 RM [n,1,1,8] -> int32 RM [1,1,n/32,512]: per token 8 ids then 8 words whose low 16 bits are the bf16 score bits."""
        n = self.n
        idx = ttnn.to_memory_config(idx, ttnn.DRAM_MEMORY_CONFIG)
        w = ttnn.to_memory_config(w, ttnn.DRAM_MEMORY_CONFIG)
        r = ttnn.concat(
            [ttnn.reshape(idx, [n, 1, 1, TOPK]), ttnn.bitcast(ttnn.reshape(w, [n, 1, 1, TOPK]), ttnn.uint16)], dim=3
        )
        r = ttnn.typecast(r, ttnn.int32)
        return ttnn.reshape(r, [1, 1, n // 32, 512])

    def _probe(self, name):
        if self.dram_log is not None:
            v = ttnn.get_memory_view(self.md, ttnn.BufferType.DRAM)
            self.dram_log.append((name, int(v.total_bytes_allocated_per_bank)))

    def forward(self, h, plan=None, routing=None):
        """routing: optional (w, idx) to bypass the router (tests)"""
        n, tt = self.n, self.tt
        w, idx = routing if routing is not None else self.route(h)
        pk = self._pack_route(w, idx)
        self._probe("start")
        pk_all = self._ag_rows(pk, 2)  # [1,1,Tt/32,512] int32 RM on every chip
        x_rm = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT)
        x_all = self._ag_rows(x_rm, 2)  # [1,1,Tt,D] RM
        self._probe("x_all + x_rm")
        ttnn.deallocate(x_rm)
        act, out = self.ex.alloc(tt)
        self._probe("act + out")
        if self.plan_mode == "dev":
            pt = self.pb(pk_all, tt)
            self.last_route = pk_all  # kept (not freed) only when keep_route is set by a test
            if not self.keep_route:
                self.last_route = None
        else:
            dev0 = ttnn.get_device_tensors(pk_all)[0]
            r = ttnn.to_torch(dev0).reshape(tt, 2 * TOPK).to(torch.int64)
            ids = r[:, :TOPK]
            sc = (r[:, TOPK:] & 0xFFFF).to(torch.int16).view(torch.bfloat16).float()
            pl = build_plan(ids, sc, nc=self.nc, xl1=self.ex.k1v != "c")
            pt = PlanTensors(self.md, pl, mapper=self.repl)
            self.last = (ids, sc)
        self._probe("plan tensors")
        self.ex(x_all, pt, act, out, tt)
        if TAP is not None:
            d0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
            TAP.append(
                dict(
                    x_all=d0(x_all),
                    pk_all=ttnn.to_torch(ttnn.get_device_tensors(pk_all)[0]).to(torch.int64),
                    out=d0(out),
                )
            )
        ttnn.deallocate(x_all)
        ttnn.deallocate(act)
        if not (self.plan_mode == "dev" and self.keep_route):
            ttnn.deallocate(pk_all)
        ttnn.deallocate(pk)
        pt.free()
        if self.rs_exact:
            Dc = int(out.shape[3]) // self.cols
            o32 = ttnn.typecast(out, ttnn.float32)
            ttnn.deallocate(out)
            y = self._rs(
                o32, 3, 1, ttnn.Topology.Ring
            )  # [1,1,Tt,D/8] fp32: the sum over the 8 columns (order depends on the destination COLUMN only: identical for every row)
            ttnn.deallocate(o32)
            yg = self._ag_rows(y, 2)  # [1,1,4 Tt,D/8]: block r' = the partial of mesh row r'
            ttnn.deallocate(y)
            if (
                self.rowbase is not None
            ):  # B49 (HY4_B49_RSTAIL=1): ordered sum of the 4 row blocks + bf16 + own rows as ONE program (13 ops -> 1)
                from models.demos.blackhole.hy4_preview.tt.hy4_rs_tail import rs_tail

                y3 = rs_tail(self.md, yg, self.rowbase, n, Dc)
                ttnn.deallocate(yg)
                self._probe("end")
                return y3
            acc = ttnn.slice(yg, [0, 0, 0, 0], [1, 1, tt, Dc])
            for k in range(1, self.rows):
                blk = ttnn.slice(yg, [0, 0, k * tt, 0], [1, 1, (k + 1) * tt, Dc])
                nxt = ttnn.add(acc, blk)
                ttnn.deallocate(acc)
                ttnn.deallocate(blk)
                acc = nxt
            ttnn.deallocate(yg)
            sb = ttnn.typecast(acc, ttnn.bfloat16)
            ttnn.deallocate(acc)
            sb = ttnn.to_layout(sb, ttnn.ROW_MAJOR_LAYOUT)
            sb = ttnn.reshape(sb, [1, 1, tt, Dc])
            y3 = ttnn.embedding(self.sel_idx, sb, layout=ttnn.TILE_LAYOUT)  # [1,n,D/8]: this row's own tokens
            ttnn.deallocate(sb)
            y3 = ttnn.reshape(y3, [1, 1, n, Dc])
            self._probe("end")
            return y3
        if self.rs_fp32:
            # product-race: the bf16 reduce_scatter over the 4 mesh rows (Linear) rounds the sum of the 4 partials in an order that depends on the DESTINATION ROW: the same token gets different bf16 bits on
            # different rows (layer-1 `y` differs in ~20-28% of elements by 1 ulp, then grows to a stray-token trigger). Reducing in fp32 and rounding once makes the result row-independent (to fp32 association noise).
            o32 = ttnn.typecast(out, ttnn.float32)
            ttnn.deallocate(out)
            y = self._rs(o32, 3, 1, ttnn.Topology.Ring)
            ttnn.deallocate(o32)
            self._probe("rs cols (out + y)")
            y2 = self._rs(y, 2, 0, ttnn.Topology.Linear)
            ttnn.deallocate(y)
            y3 = ttnn.typecast(y2, ttnn.bfloat16)
            ttnn.deallocate(y2)
            self._probe("end")
            return y3
        y = self._rs(out, 3, 1, ttnn.Topology.Ring)  # [1,1,Tt,D/8]
        self._probe("rs cols (out + y)")
        ttnn.deallocate(out)
        y2 = self._rs(y, 2, 0, ttnn.Topology.Linear)  # [1,1,n,D/8]
        ttnn.deallocate(y)
        self._probe("end")
        return y2
