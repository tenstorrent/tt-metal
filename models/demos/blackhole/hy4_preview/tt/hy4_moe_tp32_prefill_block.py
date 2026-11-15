# SPDX-License-Identifier: Apache-2.0
"""Hy4 MoE PREFILL block on TP32 experts (tp32-prefill2; HY4_TP32_PF=1, default OFF; JIT kernels tt/tp32pf_kernels).

forward(h [1,1,n,D] tile bf16, replicated over the 8 columns, row r = user r) -> y [1,1,n,D/8] (column-sharded routed partial summed over all 32 chips,
rows keep their own n tokens): router on own rows, all_gather over the 4 mesh rows (tokens + routes) -> Tt = 4 n tokens on every chip ->
plan (host in v0) -> K1b/K2b (single weight copy, the decode W01/W2 v1 layout) -> partial [Tt, D] TILE -> reduce_scatter over the 8 columns (hidden)
-> reduce_scatter over the 4 rows (tokens).  The shared expert and the final column all_gather stay with the caller.
"""
import os

import ttnn
from models.demos.blackhole.hy4_preview.tt.hy4_tp32_prefill import PlanTensors, Tp32PrefillExperts
from models.demos.blackhole.hy4_preview.tt.hy4_tp32pf_plan import build_plan

HID = 6144
TOPK = 8


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
        n = self.n
        f32 = lambda a: ttnn.typecast(ttnn.to_layout(ttnn.reshape(a, [1, 1, n, TOPK]), ttnn.TILE_LAYOUT), ttnn.float32)
        pk_t = ttnn.concat([f32(idx), f32(w)], dim=3)  # [1,1,n,16] fp32 (ids exact, scores bf16-exact)
        return ttnn.to_layout(pk_t, ttnn.ROW_MAJOR_LAYOUT)

    def forward(self, h, plan=None, routing=None):
        """routing: optional (w, idx) to bypass the router (tests)"""
        n, tt = self.n, self.tt
        w, idx = routing if routing is not None else self.route(h)
        pk = self._pack_route(w, idx)
        pk_all = self._ag_rows(pk, 2)  # [1,1,Tt,16] fp32 RM on every chip
        x_rm = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT)
        x_all = self._ag_rows(x_rm, 2)  # [1,1,Tt,D] RM
        ttnn.deallocate(x_rm)
        # v0: plan on the host from chip (0, 0)
        dev0 = ttnn.get_device_tensors(pk_all)[0]
        r = ttnn.to_torch(dev0).reshape(tt, 2 * TOPK).float()
        ids = r[:, :TOPK].round().long()
        sc = r[:, TOPK:]
        pl = build_plan(ids, sc, nc=self.nc, xl1=self.ex.k1v != "c")
        pt = PlanTensors(self.md, pl, mapper=self.repl)
        act, out = self.ex.alloc(tt)
        self.ex(x_all, pt, act, out, tt)
        ttnn.deallocate(x_all)
        ttnn.deallocate(act)
        ttnn.deallocate(pk_all)
        ttnn.deallocate(pk)
        for t_ in (pt.plan, pt.gidx, pt.dest, pt.sc):
            ttnn.deallocate(t_)
        y = self._rs(out, 3, 1, ttnn.Topology.Ring)  # [1,1,Tt,D/8]
        ttnn.deallocate(out)
        y2 = self._rs(y, 2, 0, ttnn.Topology.Linear)  # [1,1,n,D/8]
        ttnn.deallocate(y)
        self.last = (ids, sc)
        return y2
