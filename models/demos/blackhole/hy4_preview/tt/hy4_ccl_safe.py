# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4 CCL safety self-test (slot-diag D34-1, program/slot-diag/DIAG.md + BUG_REPORT.md).

Known defect: on mesh rows that contain a 4-ethernet-link hop (UMD pairs (7,31) and (15,23) of a Galaxy; .44 slots 1,2 and 4x8 rows 1,2 before the 04:20 reset), the
FIRST-USE-PER-PARITY of ``all_gather_async`` (dim=3, TILE, cluster_axis=1, num_links=2) returns stale chunks on every odd call (ping-pong semaphore set 1), silently.
``num_links=1`` is exact. The state is per boot/host: a board reset cleared it on .44. This module detects it at model build with exact integer data.

INTEGRATION CONTRACT (one line, at mesh open, before building CCLManager and before the pool context):
    links = hy4_fabric_selftest(mesh_device);  ccl = CCLManager(mesh_device, num_links=links.num_links, ...);  pool all_gather_async num_links = links.pool_ag
(HY4_POOL_AG_LINKS env is read by hy4_pool_decode as the default; DSV41_CCL_LINKS_CALL overrides lean CCL calls.)  HY4_CCL_LINKS_FORCE=1|2 skips the test.
"""
import logging
import os
import time
from dataclasses import dataclass, field

import torch

import ttnn

log = logging.getLogger("hy4_ccl_safe")
A2A_LINKS = 2  # B51: num_links of the all_to_all_async_generic calls (prefill a2a2, pool decode); the serve engine lowers it to 1 when the pf-size a2a selftest case fails


@dataclass
class Hy4Links:
    per_op: dict = field(
        default_factory=dict
    )  # op class -> safe num_links (2 = both parities exact for N calls, else 1)
    seconds: float = 0.0
    forced: bool = False

    @property
    def num_links(
        self,
    ) -> (
        int
    ):  # for CCLManager / lean calls (all-reduce + TILE gather share it): min over the classes that go through the manager
        return min(
            self.per_op.get(k, 2) for k in ("ag_tile_dim3", "allreduce_rs_ag", "ag_tile_dim3_pf", "allreduce_rs_ag_pf")
        )

    @property
    def a2a(
        self,
    ) -> (
        int
    ):  # B51: all_to_all_async_generic links (prefill heads<->sequence a2a2, pool decode a2a), 2 unless the pf-size a2a case failed
        return self.per_op.get("a2a_pf", 2)

    @property
    def pool_ag(self) -> int:  # the pool path's all_gather_async calls (q gather dim=2 row-major, key gather)
        return min(self.per_op.get(k, 2) for k in ("ag_rm_dim2", "ag_tile_dim3"))


def _cluster_shape(mesh):
    r, c = mesh.shape[0], mesh.shape[1]
    return int(r), int(c)


def _put(mesh, x, R, C, dim, layout):
    if R == 1:
        mapper = ttnn.ShardTensorToMesh(mesh, dim=dim)
    else:
        mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(R, C), dims=(0, dim))
    return ttnn.from_torch(
        x, device=mesh, layout=layout, dtype=ttnn.bfloat16, mesh_mapper=mapper, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _exact(outs, x, R, C):
    for i, y in enumerate(ttnn.get_device_tensors(outs)):
        if not torch.equal(ttnn.to_torch(y), x[i // C : i // C + 1] if R > 1 else x):
            return False
    return True


def _check(mesh, op, nl, n_calls, topology):
    from models.demos.gpt_oss.tt.ccl import (  # own manager: fresh semaphores, both ping-pong sets exercised from call 0
        CCLManager,
    )

    R, C = _cluster_shape(mesh)
    ccl = CCLManager(mesh, num_links=nl, topology=topology)
    ri = lambda *s: torch.randint(-8, 8, s).bfloat16()
    pf = op.endswith("_pf")  # B51: prefill-size case (HY4_B51_PF_ROWS rows, default 512) instead of 32 rows
    op = op[:-3] if pf else op
    NR = int(os.environ.get("HY4_B51_PF_ROWS", "512")) if pf else 32

    def ag(t, dim):
        return ttnn.experimental.all_gather_async(
            t,
            dim=dim,
            cluster_axis=1,
            mesh_device=mesh,
            topology=topology,
            multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
            num_links=nl,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=ccl.get_barrier_semaphore(),
        )

    ok = True
    for _ in range(n_calls):  # consecutive calls: odd and even parity both
        if op == "ag_tile_dim3":
            x = ri(R, 1, NR, C * 256)
            o = ag(_put(mesh, x, R, C, 3, ttnn.TILE_LAYOUT), 3)
        elif op == "a2a":  # heads<->sequence round trip (prefill a2a2 / pool decode), exact
            x = ri(R * C, 8, NR, 576)
            t = ttnn.from_torch(
                x,
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            kw = dict(num_links=nl, memory_config=ttnn.DRAM_MEMORY_CONFIG, cluster_axis=1)
            q = ttnn.experimental.all_to_all_async_generic(t, in_dim=1, out_dim=2, **kw)
            o = ttnn.experimental.all_to_all_async_generic(q, in_dim=2, out_dim=1, **kw)
            ttnn.synchronize_device(mesh)
            for i, y in enumerate(ttnn.get_device_tensors(o)):
                ok &= torch.equal(ttnn.to_torch(y).reshape(8, NR, 576), x[i])
            continue
        elif op == "ag_rm_dim2":
            x = ri(R, 8, C * 4, 576)
            o = ag(_put(mesh, x, R, C, 2, ttnn.ROW_MAJOR_LAYOUT), 2)
        else:  # allreduce_rs_ag: lean all-reduce form (reduce_scatter_minimal + TILE all_gather)
            xs = [ri(R, 1, NR, 6144) for _ in range(C)]
            if R == 1:
                t = _put(mesh, torch.cat(xs, 0).reshape(C, 1, NR, 6144), 1, C, 0, ttnn.TILE_LAYOUT)
                ref = sum(xs).float()
            else:  # per-row independent partials: device (r,c) holds xs[c][r]
                full = torch.stack(
                    [torch.stack([xs[c][r] for c in range(C)], 0) for r in range(R)], 0
                )  # [R,C,1,32,6144]
                t = ttnn.from_torch(
                    full.reshape(R * C, 1, NR, 6144),
                    device=mesh,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ref = None
            rs = ttnn.experimental.reduce_scatter_minimal_async(
                t,
                dim=3,
                multi_device_global_semaphore=ccl.get_rs_ping_pong_semaphore(),
                num_links=nl,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=topology,
                cluster_axis=1,
                barrier_semaphore=ccl.get_barrier_semaphore(),
            )
            o = ag(rs, 3)
            ttnn.synchronize_device(mesh)
            for i, y in enumerate(ttnn.get_device_tensors(o)):
                want = sum(xs).float() if R == 1 else sum(xs[c][i // C] for c in range(C)).float()
                ok &= torch.equal(ttnn.to_torch(y).float().reshape(want.shape), want)
            continue
        ttnn.synchronize_device(mesh)
        ok &= _exact(o, x, R, C)
    if os.environ.get("HY4_SELFTEST_FAKE_BAD", "") == f"{op}:{nl}":  # test hook: pretend this (op, links) fails
        ok = False
    return ok


def hy4_fabric_selftest(mesh, n_calls=4, topology=None, ops=None) -> Hy4Links:
    """Run the exact-integer odd/even all_gather_async test per op class. Safe links per class: 2 if num_links=2 is exact for n_calls consecutive calls, else 1 if num_links=1 is
    exact, else RuntimeError (slot unusable). Mesh must be open with the fabric the model uses. Typically 2-4 s warm."""
    if ops is None:  # B51 (HY4_B51_SELFTEST_PF=1, default off): also the prefill-size cases (512 rows) and the a2a
        ops = ("ag_tile_dim3", "ag_rm_dim2", "allreduce_rs_ag")
        if os.environ.get("HY4_B51_SELFTEST_PF", "0") == "1":
            ops += ("ag_tile_dim3_pf", "allreduce_rs_ag_pf", "a2a_pf")
    forced = os.environ.get("HY4_CCL_LINKS_FORCE")
    if forced:
        res = Hy4Links({k: int(forced) for k in ops}, 0.0, True)
        log.warning("hy4_fabric_selftest SKIPPED (HY4_CCL_LINKS_FORCE=%s)", forced)
        return res
    topology = topology or ttnn.Topology.Ring
    t0 = time.perf_counter()
    res = Hy4Links()
    for op in ops:
        if _check(mesh, op, 2, n_calls, topology):
            res.per_op[op] = 2
        elif _check(mesh, op, 1, n_calls, topology):
            res.per_op[op] = 1
            log.error(
                "hy4_fabric_selftest: %s WRONG with num_links=2 (stale chunks on odd calls), exact with num_links=1 -> using 1",
                op,
            )
        else:
            raise RuntimeError(
                f"hy4_fabric_selftest: {op} wrong with num_links 2 AND 1 on this mesh: chips/fabric unusable (see program/slot-diag/DIAG.md)"
            )
    res.seconds = time.perf_counter() - t0
    log.warning(
        "hy4_fabric_selftest verdict %s -> CCLManager num_links=%d pool_ag=%d (%.1fs)",
        res.per_op,
        res.num_links,
        res.pool_ag,
        res.seconds,
    )
    return res


def make_safe_ccl(mesh, topology=None, **kw):
    """NAN49 / B51 helper for tests and harnesses: CCLManager whose num_links comes from the fabric self-test (the production engine does this in hy4_serve_engine; a bare
    ``CCLManager(md, num_links=2, topology=Ring)`` on a board in the parity-defect state returns WRONG reduce-scatter / all-gather results on every odd call, e.g. NaN in
    Hy4Attention.prefill_chunk on odd chunks). HY4_B51_SAFE_CCL=0 -> plain num_links=2 (old harness behaviour)."""
    from models.demos.gpt_oss.tt.ccl import CCLManager

    topology = topology or ttnn.Topology.Ring
    nl = (
        2
        if os.environ.get("HY4_B51_SAFE_CCL", "1") == "0"
        else hy4_fabric_selftest(mesh, topology=topology, **kw).num_links
    )
    return CCLManager(mesh, num_links=nl, topology=topology)
