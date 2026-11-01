# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated cost of the MoE chain collectives on the 4x8 mesh (all chips ready: no wait for the slowest moe_compute), under trace.

The MoE tail output is [1, 1, T, 5120] bf16 TILE per chip (a partial sum per mesh column); the chain reduce-scatters it over the 8 columns (stock ``ttnn.reduce_scatter``
with the YAML config, ~66 us in the layer table, median over chips and INCLUDING the wait for the slowest chip) and layer.py all-gathers the [T, 640] result (lean AG).
Variants: stock RS, ``reduce_scatter_minimal_async`` (workers 1/2/default, links 1/2, Ring/Linear), TILE vs ROW_MAJOR, the RS+AG pair, ``all_reduce``-style variants.
Per call time = (trace of 5 calls - trace of 1 call) / 4.  Env: MC_TS (default 2,4,12), MC_OUT, MC_ONLY (comma list of variant names)."""
import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import DSV41MoEBlock

H, ROWS, COLS = 5120, 4, 8
TS = [int(x) for x in os.environ.get("MC_TS", "2,4,12").split(",")]
OUT = os.environ.get("MC_OUT", "/tmp")
ONLY = [x for x in os.environ.get("MC_ONLY", "").split(",") if x]


def sems(md, n):
    g = md.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    return [ttnn.create_global_semaphore(md, cores, 0) for _ in range(n)]


def per_call(md, fn, reps=10):
    def cap(k):
        fn()
        ttnn.synchronize_device(md)
        t = ttnn.begin_trace_capture(md, cq_id=0)
        for _ in range(k):
            o = fn()
            del o
        ttnn.end_trace_capture(md, t, cq_id=0)
        ttnn.synchronize_device(md)
        return t

    def run(t):
        for _ in range(2):
            ttnn.execute_trace(md, t, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(md, t, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        return (time.perf_counter() - t0) / reps * 1e6

    t1, t5 = cap(1), cap(5)
    a, b = run(t1), run(t5)
    ttnn.release_trace(md, t1)
    ttnn.release_trace(md, t5)
    return (b - a) / 4


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(3000)
@torch.no_grad()
def test_moe_chain_ccl_bench(mesh_device):
    md = mesh_device
    res = []
    S = {"rs": sems(md, 12), "ag": sems(md, 8), "bar": sems(md, 4)}
    cnt = {"rs": 0, "ag": 0, "bar": 0}

    def nxt(k, n=1):
        i = cnt[k]
        cnt[k] += n
        v = [S[k][(i + j) % len(S[k])] for j in range(n)]
        return v[0] if n == 1 else v

    for T in TS:
        cfg, _ = DSV41MoEBlock._configs(md, T, ttnn.Topology.Linear, False)
        rs_kw = cfg.reduce_scatter.model_dump()
        print("MCLOG rs stock kwargs", {k: str(v) for k, v in rs_kw.items()}, flush=True)
        mk = lambda rows, layout: ttnn.from_torch(
            torch.randn(1, 1, rows, H).to(torch.bfloat16),
            device=md,
            dtype=ttnn.bfloat16,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        x_tile = mk(T, ttnn.TILE_LAYOUT)
        x_rm = mk(T, ttnn.ROW_MAJOR_LAYOUT)
        # correctness input: a DIFFERENT partial on every device (the real chain input), reference = fp32 sum over the 8 columns of each mesh row
        g = torch.Generator().manual_seed(T)
        xd = torch.randn(ROWS, COLS, T, H, generator=g).to(torch.bfloat16)
        x_dist = ttnn.from_torch(
            xd,
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(ROWS, COLS)),
        )
        ref_sum = xd.float().sum(1)  # [ROWS, T, H]

        def check(fn):
            outs = []
            for _ in range(3):
                o = fn(x_dist)
                ttnn.synchronize_device(md)
                y = ttnn.to_torch(
                    o, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 1), mesh_shape=(ROWS, COLS))
                ).float()
                outs.append(y.reshape(ROWS, COLS, -1, H // COLS)[:, :, :T])  # [ROWS, COLS, T, 640]
            full = outs[0].permute(0, 2, 1, 3).reshape(ROWS, T, H)
            a, b = full.flatten().double(), ref_sum.flatten().double()
            pcc = float(((a - a.mean()) @ (b - b.mean())) / ((a - a.mean()).norm() * (b - b.mean()).norm()))
            return dict(
                pcc=pcc,
                max_abs=float((full - ref_sum).abs().max()),
                rerun_equal=all(bool(torch.equal(outs[0], o)) for o in outs[1:]),
            )

        def rs_min(x, workers, links, topo):
            kw = dict(
                dim=3,
                multi_device_global_semaphore=nxt("rs", 3),
                num_links=links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=topo,
                cluster_axis=1,
                barrier_semaphore=nxt("bar"),
            )
            if workers:
                kw["num_workers_per_link"] = workers
            return ttnn.experimental.reduce_scatter_minimal_async(x, **kw)

        def ag_min(x, workers, links, topo):
            kw = dict(
                dim=3,
                cluster_axis=1,
                mesh_device=md,
                topology=topo,
                multi_device_global_semaphore=nxt("ag", 2),
                num_links=links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                barrier_semaphore=nxt("bar"),
            )
            if workers:
                kw["num_workers_per_link"] = workers
            return ttnn.experimental.all_gather_async(x, **kw)

        R, L = ttnn.Topology.Ring, ttnn.Topology.Linear
        variants = {
            "rs_stock_tile": lambda: ttnn.reduce_scatter(x_tile, **rs_kw),
            "rs_min_tile_w1_l2_ring": lambda: rs_min(x_tile, 1, 2, R),
            "rs_min_tile_w1_l1_ring": lambda: rs_min(x_tile, 1, 1, R),
            "rs_min_tile_w2_l2_ring": lambda: rs_min(x_tile, 2, 2, R),
            "rs_min_tile_wdef_l2_ring": lambda: rs_min(x_tile, 0, 2, R),
            "rs_min_tile_w1_l2_linear": lambda: rs_min(x_tile, 1, 2, L),
            "rs_min_tile_w1_l4_ring": lambda: rs_min(x_tile, 1, 4, R),
            "rs_min_rm_w1_l2_ring": lambda: rs_min(x_rm, 1, 2, R),
            "rs_min_rm_w1_l1_ring": lambda: rs_min(x_rm, 1, 1, R),
        }
        fns = {
            "rs_stock_tile": lambda x: ttnn.reduce_scatter(x, **rs_kw),
            "rs_min_tile_w1_l2_ring": lambda x: rs_min(x, 1, 2, R),
            "rs_min_tile_w1_l1_ring": lambda x: rs_min(x, 1, 1, R),
            "rs_min_tile_w2_l2_ring": lambda x: rs_min(x, 2, 2, R),
            "rs_min_tile_wdef_l2_ring": lambda x: rs_min(x, 0, 2, R),
            "rs_min_tile_w1_l2_linear": lambda x: rs_min(x, 1, 2, L),
        }
        for name, fn in variants.items():
            if ONLY and name not in ONLY:
                continue
            try:
                us = per_call(md, fn)
                rec = dict(T=T, variant=name, us=us)
                if name in fns:
                    rec.update(check(fns[name]))
            except Exception as e:  # noqa: BLE001
                rec = dict(T=T, variant=name, error=str(e)[:200].replace("\n", " "))
            res.append(rec)
            print("MCR " + json.dumps(rec), flush=True)
        # all-gather of the [T, 640] scatter result (what layer.py does), lean and stock
        part = ttnn.reduce_scatter(x_tile, **rs_kw)
        agv = {
            "ag_min_w1_l2_ring": lambda: ag_min(part, 1, 2, R),
            "ag_min_wdef_l2_ring": lambda: ag_min(part, 0, 2, R),
            "ag_min_w1_l1_ring": lambda: ag_min(part, 1, 1, R),
        }
        for name, fn in agv.items():
            if ONLY and name not in ONLY:
                continue
            try:
                rec = dict(T=T, variant=name, us=per_call(md, fn))
            except Exception as e:  # noqa: BLE001
                rec = dict(T=T, variant=name, error=str(e)[:200].replace("\n", " "))
            res.append(rec)
            print("MCR " + json.dumps(rec), flush=True)
        # the production pair: stock RS + lean AG vs lean RS + lean AG
        pairs = {
            "pair_stock_rs_lean_ag": lambda: ag_min(ttnn.reduce_scatter(x_tile, **rs_kw), 1, 2, R),
            "pair_lean_rs_lean_ag": lambda: ag_min(rs_min(x_tile, 1, 2, R), 1, 2, R),
        }
        for name, fn in pairs.items():
            if ONLY and name not in ONLY:
                continue
            try:
                rec = dict(T=T, variant=name, us=per_call(md, fn))
            except Exception as e:  # noqa: BLE001
                rec = dict(T=T, variant=name, error=str(e)[:200].replace("\n", " "))
            res.append(rec)
            print("MCR " + json.dumps(rec), flush=True)
    json.dump(res, open(os.path.join(OUT, "mc_results.json"), "w"), indent=1)
