# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""S10 follow-up go/no-go: fused all-gather + qkv matmul (all_gather_minimal_matmul_async) vs today's
separate norm all-gather + qkv linear, on M3's TP axis (cluster_axis 1, Linear).

Shapes per chip: bf16 activation [rows, 6144/4] gathered to [rows, 6144], times the bf8_b qkv weight
[6144, 2304] -> bf16 [rows, 2304]. rows = W / SP: the (2,4) stage holds W/2, the whole (8,4) galaxy W/8.

  baseline : all_gather_async (8 workers/link past 4 MB per link, as MeshConfig.allgather) + ttnn.linear
  fused    : all_gather_minimal_matmul_async with a persistent gather buffer, over a few configs

Each config runs eagerly once and is checked against the baseline output (PCC), then CALLS back-to-back
calls are captured in a trace and replayed REPLAYS times; time = replay wall / calls. Results stream to
--out as JSON lines; --resume skips configs already there.

  M3_FABRIC=1d python models/demos/minimax_m3/tests/perf/bench_agmm_qkv.py --mesh 2x4 --out agmm.jsonl --resume
"""

import argparse
import json
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.config import _allgather_workers_per_link
from models.demos.minimax_m3.tt.ccl import L1_SMALL_SIZE, CCLManager
from models.demos.minimax_m3.utils.fabric_env import fabric_config_from_env

HIDDEN, QKV_N, TP = 6144, 2304, 4


def fused_configs(grid):
    """(name, kwargs) for the fused op. x of the matmul grid must equal num_links * workers per link."""
    gx, gy = grid.x, grid.y
    out = [("default", {})]
    for x in sorted({8, 2 * (gx // 2)}):
        for m, n in ((4, 8), (8, 8), (8, 16)):
            out.append(
                (
                    f"g{x}x{gy - 1}_m{m}n{n}",
                    dict(
                        config=ttnn.MinimalMatmulConfig(
                            M_block_size=m,
                            K_block_size=8,
                            N_block_size=n,
                            subblock_h=1,
                            subblock_w=4,
                            compute_with_storage_grid_size=ttnn.CoreCoord(x, gy - 1),
                        ),
                        num_workers_per_link=x // 2,
                        num_buffers_per_channel=8,
                        force_transpose=True,
                    ),
                )
            )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", choices=("2x4", "8x4"), required=True)
    p.add_argument("--widths", default="4096,8192")
    p.add_argument("--calls", type=int, default=20)
    p.add_argument("--replays", type=int, default=10)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    done = set()
    if args.resume and args.out.exists():
        done = {json.loads(l)["key"] for l in args.out.read_text().splitlines() if l.strip()}

    ttnn.set_fabric_config(fabric_config_from_env())
    galaxy = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), l1_small_size=L1_SMALL_SIZE, trace_region_size=64 << 20)
    try:
        mesh = galaxy.create_submeshes(ttnn.MeshShape(2, 4))[0] if args.mesh == "2x4" else galaxy
        sp = tuple(mesh.shape)[0]
        ccl = CCLManager(mesh, num_links=2, topology=ttnn.Topology.Linear)
        grid = mesh.compute_with_storage_grid_size()
        logger.info(f"mesh {tuple(mesh.shape)} grid {grid.x}x{grid.y}")
        torch.manual_seed(0)
        w_torch = torch.randn(1, 1, HIDDEN, QKV_N * TP).bfloat16()
        weight = ttnn.from_torch(
            w_torch,
            device=mesh,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=tuple(mesh.shape)),
        )
        with args.out.open("a") as f:
            for W in (int(w) for w in args.widths.split(",")):
                rows = W // sp
                x_torch = torch.randn(1, 1, W, HIDDEN).bfloat16()
                x = ttnn.from_torch(
                    x_torch,
                    device=mesh,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(2, 3), mesh_shape=tuple(mesh.shape)),
                )
                workers = _allgather_workers_per_link(x, TP, 2, ttnn.Topology.Linear)
                gbuf = [
                    ttnn.allocate_tensor_on_device(
                        ttnn.Shape([1, 1, rows, HIDDEN]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
                    )
                    for _ in range(2)
                ]
                state = {"i": 0}

                def baseline():
                    g = ttnn.experimental.all_gather_async(
                        x,
                        dim=3,
                        cluster_axis=1,
                        mesh_device=mesh,
                        topology=ttnn.Topology.Linear,
                        multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
                        num_links=2,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        barrier_semaphore=ccl.get_barrier_semaphore(),
                        num_workers_per_link=workers,
                    )
                    y = ttnn.linear(g, weight, dtype=ttnn.bfloat16)
                    g.deallocate(True)
                    return y

                def fused_fn(kw):
                    def run():
                        state["i"] ^= 1
                        return ttnn.experimental.all_gather_minimal_matmul_async(
                            input_tensor=x,
                            weight_tensor=weight,
                            multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
                            num_links=2,
                            topology=ttnn.Topology.Linear,
                            cluster_axis=1,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            dtype=ttnn.bfloat16,
                            persistent_output_buffer=gbuf[state["i"]],
                            **kw,
                        )[0]

                    return run

                ref = baseline()
                ttnn.synchronize_device(mesh)
                ref_host = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(ref)]
                ref.deallocate(True)
                cases = [("baseline", baseline)] + [(n, fused_fn(kw)) for n, kw in fused_configs(grid)]
                for name, fn in cases:
                    key = json.dumps({"mesh": args.mesh, "W": W, "case": name})
                    if key in done:
                        continue
                    row = {"key": key, "mesh": args.mesh, "W": W, "rows": rows, "case": name}
                    try:
                        y = fn()
                        ttnn.synchronize_device(mesh)
                        pccs = [
                            float(comp_pcc(r, ttnn.to_torch(t).float(), 0.99)[1])
                            for r, t in zip(ref_host, ttnn.get_device_tensors(y))
                        ]
                        y.deallocate(True)
                        row["min_pcc"] = min(pccs)
                        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                        outs = [fn() for _ in range(args.calls)]
                        ttnn.end_trace_capture(mesh, tid, cq_id=0)
                        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                        ts = []
                        for _ in range(args.replays):
                            t0 = time.perf_counter()
                            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                            ttnn.synchronize_device(mesh)
                            ts.append((time.perf_counter() - t0) / args.calls * 1e6)
                        ttnn.release_trace(mesh, tid)
                        for o in outs:
                            o.deallocate(True)
                        ts.sort()
                        row.update(ok=row["min_pcc"] > 0.99, us_min=ts[0], us_med=ts[len(ts) // 2])
                    except Exception as e:  # invalid config -> TT_FATAL; record and move on
                        row.update(ok=False, error=str(e).splitlines()[0][:300])
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    logger.info(
                        f"{args.mesh} W={W} {name}: "
                        + (
                            f"{row['us_min']:.1f} us (med {row['us_med']:.1f}) pcc {row['min_pcc']:.5f}"
                            if "us_min" in row
                            else row.get("error", "?")
                        )
                    )
                x.deallocate(True)
    finally:
        if args.mesh == "2x4":
            for sub in galaxy.get_submeshes():
                ttnn.close_mesh_device(sub)
        ttnn.close_mesh_device(galaxy)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    raise SystemExit(main())
