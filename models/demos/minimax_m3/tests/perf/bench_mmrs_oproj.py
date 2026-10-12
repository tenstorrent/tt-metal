# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""X7: tune the fused attention o_proj matmul + TP reduce-scatter
(minimal_matmul_strided_reduce_scatter_async) for M3's prefill shapes, on Ring.

Shapes per chip: bf8_b activation [rows, 64/4 * 128 = 2048] times the row-parallel bf8_b o_proj weight
[2048, 6144] -> partial [rows, 6144] -> reduce-scatter on the TP axis -> [rows, 1536]. rows = chunk / SP;
production chunk 5120 on a (2,4) stage -> 2560 rows = 80 M-tiles.

  baseline : ttnn.matmul (bf16 out, as apply_output_projection) + reduce_scatter_minimal_async (Ring)
  legacy32 : the S=1024 _FUSED_MM_RS_CONFIGS entry (8-wide MM grid, DRAM hand-off), for reference
  fused    : the cartesian product of the --grids/--mb/--kb/--nb/--cw/--handoff/--workers/--nbuf lists

Every case is checked against a torch fp32 golden and against the baseline (PCC), run eagerly twice and
compared bit-for-bit (the op was once reported to race on Blackhole), then CALLS back-to-back calls are
captured in a trace and replayed REPLAYS times; time = replay wall / calls. The last traced output is
compared bit-for-bit with the eager one as well. Results stream to --out as JSON lines; --resume skips
keys already there.

  M3_FABRIC=1d_ring TT_MESH_GRAPH_DESC_PATH=.../single_bh_galaxy_torus_xy_graph_descriptor.textproto \\
    python models/demos/minimax_m3/tests/perf/bench_mmrs_oproj.py --m-tiles 80 --out /tmp/mmrs.jsonl --resume
"""

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.tt.attention.operations import apply_output_projection_fused_rs
from models.demos.minimax_m3.tt.ccl import L1_SMALL_SIZE, CCLManager
from models.demos.minimax_m3.utils.fabric_env import fabric_config_from_env

TILE = 32
K_LOCAL, HIDDEN, TP, NUM_LINKS = 2048, 6144, 4, 2


def pick_subblock(m_block, n_block):
    """Largest sub_h * sub_w <= 4 (fp32 dest acc) dividing the blocks; 2x2 when both are even."""
    if m_block % 2 == 0 and n_block % 2 == 0:
        return 2, 2
    best = (1, 1)
    for h in range(1, min(m_block, 4) + 1):
        for w in range(1, min(n_block, 4) + 1):
            if m_block % h == 0 and n_block % w == 0 and h * w <= 4 and h * w > best[0] * best[1]:
                best = (h, w)
    return best


def ints(s):
    return [int(v) for v in s.split(",") if v]


def fused_cases(args, m_tiles, device_grid):
    """(name, params) for every fused config; params feed fused_kwargs."""
    out = [
        (
            "legacy32",
            dict(gx=8, gy=4, mb=4, kb=4, nb=6, sbh=2, sbw=2, cw=2, workers=3, handoff="dram", nbuf=None),
        )
    ]
    grids = [tuple(int(v) for v in g.split("x")) for g in args.grids.split(",")]
    workers = [None if w == "auto" else int(w) for w in args.workers.split(",")]
    nbufs = [None if b == "auto" else int(b) for b in args.nbuf.split(",")]
    for (gx, gy), mb, kb, nb, cw, handoff, w, nbuf in itertools.product(
        grids, ints(args.mb), ints(args.kb), ints(args.nb), ints(args.cw), args.handoff.split(","), workers, nbufs
    ):
        if gy >= device_grid.y or gx > device_grid.x or (K_LOCAL // TILE) % kb:
            continue
        mt_per_core = math.ceil(m_tiles / gy)
        if mb > mt_per_core:
            continue
        if handoff.startswith("l1") and math.ceil(mt_per_core / mb) < 2:
            continue  # a window over one block per core is degenerate (see tt_dit FusedMMRSConfig)
        if w is None:
            w = ((device_grid.y - gy) * device_grid.x) // (2 * NUM_LINKS) - 1
        if w < 1:
            continue
        sbh, sbw = pick_subblock(mb, nb)
        name = f"g{gx}x{gy}_m{mb}k{kb}n{nb}_cw{cw}_{handoff}_w{w}" + (f"_b{nbuf}" if nbuf else "")
        out.append(
            (
                name,
                dict(gx=gx, gy=gy, mb=mb, kb=kb, nb=nb, sbh=sbh, sbw=sbw, cw=cw, workers=w, handoff=handoff, nbuf=nbuf),
            )
        )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", choices=("2x4", "8x4"), default="2x4", help="one (2,4) stage, or the whole galaxy")
    p.add_argument("--m-tiles", default="80", help="comma list of per-chip M tiles (80 = chunk 5120 on SP=2)")
    p.add_argument("--grids", default="12x8,12x7", help="MM grids GXxGY; RS uses the rows below GY")
    p.add_argument("--mb", default="2,4,5,6")
    p.add_argument("--kb", default="4,8")
    p.add_argument("--nb", default="6,8")
    p.add_argument("--cw", default="1,2")
    p.add_argument("--handoff", default="l1w2,dram", help="l1wN = windowed L1 MM->RS hand-off with N blocks; dram")
    p.add_argument("--workers", default="auto", help="RS workers per link; auto = RS zone // (2*links) - 1")
    p.add_argument("--nbuf", default="auto", help="num_buffers_per_channel; auto = op default")
    p.add_argument("--out-dtype", choices=("bf8_b", "bf16"), default="bf8_b", help="fused RS output dtype")
    p.add_argument("--only", default="", help="comma list of case names to run (after generation)")
    p.add_argument("--wrapper", action="store_true", help="also time apply_output_projection_fused_rs itself")
    p.add_argument("--no-baseline", action="store_true", help="skip timing the unfused baseline")
    p.add_argument("--calls", type=int, default=10)
    p.add_argument("--replays", type=int, default=10)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    done = set()
    if args.resume and args.out.exists():
        done = {json.loads(l)["key"] for l in args.out.read_text().splitlines() if l.strip()}

    fabric = fabric_config_from_env()
    assert fabric == ttnn.FabricConfig.FABRIC_1D_RING, "the fused MM+RS op is Ring-only: set M3_FABRIC=1d_ring"
    ttnn.set_fabric_config(fabric)
    galaxy = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), l1_small_size=L1_SMALL_SIZE, trace_region_size=64 << 20)
    try:
        mesh = galaxy.create_submeshes(ttnn.MeshShape(2, 4))[0] if args.mesh == "2x4" else galaxy
        sp = tuple(mesh.shape)[0]
        ccl = CCLManager(mesh, num_links=NUM_LINKS, topology=ttnn.Topology.Ring)
        grid = mesh.compute_with_storage_grid_size()
        logger.info(f"mesh {tuple(mesh.shape)} grid {grid.x}x{grid.y}")
        progress, credit = ccl.get_mm_progress_counters_buffer(), ccl.get_mm_credit_counters_buffer()
        compute_config = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        out_dtype = ttnn.bfloat16 if args.out_dtype == "bf16" else None

        torch.manual_seed(0)
        # Row-parallel o_proj: global [1, 1, K_LOCAL * TP, HIDDEN], K sharded over TP (mesh axis 1).
        w_torch = (torch.randn(1, 1, K_LOCAL * TP, HIDDEN) / math.sqrt(K_LOCAL * TP)).bfloat16()
        weight = ttnn.from_torch(
            w_torch,
            device=mesh,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 2), mesh_shape=tuple(mesh.shape)),
        )
        # Golden uses the bf8_b-rounded weight, so it measures the activation/math error only.
        w_ref = torch.cat([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(weight)[:TP]], dim=-2)

        with args.out.open("a") as f:
            for m_tiles in ints(args.m_tiles):
                rows = m_tiles * TILE
                x_torch = torch.randn(1, 1, rows * sp, K_LOCAL * TP).bfloat16()
                x = ttnn.from_torch(
                    x_torch,
                    device=mesh,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(2, 3), mesh_shape=tuple(mesh.shape)),
                )
                # Device (r, c) holds rows of SP block r and output columns of TP slice c (row-major order).
                x_ref = ttnn.to_torch(
                    x, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(2, 3), mesh_shape=tuple(mesh.shape))
                ).float()
                full = x_ref @ w_ref  # [1, 1, rows*sp, HIDDEN]
                n_loc = HIDDEN // TP
                golden = [
                    full[:, :, r * rows : (r + 1) * rows, c * n_loc : (c + 1) * n_loc]
                    for r in range(sp)
                    for c in range(TP)
                ]

                def baseline():
                    y = ttnn.matmul(x, weight, dtype=ttnn.bfloat16)
                    rs = ttnn.experimental.reduce_scatter_minimal_async(
                        y,
                        dim=3,
                        multi_device_global_semaphore=ccl.get_rs_ping_pong_semaphore(),
                        num_links=NUM_LINKS,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        topology=ttnn.Topology.Ring,
                        cluster_axis=1,
                        barrier_semaphore=ccl.get_barrier_semaphore(),
                    )
                    y.deallocate(True)
                    return rs

                def fused_fn(c):
                    l1 = c["handoff"].startswith("l1")
                    kw = dict(
                        compute_kernel_config=compute_config,
                        num_links=NUM_LINKS,
                        memory_config_mm=ttnn.L1_MEMORY_CONFIG if l1 else ttnn.DRAM_MEMORY_CONFIG,
                        rs_intermediate_mem_config=ttnn.DRAM_MEMORY_CONFIG,
                        rs_output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
                        topology=ttnn.Topology.Ring,
                        cluster_axis=1,
                        config=ttnn.MinimalMatmulConfig(
                            M_block_size=c["mb"],
                            K_block_size=c["kb"],
                            N_block_size=c["nb"],
                            subblock_h=c["sbh"],
                            subblock_w=c["sbw"],
                            compute_with_storage_grid_size=ttnn.CoreCoord(c["gx"], c["gy"]),
                        ),
                        barrier_semaphore=ccl.get_barrier_semaphore(),
                        chunk_width_in_mm_blocks=c["cw"],
                        num_workers_per_link=c["workers"],
                        num_buffers_per_channel=c["nbuf"],
                        mm_progress_counters=progress,
                    )
                    if l1:
                        kw.update(mm_window_blocks=int(c["handoff"][3:]), mm_credit_counters=credit)
                    if out_dtype is not None:
                        kw["dtype"] = out_dtype

                    def run():
                        mm_out, rs_out = ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
                            x,
                            weight,
                            3,
                            ccl.get_rs_ping_pong_semaphore(),
                            ttnn.CoreCoord(0, c["gy"]),
                            **kw,
                        )[:2]
                        mm_out.deallocate(True)
                        return rs_out

                    return run

                def host(t):
                    return [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(t)]

                ref = baseline()
                ttnn.synchronize_device(mesh)
                ref_host = host(ref)
                ref.deallocate(True)
                base_golden_pcc = min(float(comp_pcc(g, r, 0.99)[1]) for g, r in zip(golden, ref_host))
                logger.info(f"M_tiles={m_tiles}: baseline vs golden pcc {base_golden_pcc:.5f}")

                def wrapper():
                    # The production entry point, including its per-call host work (bf16 out, as the model).
                    xin = ttnn.typecast(x, ttnn.bfloat16)  # the wrapper casts to bf8_b itself and frees its input
                    return apply_output_projection_fused_rs(
                        xin, SimpleNamespace(o_proj=weight), SimpleNamespace(tp_axis=1), ccl, ttnn.bfloat16
                    )

                cases = [] if args.no_baseline else [("baseline", None)]
                if args.wrapper:
                    cases.append(("wrapper", "wrapper"))
                cases += fused_cases(args, m_tiles, grid)
                only = set(filter(None, args.only.split(",")))
                if only:
                    cases = [c for c in cases if c[0] in only]
                logger.info(f"M_tiles={m_tiles}: {len(cases)} cases")
                for name, c in cases:
                    kd = {"m_tiles": m_tiles, "case": name, "out_dtype": args.out_dtype}
                    if args.mesh != "2x4":
                        kd["mesh"] = args.mesh
                    key = json.dumps(kd)
                    if key in done:
                        continue
                    row = {
                        "key": key,
                        "mesh": args.mesh,
                        "m_tiles": m_tiles,
                        "rows": rows,
                        "case": name,
                        "out_dtype": args.out_dtype,
                    }
                    if isinstance(c, dict):
                        row["cfg"] = c
                    fn = baseline if c is None else wrapper if c == "wrapper" else fused_fn(c)
                    try:
                        y1 = fn()
                        ttnn.synchronize_device(mesh)
                        h1 = host(y1)
                        y1.deallocate(True)
                        y2 = fn()
                        ttnn.synchronize_device(mesh)
                        h2 = host(y2)
                        y2.deallocate(True)
                        row["pcc_golden"] = min(float(comp_pcc(g, h, 0.99)[1]) for g, h in zip(golden, h1))
                        row["pcc_baseline"] = min(float(comp_pcc(r, h, 0.99)[1]) for r, h in zip(ref_host, h1))
                        row["eager_equal"] = all(torch.equal(a, b) for a, b in zip(h1, h2))
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
                        row["trace_equal"] = all(torch.equal(a, b) for a, b in zip(h1, host(outs[-1])))
                        ttnn.release_trace(mesh, tid)
                        for o in outs:
                            o.deallocate(True)
                        # Eager: host enqueue time and wall time per call, as the model runs it (no trace).
                        ttnn.synchronize_device(mesh)
                        t0 = time.perf_counter()
                        outs = [fn() for _ in range(args.calls)]
                        t1 = time.perf_counter()
                        ttnn.synchronize_device(mesh)
                        t2 = time.perf_counter()
                        for o in outs:
                            o.deallocate(True)
                        row.update(
                            eager_host_us=(t1 - t0) / args.calls * 1e6, eager_wall_us=(t2 - t0) / args.calls * 1e6
                        )
                        ts.sort()
                        row.update(
                            ok=row["pcc_golden"] > 0.99 and row["eager_equal"] and row["trace_equal"],
                            us_min=ts[0],
                            us_med=ts[len(ts) // 2],
                        )
                    except Exception as e:  # invalid config -> TT_FATAL; record and move on
                        row.update(ok=False, error=str(e).splitlines()[0][:300])
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    logger.info(
                        f"M_tiles={m_tiles} {name}: "
                        + (
                            f"{row['us_min']:.1f} us (med {row['us_med']:.1f}) eager host {row['eager_host_us']:.1f} "
                            f"wall {row['eager_wall_us']:.1f} pcc g {row['pcc_golden']:.5f} "
                            f"b {row['pcc_baseline']:.5f} det {row['eager_equal']}/{row['trace_equal']}"
                            if "us_min" in row
                            else row.get("error", "?")
                        )
                    )
                x.deallocate(True)
    finally:
        for sub in galaxy.get_submeshes():
            ttnn.close_mesh_device(sub)
        ttnn.close_mesh_device(galaxy)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    raise SystemExit(main())
