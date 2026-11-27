# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""S10 microbenchmark: M3's TP collectives on one (2,4) pipeline stage, swept over the CCL op knobs.

The two norm all-gathers and attn_rs of a sparse layer are plain ``all_gather_async`` /
``reduce_scatter_minimal_async`` calls on the TP axis (cluster_axis 1) of the stage sub-mesh; the model
passes only ``num_links`` (2 on Blackhole, utils/general_utils.get_default_num_links) and leaves
``num_workers_per_link`` / ``chunks_per_sync`` / ``num_buffers_per_channel`` to the op heuristics. This
script runs exactly those shapes, outside the model, so each knob can be timed in isolation:

  all-gather      bf16 [1, 1, W/2, 6144/4] per chip -> [1, 1, W/2, 6144]   (input_norm / post_attn_norm)
  reduce-scatter  bf16 [1, 1, W/2, 6144]   per chip -> [1, 1, W/2, 6144/4] (attn_rs, shared_rs, mlp_rs)

Each config is first run eagerly and checked against torch (all-gather must match exactly, reduce-scatter
by PCC), then ``--calls`` back-to-back calls are captured in one trace and replayed ``--replays`` times.
The reported time is replay wall time / calls: host dispatch is out of the loop, so it tracks device time
of the slowest chip to within the few-µs gap between traced ops. Confirm a winner with the zone profiler.

The galaxy is opened once (8x4, as profile_prefill.py does) and stage 0's (2,4) sub-mesh is carved out, so
both SP rows run their TP collective concurrently, like the model. Results stream to ``--out`` as JSON
lines; ``--resume`` skips configs already in that file, so a hang costs one config: reset the galaxy
(``tt-smi -glx_reset``) and re-run the same command.

  cd $TT_METAL_HOME && source python_env/bin/activate && export PYTHONPATH=$TT_METAL_HOME
  export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto
  M3_FABRIC=1d M3_CCL_TOPOLOGY=linear python models/demos/minimax_m3/tests/perf/bench_tp_ccl.py \\
      --out tp_ccl_linear.jsonl --resume

Sweep modes: ``--sweep oat`` (default) varies one knob at a time around the op defaults; ``--sweep grid``
runs the full cross product of the given lists. Lists are comma separated, ``none`` = op default.
"""

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.tt.ccl import L1_SMALL_SIZE, CCLManager
from models.demos.minimax_m3.utils.fabric_env import ccl_topology_from_env, fabric_config_from_env

HIDDEN = 6144
KNOBS = ("num_links", "num_workers_per_link", "chunks_per_sync", "num_buffers_per_channel")


def _int_list(text):
    return [None if v.strip().lower() == "none" else int(v) for v in text.split(",") if v.strip()]


def configs(args):
    """(op, W, knob dict) tuples, op defaults first so every sweep has its own baseline."""
    lists = {
        "num_links": _int_list(args.num_links),
        "num_workers_per_link": _int_list(args.workers),
        "chunks_per_sync": _int_list(args.chunks),
        "num_buffers_per_channel": _int_list(args.buffers),
    }
    default = {k: None for k in KNOBS}
    default["num_links"] = lists["num_links"][-1]
    if args.sweep == "grid":
        knob_sets = [dict(zip(KNOBS, combo)) for combo in itertools.product(*(lists[k] for k in KNOBS))]
    else:
        knob_sets = [default]
        for k in KNOBS:
            knob_sets += [{**default, k: v} for v in lists[k] if v != default[k]]
    out = []
    for op in args.ops.split(","):
        for w in (int(x) for x in args.widths.split(",")):
            for knobs in knob_sets:
                # chunks_per_sync is not exposed on all_gather_async's mesh_device overload, but the
                # cluster_axis overload used below takes it, so both ops sweep the same four knobs.
                out.append((op, w, knobs))
    return out


def config_key(op, w, topology, knobs):
    return json.dumps({"op": op, "W": w, "topology": topology, **knobs}, sort_keys=True)


def make_inputs(mesh, op, w, seed=0):
    """Torch reference + device tensor, laid out as the model holds it on a (SP, TP) stage."""
    sp, tp = tuple(mesh.shape)
    rows = w // sp
    torch.manual_seed(seed)
    if op == "ag":
        # Each chip holds a distinct [rows, HIDDEN/tp] slice; the gather rebuilds [rows, HIDDEN] per SP row.
        full = torch.randn(1, 1, w, HIDDEN).bfloat16()
        dims = (2, 3)
    else:
        # Each chip holds a distinct partial-sum [rows, HIDDEN]; tp of them are summed per SP row.
        full = torch.randn(1, 1, w, HIDDEN * tp).bfloat16()
        dims = (2, 3)
    tt = ttnn.from_torch(
        full,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=dims, mesh_shape=(sp, tp)),
    )
    return full, tt


def expected_shard(full, op, sp_row, tp_col, sp, tp):
    rows = full.shape[2] // sp
    block = full[:, :, sp_row * rows : (sp_row + 1) * rows, :]
    if op == "ag":
        return block
    parts = block.reshape(1, 1, rows, tp, HIDDEN).float().sum(dim=3)
    width = HIDDEN // tp
    return parts[..., tp_col * width : (tp_col + 1) * width]


def run_op(op, tt_in, ccl, topology, knobs):
    if op == "ag":
        return ttnn.experimental.all_gather_async(
            tt_in,
            dim=3,
            cluster_axis=1,
            multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
            barrier_semaphore=ccl.get_barrier_semaphore(),
            topology=topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **knobs,
        )
    return ttnn.experimental.reduce_scatter_minimal_async(
        tt_in,
        dim=3,
        cluster_axis=1,
        multi_device_global_semaphore=ccl.get_rs_ping_pong_semaphore(),
        barrier_semaphore=ccl.get_barrier_semaphore(),
        topology=topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        **knobs,
    )


def check(mesh, op, full, tt_out):
    sp, tp = tuple(mesh.shape)
    worst = 1.0
    for idx, shard in enumerate(ttnn.get_device_tensors(tt_out)):
        r, c = divmod(idx, tp)
        got = ttnn.to_torch(shard).float()
        ref = expected_shard(full, op, r, c, sp, tp).float()
        if op == "ag":
            if not torch.equal(got, ref):
                return False, f"chip ({r},{c}) all-gather mismatch"
        else:
            ok, pcc = comp_pcc(ref, got, 0.999)
            worst = min(worst, float(pcc) if not isinstance(pcc, str) else 0.0)
            if not ok:
                return False, f"chip ({r},{c}) reduce-scatter pcc {pcc}"
    return True, "exact" if op == "ag" else f"min pcc {worst:.6f}"


def bench_one(mesh, ccl, topology, op, w, knobs, calls, replays):
    full, tt_in = make_inputs(mesh, op, w)
    # Eager: compile + correctness.
    out = run_op(op, tt_in, ccl, topology, knobs)
    ttnn.synchronize_device(mesh)
    ok, msg = check(mesh, op, full, out)
    out.deallocate(True)
    if not ok:
        tt_in.deallocate(True)
        return {"ok": False, "check": msg}

    # Trace `calls` back-to-back invocations; outputs stay live inside the trace (no realloc churn).
    trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
    outs = [run_op(op, tt_in, ccl, topology, knobs) for _ in range(calls)]
    ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)  # warm
    times = []
    for _ in range(replays):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        times.append((time.perf_counter() - t0) / calls * 1e6)
    ttnn.release_trace(mesh, trace_id)
    for o in outs:
        o.deallocate(True)
    tt_in.deallocate(True)
    times.sort()
    return {"ok": True, "check": msg, "us_min": times[0], "us_med": times[len(times) // 2], "us_all": times}


def link_rate_pct(op, w, topology, us, num_links, tp=4, line_rate=25.0):
    """Bottleneck-link bandwidth vs Blackhole galaxy line rate (25 GB/s/link/dir, tech_reports/CCLs)."""
    total = (w // 2) * HIDDEN * 2  # gathered / pre-reduce bytes per SP row
    bottleneck = (tp - 1) / tp * total
    if topology == ttnn.Topology.Ring:
        bottleneck /= 2
    links = num_links or 2
    return 100.0 * bottleneck / (us * 1e-6) / links / 1e9 / line_rate


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ops", default="ag,rs")
    p.add_argument("--widths", default="4096,8192", help="tokens per forward (W); each chip holds W/2 rows")
    p.add_argument("--sweep", choices=("oat", "grid"), default="oat")
    p.add_argument("--num-links", default="1,2", help="last value is the baseline in oat mode")
    p.add_argument("--workers", default="none,1,2,4,8")
    p.add_argument("--chunks", default="none,1,2,4,8,16,32,64,160")
    p.add_argument("--buffers", default="none,1,2,4,8")
    p.add_argument("--calls", type=int, default=20, help="ops captured per trace")
    p.add_argument("--replays", type=int, default=5)
    p.add_argument("--stages", type=int, default=4)
    p.add_argument("--stage", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    fabric = fabric_config_from_env()
    topology = ccl_topology_from_env()
    todo = configs(args)
    done = set()
    if args.resume and args.out.exists():
        done = {json.loads(line)["key"] for line in args.out.read_text().splitlines() if line.strip()}
    todo = [c for c in todo if config_key(c[0], c[1], str(topology), c[2]) not in done]
    logger.info(f"{len(todo)} configs to run ({len(done)} already in {args.out}); fabric={fabric} topology={topology}")
    if not todo:
        return 0

    ttnn.set_fabric_config(fabric)
    galaxy = ttnn.open_mesh_device(
        ttnn.MeshShape(8, 4), l1_small_size=L1_SMALL_SIZE, trace_region_size=int(os.getenv("TRACE_REGION", 8 << 20))
    )
    try:
        mesh = galaxy.create_submeshes(ttnn.MeshShape(8 // args.stages, 4))[args.stage]
        a, b = mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, 0)), mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, 1))
        links = len(ttnn.get_forwarding_link_indices(a, b))
        logger.info(f"stage {args.stage} sub-mesh {tuple(mesh.shape)}; fabric links (0,0)->(0,1): {links}")
        ccl = CCLManager(mesh, num_links=2, topology=topology)
        with args.out.open("a") as f:
            for op, w, knobs in todo:
                key = config_key(op, w, str(topology), knobs)
                logger.info(f"run {key}")
                try:
                    res = bench_one(mesh, ccl, topology, op, w, knobs, args.calls, args.replays)
                except Exception as e:  # invalid knob combos raise TT_FATAL; record and move on
                    res = {"ok": False, "check": f"error: {str(e).splitlines()[0][:300]}"}
                if res.get("ok"):
                    res["link_rate_pct"] = link_rate_pct(op, w, topology, res["us_min"], knobs["num_links"])
                row = {"key": key, "op": op, "W": w, "topology": str(topology), "fabric": str(fabric), **knobs, **res}
                row["fabric_links"] = links
                f.write(json.dumps(row) + "\n")
                f.flush()
                logger.info(
                    f"  -> {res['check']}"
                    + (
                        f"  {res['us_min']:.1f} us (med {res['us_med']:.1f})  {res['link_rate_pct']:.0f}% of 25 GB/s"
                        if res.get("ok")
                        else ""
                    )
                )
    finally:
        ttnn.close_mesh_device(galaxy)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
