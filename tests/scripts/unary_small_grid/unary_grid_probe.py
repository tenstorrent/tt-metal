#!/usr/bin/env python3
# Unary small-tensor grid probe (#59797 small_tensor_grid): for DRAM interleaved TILE tensors, time unary ops on
# the grid main picks ("default") and on forced grids (sub_core_grids = the first N cores row-wise, or all workers).
#   --mode timing: trace replay ns/op (device-bound, what traced models see) and eager ns/call (host enqueue + device).
#   --mode kernel: run each config a few times between signposts; run under `python -m tracy -r -p` and parse the
#                  ops report (DEVICE KERNEL DURATION, CORE COUNT) with --parse.
# Every forced grid's output is compared bit-exact with the default grid's output.
import argparse
import csv
import glob
import math
import os
import statistics
import sys
import time

COST = {
    "identity": "light",
    "relu": "light",
    "abs": "light",
    "exp": "light",
    "log": "light",
    "tanh": "light",
    "cos": "moderate",
    "rsqrt": "moderate",
    "silu": "mid",
    "sigmoid": "mid",
    "gelu": "veryheavy",
}
# Rows x 256 columns, or one row of tiles for fewer than 8 tiles.
SHAPES = {t: (32 * max(1, t // 8), 32 * min(t, 8)) for t in (1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 248, 256, 512)}


def predicted_cores(cost, pages, workers, height_sharded=False):
    """Python copy of main's small_tensor_grid() for interleaved DRAM TILE tensors."""
    ppc = 4 if cost == "light" else 2 if cost in ("moderate", "mid") else 1
    ppc_above_32 = (4 if height_sharded else 3) if cost == "light" else 1
    if pages >= ppc * workers:
        return workers
    cores_for = lambda p: max(8, 1 << (math.ceil(pages / p) - 1).bit_length())
    cores = cores_for(ppc)
    if cores > 32:
        cores = cores_for(ppc_above_32)
    return workers if cores >= workers else cores


def first_cores(ttnn, grid, n):
    """The first n cores row-wise from (0, 0), as small_tensor_grid() lays them out."""
    full_rows, rest = divmod(n, grid.x)
    ranges = []
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, full_rows - 1)))
    if rest:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rest - 1, full_rows)))
    return ttnn.CoreRangeSet(set(ranges))


def configs(args, workers):
    for dtype in args.dtypes.split(","):
        for op in args.ops.split(","):
            for tiles in [int(t) for t in args.tiles.split(",")]:
                grids = ["default"]
                for g in args.grids.split(","):
                    n = workers if g == "full" else int(g)
                    if n <= workers and (n <= max(8, tiles) or g == "full"):
                        grids.append(g)
                yield dtype, op, tiles, list(dict.fromkeys(grids))


def make_input(ttnn, torch, device, dtype, op, tiles):
    h, w = SHAPES[tiles]
    torch.manual_seed(tiles)
    x = torch.rand((1, 1, h, w), dtype=torch.float32) + 0.5  # positive, so log and rsqrt stay finite
    ttdt = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32}[dtype]
    return ttnn.from_torch(x, dtype=ttdt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def trace_ns(ttnn, device, calls, args):
    """ns per pass over calls: one trace of trace_ops passes, replayed trace_replays times without blocking."""
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    outs = [c() for _ in range(args.trace_ops) for c in calls]
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    times = []
    for _ in range(args.reps):
        t0 = time.perf_counter_ns()
        for _ in range(args.trace_replays):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        times.append((time.perf_counter_ns() - t0) / (args.trace_replays * args.trace_ops))
    ttnn.release_trace(device, tid)
    del outs
    return times


def run(args):
    import torch
    import ttnn

    device = ttnn.open_device(device_id=0, trace_region_size=args.trace_region)
    grid = device.compute_with_storage_grid_size()
    workers = grid.x * grid.y
    print(f"PROBE grid {grid.x}x{grid.y} workers={workers} mode={args.mode}", flush=True)
    if args.mode == "kernel":
        from tracy import signpost
    filler = None
    if args.filler and args.mode == "timing":
        n = args.filler
        a = ttnn.from_torch(torch.rand((1, 1, n, n)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        b = ttnn.from_torch(torch.rand((1, 1, n, n)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        filler = lambda: ttnn.matmul(a, b)
        filler()  # into the program cache: a trace cannot compile it
        print(f"PROBE filler matmul {n}x{n} trace_ns={statistics.median(trace_ns(ttnn, device, [filler], args)):.0f}")

    for dtype, op, tiles, grids in configs(args, workers):
        fn = getattr(ttnn, op)
        x = make_input(ttnn, torch, device, dtype, op, tiles)
        ref = None
        for g in grids:
            n = workers if g == "full" else None if g == "default" else int(g)
            kw = {} if n is None else {"sub_core_grids": first_cores(ttnn, grid, n)}
            call = lambda: fn(x, **kw)
            pred = predicted_cores(COST[op], tiles, workers) if n is None else n
            tag = f"{op} {dtype} tiles={tiles} grid={g} cores={pred}"
            try:
                y = call()
                out = ttnn.to_torch(y)
                if ref is None:
                    ref = out
                match = "ref" if g == "default" else ("same" if torch.equal(out, ref) else "DIFF")
                del y
                if args.mode == "kernel":
                    ttnn.synchronize_device(device)
                    signpost(header=f"CFG {tag}")
                    for _ in range(args.kernel_reps):
                        y = call()
                    ttnn.synchronize_device(device)
                    ttnn.ReadDeviceProfiler(device)
                    del y
                    print(f"KCFG {tag} match={match}", flush=True)
                    continue

                # Eager: back-to-back calls, each output freed by the next assignment.
                eager, host = [], []
                for _ in range(args.reps):
                    ttnn.synchronize_device(device)
                    t0 = time.perf_counter_ns()
                    for _ in range(args.eager_calls):
                        y = call()
                    t1 = time.perf_counter_ns()
                    ttnn.synchronize_device(device)
                    t2 = time.perf_counter_ns()
                    host.append((t1 - t0) / args.eager_calls)
                    eager.append((t2 - t0) / args.eager_calls)
                    del y

                # Trace replay of back-to-back calls: every call's dispatch is exposed.
                trace = trace_ns(ttnn, device, [call], args)
                hidden = ""
                if filler:
                    # After a long full-grid op the dispatcher writes the unary's args while that op runs, so the
                    # unary costs about its kernel time: time (filler, unary) pairs minus the filler alone.
                    pair = trace_ns(ttnn, device, [filler, call], args)
                    alone = trace_ns(ttnn, device, [filler], args)
                    hidden = f" after_filler_ns={statistics.median(pair) - statistics.median(alone):.0f}"
                print(
                    f"TIMING {tag} trace_ns={statistics.median(trace):.0f} trace_min={min(trace):.0f} "
                    f"eager_ns={statistics.median(eager):.0f} host_ns={statistics.median(host):.0f}{hidden} match={match}",
                    flush=True,
                )
            except Exception as e:  # keep going; a failed config must not hide the rest
                print(f"FAIL {tag} {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)
        ttnn.deallocate(x)
    ttnn.close_device(device)


def parse(report_glob):
    """Per CFG signpost: core count and median DEVICE KERNEL DURATION of the ops that follow it."""
    paths = sorted(glob.glob(report_glob, recursive=True), key=os.path.getmtime)
    if not paths:
        print(f"PARSE no ops report under {report_glob}")
        return
    with open(paths[-1]) as f:
        rows = list(csv.DictReader(f))
    print(f"PARSE {paths[-1]} rows={len(rows)}")
    cfg, durs, cores = None, [], set()

    def flush():
        if cfg and durs:
            print(f"KERNEL {cfg} kernel_ns={statistics.median(durs):.0f} n={len(durs)} core_count={sorted(cores)}")

    for r in rows:
        if r.get("OP TYPE") == "signpost":
            flush()
            code = r.get("OP CODE", "").strip("` ")
            cfg = code[4:] if code.startswith("CFG ") else None
            durs, cores = [], set()
        elif cfg and r.get("DEVICE KERNEL DURATION [ns]", "").strip():
            durs.append(float(r["DEVICE KERNEL DURATION [ns]"]))
            cores.add(r.get("CORE COUNT"))
    flush()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["timing", "kernel"], default="timing")
    p.add_argument("--parse", help="parse an ops report glob instead of running")
    p.add_argument("--ops", default="identity,relu,exp,log,tanh,cos,silu,gelu")
    p.add_argument("--tiles", default="1,8,16,32,64,128,192,248")
    p.add_argument("--dtypes", default="bf16,fp32")
    p.add_argument("--grids", default="8,16,32,full")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--eager-calls", type=int, default=200)
    p.add_argument("--trace-ops", type=int, default=64)
    p.add_argument("--trace-replays", type=int, default=10)
    p.add_argument("--kernel-reps", type=int, default=5)
    p.add_argument("--trace-region", type=int, default=64 << 20)
    p.add_argument("--filler", type=int, default=0, help="also time each call after an NxN bf16 matmul (0: off)")
    a = p.parse_args()
    if a.parse:
        parse(a.parse)
    else:
        run(a)
