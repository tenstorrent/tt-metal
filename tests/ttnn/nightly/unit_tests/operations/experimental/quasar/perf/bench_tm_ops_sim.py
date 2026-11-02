# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Measure tilize/untilize/reshape/slice on the Quasar simulator: one process per variant, profiler cycles per op.

Usage (parent): python bench_tm_ops_sim.py --sim-dir <dir with libttsim.so + soc_descriptor.yaml> \
    --manifest cases.json --out results.tsv [--grid 3,2] [--perf-trace] [--timeout 1800]
Each manifest case is {"op": "untilize", "shape": [1,1,32,2048], "dtype": "bfloat16", "layout": "TILE",
"memcfg": "DRAM", "variant": "mainline", "kwargs": {...}}. Shapes in one case share a process; cases do not,
because a program-cache hit would replay the first variant's program.
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

PINNED_ENV = {
    "TT_METAL_SLOW_DISPATCH_MODE": "1",
    "TT_METAL_DISABLE_SFPLOADMACRO": "1",
    "ARCH_NAME": "quasar",
    "CHIP_ARCH": "quasar",
    "TT_METAL_DEVICE_PROFILER": "1",
    "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
}


def tile_count(shape):
    h = shape[-2] if len(shape) >= 2 else 1
    w = shape[-1]
    batch = 1
    for d in shape[:-2]:
        batch *= d
    return batch * (-(-h // 32)) * (-(-w // 32))


# ---------------------------------------------------------------- child ----------------------------------------------
def run_child(case, out_path):
    import torch
    import ttnn
    from sim_profile import read_new_rows, reduce_rows

    dtype = {
        "bfloat16": ttnn.bfloat16,
        "float32": ttnn.float32,
        "int32": ttnn.int32,
        "uint16": ttnn.uint16,
        "uint32": ttnn.uint32,
    }[case.get("dtype", "bfloat16")]
    layout = {"TILE": ttnn.TILE_LAYOUT, "ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT}[case.get("layout", "TILE")]
    memcfg = {"DRAM": ttnn.DRAM_MEMORY_CONFIG, "L1": ttnn.L1_MEMORY_CONFIG}[case.get("memcfg", "DRAM")]
    op = case["op"]
    variant = case.get("variant", "mainline")
    kwargs = case.get("kwargs", {})
    ns = ttnn.experimental.quasar if variant == "quasar" else ttnn

    device = ttnn.open_device(device_id=0)
    log_path = os.path.join(
        os.environ.get("TT_METAL_PROFILER_DIR", "generated/profiler"), ".logs", "profile_log_device.csv"
    )
    offset = 0
    results = []
    try:
        for shape in case["shapes"]:
            torch_dtype = torch.bfloat16 if case.get("dtype", "bfloat16") == "bfloat16" else torch.float32
            if case.get("dtype") in ("int32", "uint16", "uint32"):
                x = torch.randint(0, 100, shape, dtype=torch.int32)
            else:
                x = torch.randn(shape).to(torch_dtype)
            t = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=memcfg)
            for run in range(2):  # run 0 = program-cache miss, run 1 = hit; device zones are what we compare
                ttnn.ReadDeviceProfiler(device)
                _, offset = read_new_rows(log_path, offset)  # discard rows from from_torch / previous op
                t0 = time.time()
                if op == "tilize":
                    y = ns.tilize(t, use_multicore=kwargs.get("use_multicore", True), memory_config=memcfg)
                    golden = x
                elif op == "untilize":
                    y = ns.untilize(t, use_multicore=kwargs.get("use_multicore", True), memory_config=memcfg)
                    golden = x
                elif op == "untilize_with_unpadding":
                    end = kwargs["output_tensor_end"]
                    y = ns.untilize_with_unpadding(
                        t, end, memory_config=memcfg, use_multicore=kwargs.get("use_multicore", True)
                    )
                    golden = x[tuple(slice(0, e + 1) for e in end)]
                elif op == "reshape":
                    new_shape = kwargs["new_shape"]
                    y = ns.reshape(t, new_shape, memory_config=memcfg)
                    golden = x.reshape(new_shape)
                elif op == "slice":
                    starts, ends = kwargs["starts"], kwargs["ends"]
                    steps = kwargs.get("steps", [1] * len(starts))
                    y = ns.slice(t, starts, ends, steps, memory_config=memcfg)
                    golden = x[tuple(slice(s, e, st) for s, e, st in zip(starts, ends, steps))]
                else:
                    raise ValueError(op)
                host_s = time.time() - t0
                ttnn.ReadDeviceProfiler(device)
                rows, offset = read_new_rows(log_path, offset)
                prof = reduce_rows(rows)
                yt = ttnn.to_torch(y)
                exact = bool(torch.equal(yt.to(golden.dtype), golden))
                maxdiff = float((yt.to(torch.float32) - golden.to(torch.float32)).abs().max()) if yt.numel() else 0.0
                ntiles = tile_count(list(golden.shape))  # output tiles: the op moves the output, not the input
                results.append(
                    {
                        "op": op,
                        "variant": variant,
                        "shape": shape,
                        "dtype": case.get("dtype", "bfloat16"),
                        "layout": case.get("layout", "TILE"),
                        "memcfg": case.get("memcfg", "DRAM"),
                        "kwargs": kwargs,
                        "run": run,
                        "span_cycles": prof.span,
                        "fw_span_cycles": prof.fw_span,
                        "n_cores": prof.n_cores,
                        "tiles": ntiles,
                        "cyc_per_tile_per_cluster": (prof.span * prof.n_cores / ntiles)
                        if prof.span and ntiles
                        else None,
                        "active_riscs": prof.active_set(),
                        "util_by_risc": prof.by_risc_type(),
                        "overflow": prof.overflow,
                        "rows": len(rows),
                        "exact": exact,
                        "maxdiff": maxdiff,
                        "host_s": host_s,
                    }
                )
                ttnn.deallocate(y)
            ttnn.deallocate(t)
    finally:
        ttnn.close_device(device)
    # The simulator may flush the CSV only at close. Every program carries its own run-host id, so group the
    # whole log by that id in order of appearance and assign the groups to the op runs that got no rows.
    pending = [r for r in results if r["rows"] == 0]
    if pending:
        all_rows, _ = read_new_rows(log_path, 0)
        groups = {}
        for r in all_rows:
            if len(r) > 7:
                groups.setdefault(r[7].strip(), []).append(r)
        ordered = list(groups.values())
        if len(ordered) == len(results):
            for res, rows in zip(results, ordered):
                if res["rows"] == 0:
                    prof = reduce_rows(rows)
                    ntiles = res["tiles"]
                    res.update(
                        {
                            "span_cycles": prof.span,
                            "fw_span_cycles": prof.fw_span,
                            "n_cores": prof.n_cores,
                            "cyc_per_tile_per_cluster": (prof.span * prof.n_cores / ntiles)
                            if prof.span and ntiles
                            else None,
                            "active_riscs": prof.active_set(),
                            "util_by_risc": prof.by_risc_type(),
                            "overflow": prof.overflow,
                            "rows": len(rows),
                            "row_source": "run_id_groups",
                        }
                    )
        else:
            for res in results:
                res["row_source"] = f"ambiguous: {len(ordered)} programs for {len(results)} op runs"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------- parent ---------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", default=None, help="TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE value, e.g. 3,2")
    ap.add_argument("--perf-trace", action="store_true", help="craq-sim only: TTSIM_PERF_TRACE per dispatch")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--child-out", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        run_child(json.load(open(args.child)), args.child_out)
        return

    cases = json.load(open(args.manifest))
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    tt_metal_home = os.environ["TT_METAL_HOME"]
    all_results = []
    for i, case in enumerate(cases):
        tag = f"{i:03d}_{case['op']}_{case.get('variant', 'mainline')}"
        case_dir = os.path.join(out_dir, tag)
        os.makedirs(case_dir, exist_ok=True)
        case_path = os.path.join(case_dir, "case.json")
        json.dump(case, open(case_path, "w"))
        env = dict(os.environ)
        env.update(PINNED_ENV)
        env["TT_METAL_SIMULATOR"] = os.path.join(args.sim_dir, "libttsim.so")
        env["TT_METAL_SIMULATOR_HOME"] = args.sim_dir
        env["TT_METAL_PROFILER_DIR"] = os.path.join(case_dir, "profiler")
        env["PYTHONPATH"] = f"{tt_metal_home}:{HERE}:" + env.get("PYTHONPATH", "")
        env["LD_LIBRARY_PATH"] = f"{tt_metal_home}/build/lib:" + env.get("LD_LIBRARY_PATH", "")
        env.pop("TT_METAL_QUASAR_NOC_API_VERSION", None)
        if args.grid:
            env["TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"] = args.grid
        if args.perf_trace:
            env.update(
                {
                    "TTSIM_PERF_TRACE": "1",
                    "TTSIM_PERF_TRACE_PER_DISPATCH": "1",
                    "TTSIM_PERF_TRACE_OUT": os.path.join(case_dir, "perf_trace"),
                }
            )
            os.makedirs(env["TTSIM_PERF_TRACE_OUT"], exist_ok=True)
        child_out = os.path.join(case_dir, "results.jsonl")
        log = open(os.path.join(case_dir, "run.log"), "w")
        with open(os.path.join(case_dir, "env.txt"), "w") as f:
            for k in sorted(env):
                if k.startswith(("TT_METAL", "TTSIM", "ARCH_NAME", "CHIP_ARCH")):
                    f.write(f"{k}={env[k]}\n")
        t0 = time.time()
        try:
            rc = subprocess.run(
                [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--sim-dir",
                    args.sim_dir,
                    "--manifest",
                    args.manifest,
                    "--out",
                    args.out,
                    "--child",
                    case_path,
                    "--child-out",
                    child_out,
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                cwd=tt_metal_home,
            ).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        log.close()
        status = "ok" if rc == 0 else f"fail({rc})"
        print(f"[{tag}] {status} {time.time() - t0:.0f}s", flush=True)
        if os.path.exists(child_out):
            for line in open(child_out):
                all_results.append(json.loads(line))
    cols = [
        "op",
        "variant",
        "shape",
        "dtype",
        "layout",
        "memcfg",
        "run",
        "span_cycles",
        "fw_span_cycles",
        "n_cores",
        "tiles",
        "cyc_per_tile_per_cluster",
        "active_riscs",
        "util_by_risc",
        "overflow",
        "exact",
        "maxdiff",
        "host_s",
    ]
    with open(args.out, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in all_results:
            f.write("\t".join(json.dumps(r.get(c)) if not isinstance(r.get(c), str) else r[c] for c in cols) + "\n")
    print(f"wrote {args.out} ({len(all_results)} rows)")


if __name__ == "__main__":
    main()
