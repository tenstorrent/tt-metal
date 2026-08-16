# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Real-silicon baseline for ttnn.topk / ttnn.sort.

WHY THIS EXISTS
---------------
A threshold-based selection kernel was measured at 2.0 cycles per 32-element
vector on Blackhole (tt_metal/tt-llk/tests/sources/sfpu_count_above_perf.cpp,
correctness-verified by test_sfpu_count_above.py). That number is meaningless
without knowing what the shipping ops actually cost at the same shapes. This
script produces that baseline.

MEASUREMENT DISCIPLINE
----------------------
- Device Kernel Duration from Tracy, never time.perf_counter. Host dispatch is
  hundreds of microseconds and would swamp every number here.
- Program cache warmed before the measured iterations, so JIT compile time is
  excluded.
- Unsupported configurations are RECORDED WITH THEIR ERROR, not silently
  dropped and not replaced with a model. Half the point of this script is to
  discover empirically which (shape, k) pairs the shipping ops even accept --
  an earlier effort asserted numbers for configurations that cannot run.

USAGE
-----
  # 1. measure (must be under tracy for DEVICE KERNEL DURATION to exist)
  python -m tracy -r -v tests/ttnn/unit_tests/operations/reduction/_topk_sort_bench.py

  # 2. report against the newest Tracy CSV
  python tests/ttnn/unit_tests/operations/reduction/_topk_sort_bench.py --report

  # subset / custom sweep
  python -m tracy -r -v ... _topk_sort_bench.py -- --preset moe
  python -m tracy -r -v ... _topk_sort_bench.py -- --shapes 1x32768 --ks 32,64 --ops topk

Underscore-prefixed so routine `pytest tests/...` does not collect it and run
the module-level device code (see docs/profiling.md).
"""

import argparse
import csv
import glob
import json
import os
import sys
import traceback

REPO = os.environ.get("TT_METAL_HOME", "/home/nachiket/tt-metal")

# Blackhole nominal Tensix clock. Used ONLY to convert ns -> cycles so the
# result is comparable with the LLK-level cycles/vector figure. AICLK is
# DVFS-managed, so treat derived cycle counts as approximate; the ns column is
# the primary measurement.
CLOCK_GHZ = 1.35

# ---------------------------------------------------------------------------
# Configuration battery.
#
# Shapes are [batch, N] and get expanded to the 4D [1, 1, batch, N] that
# ttnn.topk requires. The MoE / vocab / vector-search groupings mirror the
# workloads a selection kernel would actually serve.
#
# NOTE: several of these are expected to FAIL validation -- multi-core topk
# requires k <= 64 (topk_device_operation.cpp:75) and W >= 8192
# (topk_constants.hpp:11), and the large-k vector-search rows exceed that.
# They are kept deliberately: the failure and its exact message is the useful
# output, and it is the datum an earlier analysis fabricated instead of
# obtaining.
# ---------------------------------------------------------------------------
PRESETS = {
    "moe": [
        ("DeepSeek-V3", 1, 256, 8),
        ("DeepSeek-V3", 32, 256, 8),
        ("DeepSeek-V3", 128, 256, 8),
        ("Mixtral-8x7B", 1, 32, 2),
        ("Mixtral-8x7B", 32, 32, 2),
        ("DeepSeek-V2", 32, 64, 6),
    ],
    "vocab": [
        ("LLaMA-3-32k", 1, 32768, 32),
        ("LLaMA-3-32k", 8, 32768, 32),
        ("Qwen-64k", 1, 65536, 32),
        ("Qwen-64k", 8, 65536, 64),
        ("DeepSeek-128k", 1, 131072, 32),
    ],
    "search": [
        ("Vector-Search-4k", 1, 4096, 128),
        ("Vector-Search-16k", 1, 16384, 256),
        ("Vector-Search-64k", 1, 65536, 512),
    ],
    "sweep": [("sweep", 1, n, k) for n in (4096, 8192, 32768, 131072) for k in (8, 32, 64)],
}


def build_configs(args):
    if args.shapes or args.ks:
        shapes = []
        for s in (args.shapes or "1x32768").split(","):
            b, n = s.lower().split("x")
            shapes.append((int(b), int(n)))
        ks = [int(k) for k in (args.ks or "32").split(",")]
        return [("custom", b, n, k) for (b, n) in shapes for k in ks]
    out = []
    for p in args.preset.split(","):
        out.extend(PRESETS[p])
    return out


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def run_measurements(args):
    import torch

    import ttnn

    device = ttnn.open_device(device_id=0, l1_small_size=args.l1_small)
    arch = ttnn.get_arch_name()
    print(f"BENCH: arch={arch} iters={args.iters} warmup={args.warmup}", flush=True)

    manifest = []
    torch.manual_seed(0)

    for preset, batch, n, k in build_configs(args):
        shape = (1, 1, batch, n)
        for op in args.ops.split(","):
            tag = f"{op}|{preset}|{batch}x{n}|k{k}"
            entry = {
                "op": op,
                "preset": preset,
                "batch": batch,
                "n": n,
                "k": k,
                "tag": tag,
                "status": "",
                "error": "",
            }
            try:
                t = torch.randn(shape, dtype=torch.bfloat16)
                x = ttnn.from_torch(
                    t,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

                def call():
                    if op == "topk":
                        return ttnn.topk(x, k=k, dim=-1, largest=True, sorted=True)
                    elif op == "sort":
                        return ttnn.sort(x, dim=-1, descending=True)
                    raise ValueError(f"unknown op {op}")

                # Warm the program cache so JIT compile is not in the measured
                # iterations. Correctness is spot-checked on the warmup result
                # for topk: a baseline that silently returns garbage is not a
                # baseline.
                out = call()
                if op == "topk":
                    vals = ttnn.to_torch(out[0])[..., :k].float()
                    ref = torch.topk(t.float(), k=k, dim=-1).values
                    max_err = (vals - ref).abs().max().item()
                    entry["warmup_max_abs_err"] = max_err
                ttnn.synchronize_device(device)

                for _ in range(args.warmup):
                    call()
                ttnn.synchronize_device(device)

                # Measured region. Tracy attributes DEVICE KERNEL DURATION per
                # op invocation; the report step aggregates by op name.
                for _ in range(args.iters):
                    call()
                ttnn.synchronize_device(device)

                entry["status"] = "RAN"
                print(f"BENCH_OK   {tag}", flush=True)

            except Exception as e:  # noqa: BLE001 - the message IS the result
                entry["status"] = "UNSUPPORTED"
                entry["error"] = f"{type(e).__name__}: {e}".split("\n")[0][:400]
                print(f"BENCH_FAIL {tag} :: {entry['error']}", flush=True)
                if args.verbose:
                    traceback.print_exc()

            manifest.append(entry)

    ttnn.close_device(device)

    with open(args.manifest, "w") as f:
        json.dump({"arch": arch, "iters": args.iters, "entries": manifest}, f, indent=1)
    print(f"BENCH: manifest -> {args.manifest}", flush=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def newest_csv():
    pat = os.path.join(REPO, "generated/profiler/reports/*/ops_perf_results_*.csv")
    files = sorted(glob.glob(pat), key=os.path.getmtime)
    return files[-1] if files else None


def report(args):
    csv_path = args.csv or newest_csv()
    if not csv_path:
        print(
            "REPORT: no ops_perf_results_*.csv found. Did the run go through "
            "`python -m tracy -r -v`? Without -r there is no device profiling."
        )
        return 1
    print(f"REPORT: csv = {csv_path}")

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    # Column names carry units and vary across Tracy versions; select by name.
    dur_col = next((c for c in rows[0] if "DEVICE KERNEL DURATION" in c.upper()), None)
    name_col = next((c for c in rows[0] if c.strip().upper() == "OP CODE"), None) or list(rows[0])[0]
    if dur_col is None:
        print(f"REPORT: no DEVICE KERNEL DURATION column. Columns: {list(rows[0])[:12]}")
        return 1

    agg = {}
    for r in rows:
        name = (r.get(name_col) or "").strip()
        raw = (r.get(dur_col) or "").strip()
        if not name or not raw:
            continue
        try:
            ns = float(raw)
        except ValueError:
            continue
        agg.setdefault(name, []).append(ns)

    man = {}
    if os.path.exists(args.manifest):
        man = json.load(open(args.manifest))

    print()
    print(f"{'op':<26}{'n':>6}{'ns (median)':>13}{'~cycles':>10}{'elem/cyc':>10}")
    print("-" * 66)
    for name in sorted(agg):
        v = sorted(agg[name])
        med = v[len(v) // 2]
        cyc = med * CLOCK_GHZ
        print(f"{name:<26}{'':>6}{med:>13.0f}{cyc:>10.0f}{'':>10}   n={len(v)}")

    if man:
        unsup = [e for e in man["entries"] if e["status"] == "UNSUPPORTED"]
        ran = [e for e in man["entries"] if e["status"] == "RAN"]
        print()
        print(f"CONFIGS RAN: {len(ran)}   UNSUPPORTED: {len(unsup)}")
        for e in unsup:
            print(f"  UNSUPPORTED {e['tag']}")
            print(f"      {e['error']}")
        errs = [e for e in ran if e.get("warmup_max_abs_err", 0) > 0.05]
        if errs:
            print()
            print("WARNING: topk warmup mismatch vs torch.topk (values differ):")
            for e in errs:
                print(f"  {e['tag']}  max_abs_err={e['warmup_max_abs_err']:.4g}")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", action="store_true", help="parse newest Tracy CSV instead of measuring")
    p.add_argument("--csv", default=None)
    p.add_argument("--preset", default="moe,vocab,search")
    p.add_argument("--shapes", default=None, help="e.g. 1x32768,8x65536")
    p.add_argument("--ks", default=None, help="e.g. 8,32,64")
    p.add_argument("--ops", default="topk", help="topk,sort")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--l1-small", dest="l1_small", type=int, default=32768)
    p.add_argument("--manifest", default="/tmp/topk_bench_manifest.json")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.report:
        return report(args)
    run_measurements(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
