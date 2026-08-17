#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Driver: shard the ff2 block-shape sweep into small, profiler-friendly batches.

For the fixed LTX video-ff2 shape (4864/4096/4096) on the (12,8) MM grid, sweeps
mm_block_m / mm_block_k / mm_block_n. Running all ~2025 combos under one tracy capture would
overwhelm the profiler, so this driver breaks it up:

    for M_block in [m_lo .. m_hi]:
        run batch (1 M) x (all K blocks) x (first half of the N block range)
        run batch (1 M) x (all K blocks) x (second half of the N block range)   # reaches N end
        # then reset N to the first half and increment M

Each batch is run as `python -m tracy -r -m pytest ...` (via the SWEEP_*_BLOCKS env the test
reads). tracy prints a line like:
    OPs csv generated at: /home/.../ops_perf_results_YYYY_..._.csv
We parse that path, run `tt-perf-report --ignore-signposts --csv <merged> <raw>` to get the
device-merged per-op time, and join it (by op call-count ID) to the raw CSV's ATTRIBUTES, which
carry M_block_size / K_block_size / N_block_size. The fastest config per batch (and overall) is
reported, and every timed config is written to a results CSV.

Run from the repo root:
    ./python_env/bin/python tests/nightly/tg/ccl/run_ff2_block_sweep.py
"""
import argparse
import csv
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
TEST_FILE = "tests/nightly/tg/ccl/test_minimal_matmul_strided_reduce_scatter_async.py"
TEST_NODE = TEST_FILE + "::test_minimal_matmul_strided_reduce_scatter_block_sweep"
OP_NAME = "MinimalMatmulStridedReduceScatterAsync"

# Must match the fixed shape/grid in the test (see _SWEEP_M/_SWEEP_GRID there).
M_DIM, K_DIM, N_DIM = 2656, 3456, 5120
GRID_X, GRID_Y = 12, 8
TILE = 32

PY = sys.executable
TT_PERF_REPORT = os.path.join(os.path.dirname(PY), "tt-perf-report")
_CSV_RE = re.compile(r"OPs csv generated at:\s*(\S+\.csv)")


def n_block_range():
    """Valid N block tiles for this shape/grid: 2 .. min(16, Nt // grid.x)."""
    nt_per_core = (N_DIM // TILE) // GRID_X  # 160 // 12 = 13
    return list(range(2, min(16, nt_per_core) + 1))


def n_halves():
    r = n_block_range()
    mid = (len(r) + 1) // 2
    return [r[:mid], r[mid:]]


K_BLOCK_RANGE = list(range(2, 17))  # K block tiles the sweep covers


def k_halves():
    """Split the K range so each batch stays within what one tracy capture can hold.

    Batch size is the binding constraint on the profiler, not the device: too many ops under one
    capture and the post-run pass dies with "Start and end marker IDs do not match" (and takes the
    whole run down with it, since the aborted child never exits). Halving K alongside the existing N
    halves keeps a batch to roughly a quarter of the (K x N) grid for one M.
    """
    mid = (len(K_BLOCK_RANGE) + 1) // 2
    return [K_BLOCK_RANGE[:mid], K_BLOCK_RANGE[mid:]]


def _k_expr(packet):
    return f"block_sweep and axis_0 and " + ("8kib" if packet == "8k" else "not 8kib")


def run_batch(m_block, k_half, n_half, packet):
    """Run one tracy-profiled batch; return list of (device_time_us, M_block, K_block, N_block)."""
    env = dict(os.environ)
    env["SWEEP_M_BLOCKS"] = str(m_block)
    env["SWEEP_N_BLOCKS"] = f"{n_half[0]}:{n_half[-1]}"
    env["SWEEP_K_BLOCKS"] = f"{k_half[0]}:{k_half[-1]}"
    # tracy re-shells the command, so the -k expression (which contains spaces) must be wrapped in
    # single quotes so it survives as one argument.
    cmd = [PY, "-m", "tracy", "-r", "-m", "pytest", TEST_NODE, "-k", f"'{_k_expr(packet)}'"]
    print(
        f"  $ SWEEP_M_BLOCKS={m_block} SWEEP_K_BLOCKS={env['SWEEP_K_BLOCKS']} "
        f"SWEEP_N_BLOCKS={env['SWEEP_N_BLOCKS']} {' '.join(cmd)}",
        flush=True,
    )
    # Stream the child's output live (merged stdout+stderr) while capturing it so we can still parse
    # out the 'OPs csv generated at:' path afterwards.
    proc = subprocess.Popen(
        cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    captured = []
    for line in proc.stdout:
        sys.stdout.write(line)
        captured.append(line)
    sys.stdout.flush()
    proc.wait()
    out = "".join(captured)
    m = _CSV_RE.search(out)
    if not m:
        print("  !! no 'OPs csv generated at' line found. Last 25 lines of output:")
        print("\n".join(out.splitlines()[-25:]))
        return []
    return analyze(m.group(1))


def analyze(raw_csv):
    if not os.path.isabs(raw_csv):
        raw_csv = os.path.join(REPO_ROOT, raw_csv)
    if not os.path.exists(raw_csv):
        print(f"  !! ops csv not found: {raw_csv}")
        return []

    merged = raw_csv[:-4] + "_ttpr.csv"
    tpr = subprocess.run(
        [TT_PERF_REPORT, "--ignore-signposts", "--csv", merged, raw_csv],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if not os.path.exists(merged):
        print("  !! tt-perf-report produced no CSV. stderr tail:")
        print("\n".join((tpr.stdout + tpr.stderr).splitlines()[-15:]))
        return []

    # Identify each config by the block sizes in the raw ATTRIBUTES (all configs share the same op
    # name + shape, so ATTRIBUTES is the only discriminator). Group the raw rows by (M,K,N block);
    # each group is one config, spread over its per-device rows. Track the group's execution order
    # (min GLOBAL CALL COUNT) and its raw device-kernel durations (max across devices = wall-clock).
    groups = {}
    with open(raw_csv, newline="") as f:
        for row in csv.DictReader(f):
            if OP_NAME not in row.get("OP CODE", ""):
                continue
            attr = row.get("ATTRIBUTES", "")
            mb = re.search(r"M_block_size=(\d+)", attr)
            kb = re.search(r"K_block_size=(\d+)", attr)
            nb = re.search(r"N_block_size=(\d+)", attr)
            if not (mb and kb and nb):
                continue
            key = (int(mb.group(1)), int(kb.group(1)), int(nb.group(1)))
            try:
                gcc = int(row.get("GLOBAL CALL COUNT", "0") or 0)
            except ValueError:
                gcc = 0
            try:
                dur_us = float(row.get("DEVICE KERNEL DURATION [ns]", "0") or 0) / 1000.0
            except ValueError:
                dur_us = 0.0
            g = groups.setdefault(key, {"min_gcc": gcc, "durs": []})
            g["min_gcc"] = min(g["min_gcc"], gcc)
            g["durs"].append(dur_us)
    # configs in execution order
    configs = sorted(groups.items(), key=lambda kv: kv[1]["min_gcc"])

    # tt-perf-report's device-merged Device Time [us], in execution order (ID ascending). Its ID is
    # an exec-order index, not a joinable key, so we rank-align it to the configs. When the counts
    # match (the normal case: each config runs once), use the merged time; otherwise fall back to
    # the raw max device-kernel duration.
    merged_times = []
    with open(merged, newline="") as f:
        for row in csv.DictReader(f):
            if OP_NAME in row.get("OP Code", ""):
                try:
                    merged_times.append((int(row["ID"]), float(row["Device Time"])))
                except (KeyError, ValueError):
                    pass
    merged_times.sort()

    results = []
    if len(merged_times) == len(configs) and merged_times:
        for (key, _g), (_id, t) in zip(configs, merged_times):
            results.append((t, *key))
    else:
        print(
            f"  (note: {len(merged_times)} merged ops vs {len(configs)} configs; "
            "using raw max device-kernel duration instead of tt-perf-report)"
        )
        for key, g in configs:
            results.append((max(g["durs"]) if g["durs"] else 0.0, *key))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m-lo", type=int, default=2, help="lowest M block (tiles)")
    ap.add_argument("--m-hi", type=int, default=16, help="highest M block (tiles)")
    ap.add_argument("--packet", default="8k", choices=["4k", "8k"], help="fabric packet payload (default 8k)")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "generated", "ff2_block_sweep_results.csv"))
    args = ap.parse_args()

    print(f"repo={REPO_ROOT}\npython={PY}\ntt-perf-report={TT_PERF_REPORT}")
    print(f"N block range={n_block_range()}  halves={n_halves()}  packet={args.packet}\n")

    all_results = []
    for m in range(args.m_lo, args.m_hi + 1):
        for k_half in k_halves():
            for half in n_halves():
                if not half or not k_half:
                    continue
                print(
                    f"=== M_block={m}  K_block {k_half[0]}..{k_half[-1]}  N_block {half[0]}..{half[-1]} ===",
                    flush=True,
                )
                res = run_batch(m, k_half, half, args.packet)
                all_results.extend(res)
                if res:
                    bt, bm, bk, bn = min(res)
                    print(
                        f"    batch best: {bt:9.2f} us   M{bm} K{bk} N{bn}   ({len(res)} configs timed)\n",
                        flush=True,
                    )
                else:
                    print("    (no results parsed for this batch)\n", flush=True)

    all_results.sort()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device_time_us", "M_block", "K_block", "N_block"])
        for t, bm, bk, bn in all_results:
            w.writerow([f"{t:.3f}", bm, bk, bn])

    print("==== TOP 10 (lowest device time) ====")
    for t, bm, bk, bn in all_results[:10]:
        print(f"  {t:9.2f} us   M_block={bm} K_block={bk} N_block={bn}")
    if all_results:
        t, bm, bk, bn = all_results[0]
        print(f"\nGLOBAL BEST: {t:.2f} us  M_block={bm} K_block={bk} N_block={bn}")
    print(f"\nFull results ({len(all_results)} configs): {args.out}")


if __name__ == "__main__":
    main()
