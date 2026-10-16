# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device cycles per SDPA call from traced replays, for one arm of bench_row_sums.py's arm file with
math_thread_row_sums off or on.

Wall-clock us/call is unreliable here: Blackhole's AICLK drifts with load and the batch-16 pplx shape sits
between two clock states, so the same config re-measured minutes later moves by ~9%. Device cycles do not
depend on the clock.

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=<dir> TT_VISIBLE_DEVICES=<chip> \
        python bench_cycles.py <arms.json> <arm name> <0|1>
    python bench_cycles.py --parse <dir>
"""
import collections
import csv
import json
import statistics
import sys

REPLAYS = int(__import__("os").environ.get("REPLAYS", 11))


def parse(d):
    f = open(f"{d}/.logs/profile_log_device.csv")
    next(f)
    r = csv.reader(f)
    ix = {h.strip(): i for i, h in enumerate(next(r))}
    lo, hi = collections.defaultdict(lambda: 1 << 62), collections.defaultdict(int)
    for row in r:
        if not row[ix["zone name"]].strip().endswith("-FW"):
            continue
        tid = row[ix["trace id"]].strip()
        if tid == "":
            continue
        key = (int(tid), row[ix["trace id counter"]].strip())
        t = int(row[ix["time[cycles since reset]"]])
        lo[key], hi[key] = min(lo[key], t), max(hi[key], t)
    # Drop the first replay of each trace (cold caches).
    per = collections.defaultdict(list)
    for k in sorted(lo, key=lambda k: (k[0], int(k[1]))):
        per[k[0]].append(hi[k] - lo[k])
    out = [v for tid in sorted(per) for v in per[tid][1:]]
    print(
        json.dumps(
            {"n": len(out), "median_cycles": statistics.median(out), "min_cycles": min(out), "max_cycles": max(out)}
        )
    )


def main():
    import torch

    import ttnn

    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from bench_row_sums import DEFAULTS, DT, FID

    arms = json.load(open(sys.argv[1]))
    a = dict(DEFAULTS)
    a.update(next(x for x in arms if x["name"] == sys.argv[2]))
    flag = sys.argv[3] == "1"

    B, nqh, nkh, s, d = a["B"], a["nqh"], a["nkh"], a["s"], a["d"]
    skv = a["skv"] or s
    D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=200 * 1024 * 1024)
    DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    qm, kvm, om = {"dram": (DRAM, DRAM, DRAM), "kvl1": (DRAM, L1, DRAM), "l1": (L1, L1, L1)}[a["placement"]]
    CK = ttnn.WormholeComputeKernelConfig(
        math_fidelity=FID[a["fid"]], math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*a["grid"]),
        q_chunk_size=a["qc"],
        k_chunk_size=a["kc"],
        exp_approx_mode=a["exp_approx"],
    )
    torch.manual_seed(0)
    dt = DT[a["dtype"]]
    try:
        q = ttnn.from_torch(torch.randn(B, nqh, s, d), dtype=dt, layout=ttnn.TILE_LAYOUT, device=D, memory_config=qm)
        k = ttnn.from_torch(torch.randn(B, nkh, skv, d), dtype=dt, layout=ttnn.TILE_LAYOUT, device=D, memory_config=kvm)
        v = ttnn.from_torch(torch.randn(B, nkh, skv, d), dtype=dt, layout=ttnn.TILE_LAYOUT, device=D, memory_config=kvm)
        run = lambda: ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=d**-0.5,
            program_config=cfg,
            compute_kernel_config=CK,
            memory_config=om,
            output_concat_heads=a["concat"],
            pack_gqa_heads=a["pack_gqa"],
            reuse_kv=a["reuse_kv"],
            math_thread_row_sums=flag,
        )
        for _ in range(2):
            ttnn.deallocate(run())
        ttnn.synchronize_device(D)
        tid = ttnn.begin_trace_capture(D, cq_id=0)
        out = run()
        ttnn.end_trace_capture(D, tid, cq_id=0)
        for _ in range(REPLAYS):
            ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
        ttnn.synchronize_device(D)
        ttnn.ReadDeviceProfiler(D)
        ttnn.release_trace(D, tid)
        ttnn.deallocate(out)
    finally:
        ttnn.close_device(D)


if __name__ == "__main__":
    parse(sys.argv[2]) if sys.argv[1] == "--parse" else main()
