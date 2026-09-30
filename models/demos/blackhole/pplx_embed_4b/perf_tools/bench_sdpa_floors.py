# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Traced batched SDPA at the model's reuse_kv q128 config (12x8 at bs8, 12x10 above), with K / V placements all DRAM /
K / V in L1 / all L1, for a kernel variant from sdpa_kernel_variants.py (NEGATIVE_RESULTS §65).

    cd <dir without a ttnn/ tree>
    TT_VISIBLE_DEVICES=<chip> TT_METAL_KERNEL_PATH=<out>/<variant> TT_METAL_CACHE=<out>/cache_<variant> \\
        python bench_sdpa_floors.py <label> <batch> [<batch> ...]

Prints wall-clock µs per call (median of 7 traced replays). With TT_METAL_DEVICE_PROFILER=1 and
TT_METAL_PROFILER_DIR=<dir>, `bench_sdpa_floors.py --parse <dir>` prints the device time per call in 1.35 GHz cycles
from the traced replays (eager back-to-back runs of a data-movement-only variant swing between ~484 and ~640 µs at bs32;
traced replays do not). Run one batch per process when parsing: the parser maps each 7-trace group to a placement.
"""

import collections
import csv
import statistics
import sys

ARMS = ("all DRAM", "K/V L1", "all L1")


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
    by = collections.defaultdict(list)
    for tid, c in lo:
        by[ARMS[tid // 7]].append((hi[(tid, c)] - lo[(tid, c)]) / 1350)
    print("device µs @1.35 GHz:" + "".join(f"  {a}: {statistics.median(v):.1f}" for a, v in by.items()))


def main():
    import torch

    import ttnn

    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from bench_common_traced import make_traced

    B8 = ttnn.bfloat8_b
    D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=128 * 1024 * 1024)
    traced = make_traced(D)
    DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    CK = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    label = sys.argv[1]
    try:
        for B in map(int, sys.argv[2:]):
            grid = (12, 8) if B == 8 else (12, 10)
            cfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
                q_chunk_size=128,
                k_chunk_size=512,
                exp_approx_mode=True,
            )
            q_h, k_h, v_h = (torch.randn(B, n, 512, 128) for n in (32, 8, 8))
            for name, (qm, kvm, om) in zip(ARMS, ((DRAM, DRAM, DRAM), (DRAM, L1, DRAM), (L1, L1, L1))):
                try:
                    q = ttnn.from_torch(q_h, dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=qm)
                    k = ttnn.from_torch(k_h, dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=kvm)
                    v = ttnn.from_torch(v_h, dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=kvm)
                    run = lambda: ttnn.transformer.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        is_causal=False,
                        scale=0.088388346,
                        program_config=cfg,
                        compute_kernel_config=CK,
                        memory_config=om,
                        output_heads_concat=True,
                        reuse_kv=True,
                    )
                    ttnn.deallocate(run())
                    ts = [traced(run, n=1) for _ in range(7)]
                    print(
                        f"{label:8s} bs{B:<3d} {name:9s} {statistics.median(ts):7.1f} us  (min {min(ts):.1f})",
                        flush=True,
                    )
                    for t in (q, k, v):
                        ttnn.deallocate(t)
                except Exception as e:
                    print(f"{label:8s} bs{B:<3d} {name:9s} FAILED: {str(e).splitlines()[0][:120]}", flush=True)
    finally:
        ttnn.close_device(D)


if __name__ == "__main__":
    parse(sys.argv[2]) if sys.argv[1] == "--parse" else main()
