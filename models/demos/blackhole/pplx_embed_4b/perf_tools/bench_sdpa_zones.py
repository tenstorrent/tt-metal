# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Per-work-unit device-profiler zones of the SDPA compute kernel (NEGATIVE_RESULTS §65), for the "zones" / "zconly"
variants of sdpa_kernel_variants.py: 4 eager runs of the model's reuse_kv q128 config, the first skipped.

    cd <dir without a ttnn/ tree>
    TT_VISIBLE_DEVICES=<chip> TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=<dir> \\
        TT_METAL_KERNEL_PATH=<out>/zconly TT_METAL_CACHE=<out>/cache_zconly python bench_sdpa_zones.py <batch> <dram|l1>
    python bench_sdpa_zones.py --parse <dir>

The parse prints each zone's mean cycles per occurrence and RISC, numbered by its order inside a unit (#1 / #2 for the
two row groups).
"""

import collections
import csv
import statistics
import sys


def parse(d):
    f = open(f"{d}/.logs/profile_log_device.csv")
    next(f)
    r = csv.reader(f)
    ix = {h.strip(): i for i, h in enumerate(next(r))}
    seq, opened = collections.defaultdict(list), {}
    runs = set()
    for row in r:
        risc = row[ix["RISC processor type"]].strip()
        if risc not in ("TRISC_0", "TRISC_1", "TRISC_2"):
            continue
        z, typ = row[ix["zone name"]].strip(), row[ix["type"]].strip()
        run, t = int(row[ix["run host ID"]]), int(row[ix["time[cycles since reset]"]])
        runs.add(run)
        k = (run, row[1], row[2], risc, z)
        if typ == "ZONE_START":
            opened[k] = t
        elif k in opened:
            seq[k[:4]].append((opened.pop(k), t, z))
    first = min(runs)
    out = collections.defaultdict(list)
    for (run, *_, risc), ev in seq.items():
        if run == first:
            continue
        ev.sort()
        for us, ue, _ in (e for e in ev if e[2] == "unit"):
            cnt = collections.Counter()
            for s, e, z in (e for e in ev if e[0] >= us and e[1] <= ue and e[2] != "unit"):
                cnt[z] += 1
                out[(risc, f"{z} #{cnt[z]}")].append(e - s)
            out[(risc, "unit")].append(ue - us)
    for risc in ("TRISC_0 unpack", "TRISC_1 math", "TRISC_2 pack"):
        rk = risc.split()[0]
        names = sorted(n for (x, n) in out if x == rk)
        print(risc + ": " + "  ".join(f"{n}={statistics.mean(out[(rk, n)]):.0f}" for n in names))


def main():
    import torch

    import ttnn

    B, kvp = int(sys.argv[1]), sys.argv[2]
    D = ttnn.open_device(device_id=0, l1_small_size=32768)
    DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    CK = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    grid = (12, 8) if B == 8 else (12, 10)
    cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid), q_chunk_size=128, k_chunk_size=512, exp_approx_mode=True
    )
    torch.manual_seed(0)
    q_h, k_h, v_h = (torch.randn(B, n, 512, 128) for n in (32, 8, 8))
    kvm = L1 if kvp == "l1" else DRAM
    try:
        q = ttnn.from_torch(q_h, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=D, memory_config=DRAM)
        k = ttnn.from_torch(k_h, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=D, memory_config=kvm)
        v = ttnn.from_torch(v_h, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=D, memory_config=kvm)
        for _ in range(4):
            ttnn.deallocate(
                ttnn.transformer.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    is_causal=False,
                    scale=0.088388346,
                    program_config=cfg,
                    compute_kernel_config=CK,
                    memory_config=DRAM,
                    output_heads_concat=True,
                    reuse_kv=True,
                )
            )
        ttnn.synchronize_device(D)
        ttnn.ReadDeviceProfiler(D)
    finally:
        ttnn.close_device(D)


if __name__ == "__main__":
    parse(sys.argv[2]) if sys.argv[1] == "--parse" else main()
