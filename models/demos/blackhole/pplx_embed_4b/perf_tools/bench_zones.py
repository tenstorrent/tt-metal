# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-work-unit device-profiler zones of the SDPA compute kernel, for one arm of bench_row_sums.py's arm file
with math_thread_row_sums off or on. Use with the "zconly" (phase zones) or "zfconly" (the kernel's own
SUB / EXP / PACK SUB_EXP zones) variant of sdpa_kernel_variants.py.

    TT_VISIBLE_DEVICES=<chip> TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=<dir> \
      TT_METAL_KERNEL_PATH=<out>/zconly TT_METAL_CACHE=<out>/cache_zconly \
      python bench_zones.py <arms.json> <arm name> <0|1>
    python bench_zones.py --parse <dir>          # zones nested in the "unit" zone (zconly)
    python bench_zones.py --parse-flat <dir>     # every zone, no nesting (zfconly: there is no "unit" zone)
"""
import collections
import csv
import json
import statistics
import sys


def _rows(d):
    f = open(f"{d}/.logs/profile_log_device.csv")
    next(f)
    r = csv.reader(f)
    ix = {h.strip(): i for i, h in enumerate(next(r))}
    seq, opened, runs = collections.defaultdict(list), {}, set()
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
    return seq, min(runs)


def _print(out):
    for risc in ("TRISC_0 unpack", "TRISC_1 math", "TRISC_2 pack"):
        rk = risc.split()[0]
        names = sorted(n for (x, n) in out if x == rk)
        if not names:
            continue
        print(risc + ":")
        for n in names:
            v = out[(rk, n)]
            print(f"    {n:28s} n={len(v):5d}  mean={statistics.mean(v):8.0f}  median={statistics.median(v):8.0f}")


def parse(d):
    seq, first = _rows(d)
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
    _print(out)


def parse_flat(d):
    seq, first = _rows(d)
    out = collections.defaultdict(list)
    for (run, *_, risc), ev in seq.items():
        if run == first:
            continue
        for s, e, z in ev:
            out[(risc, z)].append(e - s)
    _print(out)


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
    D = ttnn.open_device(device_id=0, l1_small_size=32768)
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
        for _ in range(4):
            ttnn.deallocate(
                ttnn.transformer.scaled_dot_product_attention(
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
            )
        ttnn.synchronize_device(D)
        ttnn.ReadDeviceProfiler(D)
    finally:
        ttnn.close_device(D)


if __name__ == "__main__":
    if sys.argv[1] == "--parse":
        parse(sys.argv[2])
    elif sys.argv[1] == "--parse-flat":
        parse_flat(sys.argv[2])
    else:
        main()
