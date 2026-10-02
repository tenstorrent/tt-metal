# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Sweep explicit program configs per case to find the achievable ("ceiling") time and which config wins.

For each selected case this runs the default selection (mode "oob") and then a set of explicit candidates
(mode "sweep"): 2D mcast, 1D in0-mcast and 1D in1-mcast, and Reuse when B is batched, over several block
shapes. Candidates that don't fit L1 or fail validation are recorded as errors. Rows use the run_suite.py
CSV format, one per candidate. `summarize.py base.csv sweep.csv --base-mode oob --new-mode sweep` compares
the default against the best candidate per case.

  python sweep_configs.py --out generated/matmul_oob/wh_sweep.csv --tiers issues --filter i40845
  python sweep_configs.py --out generated/matmul_oob/wh_sweep.csv --cases-csv generated/matmul_oob/<run>/validation/suite.csv
"""

import argparse
import csv
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_suite  # noqa: E402  (sets the profiler env vars before ttnn is imported)
from run_suite import FIELDS, CaseRun, case_fields, git_rev  # noqa: E402
from suite import TIERS, cases_from_csv, get_cases  # noqa: E402

import ttnn  # noqa: E402

# Per-core L1 left for CBs after L1-resident tensors, on a 1.5 MiB core (actual failures are still recorded)
L1_PER_CORE = 1536 * 1024
L1_RESERVED = 100 * 1024
K_BLOCKS = (1, 2, 4, 8, 16, 32, 64, 128)
DRAM_ALIGN = {"wormhole_b0": 32, "blackhole": 64}
# Data formats as matmul sees them: interm0 may be Float16_b or Float32 regardless of the output dtype
FORMAT_BYTES = {"bf16": 2048, "bfp8": 1088, "bfp4": 576, "fp32": 4096}


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def largest_divisors(n, count):
    return sorted(divisors(n), reverse=True)[:count]


def subblock(block_h, block_w, fp32_acc):
    """Largest-area (h, w) subblock dividing the block; ties prefer wider."""
    limit = 4 if fp32_acc else 8
    best = (1, 1)
    for h in divisors(block_h):
        for w in divisors(block_w):
            if h * w <= limit and (h * w, w) > (best[0] * best[1], best[1]):
                best = (h, w)
    return best


def align(n, a):
    return (n + a - 1) // a * a


def cb_bytes(case, family, arch, per_core_m, per_core_n, k, out_block_h, out_block_w, kt):
    """Per-core CB footprint of the interleaved-operand factories (see matmul-factory-l1-model notes).

    family: "2d", "1d_in0", "1d_in1" or "reuse".
    """
    dram_align = DRAM_ALIGN.get(arch, 32)
    t0 = align(FORMAT_BYTES[case.a_dtype], dram_align)
    t1 = align(FORMAT_BYTES[case.b_dtype], dram_align)
    out_fmt = case.out_dtype or case.a_dtype
    tout = FORMAT_BYTES[out_fmt]
    nblk = kt // k
    if family == "1d_in0":
        l1_acc = case.packer_l1_acc and nblk > 1
    elif family == "reuse":
        l1_acc = case.packer_l1_acc and nblk > 2
    else:
        l1_acc = case.packer_l1_acc and ((case.bias and nblk > 1) or nblk > 2)
    interm_fmt = ("fp32" if case.fp32_acc else "bf16") if l1_acc else ("fp32" if case.fp32_acc else out_fmt)
    shared = interm_fmt == out_fmt

    if family == "reuse":
        batch, M, _, _ = case.mkn
        pm_b = min(per_core_m, math.ceil(M / 32))
        in0 = pm_b * k * 2 * t0
        in1 = per_core_n * k * 2 * t1
        out_tiles = per_core_m * per_core_n
        bias = pm_b * per_core_n * FORMAT_BYTES[case.b_dtype] if case.bias else 0
    else:
        dbl = 2 if nblk > 1 else 1
        in0 = out_block_h * k * dbl * t0
        in1 = out_block_w * k * dbl * t1
        out_tiles = out_block_h * out_block_w
        bias = out_block_w * align(FORMAT_BYTES[case.b_dtype], dram_align) if case.bias else 0
    size = in0 + in1 + out_tiles * tout + (0 if shared else out_tiles * FORMAT_BYTES[interm_fmt]) + bias
    if case.transpose_a:
        size += in0
    return size


def cb_budget(case, grid):
    """Per-core L1 available for CBs: L1 minus reserved space and each core's share of L1-resident tensors."""
    from run_suite import placement_bytes

    l1_tensors, _ = placement_bytes(case)
    return L1_PER_CORE - L1_RESERVED - l1_tensors // (grid.x * grid.y)


def k_options(kt):
    opts = {k for k in K_BLOCKS if kt % k == 0}
    opts |= set(largest_divisors(kt, 2))  # includes all of K in one block
    return sorted(opts)


def block_options(case, family, arch, budget, per_core_m, per_core_n, kt, bh_ok=None):
    """(in0_block_w, out_block_h, out_block_w) choices that fit L1, largest products first."""
    opts = []
    for k in k_options(kt):
        for bh in divisors(per_core_m):
            if bh_ok and not bh_ok(bh):
                continue
            for bw in divisors(per_core_n):
                if cb_bytes(case, family, arch, per_core_m, per_core_n, k, bh, bw, kt) <= budget:
                    opts.append((k, bh, bw))
    opts.sort(key=lambda o: (o[0] * o[1] * o[2], o[1] * o[2]), reverse=True)
    return opts


# in0_block_w values the sweep covers first, so both shallow and deep K blocks are always measured
K_PREFERENCE = (2, 8, 4, 16, 32, 1, 64, 128)


def pick(opts, count):
    """Up to `count` options: the largest block for each in0_block_w, taking k values in K_PREFERENCE order
    (then any others), and then second-largest blocks, so the K-blocking tradeoff is covered."""
    by_k = {}
    for o in opts:  # opts are sorted largest product first
        by_k.setdefault(o[0], []).append(o)
    order = [k for k in K_PREFERENCE if k in by_k] + sorted(k for k in by_k if k not in K_PREFERENCE)
    chosen = []
    while len(chosen) < count and any(by_k[k] for k in order):
        for k in order:
            if by_k[k] and len(chosen) < count:
                chosen.append(by_k[k].pop(0))
    return chosen


def candidates(case, grid, arch, max_per_family):
    batch, M, K, N = case.mkn
    mt, kt, nt = math.ceil(M / 32), math.ceil(K / 32), math.ceil(N / 32)
    b_batched = math.prod(case.b_shape[:-2]) > 1
    gx, gy = grid.x, grid.y
    cores = gx * gy
    budget = cb_budget(case, grid)
    out = []

    if b_batched:
        # Reuse: per_core_N must be Nt; per_core_M divides Mt or is a multiple of Mt dividing batch*Mt
        pms = [d for d in divisors(mt)] + [mt * d for d in divisors(batch) if d > 1]
        # prefer block counts close to a multiple of the core count
        pms.sort(key=lambda pm: (math.ceil(batch * mt / pm / cores) * cores * pm) / (batch * mt))
        for pm in pms[:4]:
            for k in reversed(k_options(kt)):
                if cb_bytes(case, "reuse", arch, pm, nt, k, pm, nt, kt) > budget:
                    continue
                sh, sw = subblock(pm, nt, case.fp32_acc)
                sh = sh if mt % sh == 0 else 1  # batched A and B: Mt % out_subblock_h == 0
                out.append(
                    ttnn.MatmulMultiCoreReuseProgramConfig(
                        compute_with_storage_grid_size=(gx, gy),
                        in0_block_w=k,
                        out_subblock_h=sh,
                        out_subblock_w=sw,
                        per_core_M=pm,
                        per_core_N=nt,
                    )
                )
                if sum(1 for c in out if c.per_core_M == pm) >= 3:
                    break
        return out

    mt_fused = batch * mt

    # 2D mcast over the full grid
    pm, pn = math.ceil(mt_fused / gy), math.ceil(nt / gx)
    for k, bh, bw in pick(block_options(case, "2d", arch, budget, pm, pn, kt), max_per_family):
        sh, sw = subblock(bh, bw, case.fp32_acc)
        out.append(
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=k,
                out_subblock_h=sh,
                out_subblock_w=sw,
                out_block_h=bh,
                out_block_w=bw,
                per_core_M=pm,
                per_core_N=pn,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            )
        )

    # 1D in0 mcast (each core: all of M, a slice of N) and in1 mcast (a slice of M, all of N)
    for mcast_in0 in (True, False):
        split = nt if mcast_in0 else mt_fused
        family = "1d_in0" if mcast_in0 else "1d_in1"
        seen = set()
        for scale in (1, 2, 4):
            per_core_split = min(split, math.ceil(split / cores) * scale)
            if per_core_split in seen:
                continue
            seen.add(per_core_split)
            pm, pn = (mt_fused, per_core_split) if mcast_in0 else (per_core_split, nt)
            bh_ok = None
            if not mcast_in0 and math.ceil(mt_fused / pm) == 1:
                # single block row: Mt % out_block_h == 0 or one out block per core
                bh_ok = lambda bh, pm=pm: mt_fused % bh == 0 or pm // bh == 1
            opts = block_options(case, family, arch, budget, pm, pn, kt, bh_ok=bh_ok)
            for k, bh, bw in pick(opts, max_per_family):
                sh, sw = subblock(bh, bw, case.fp32_acc)
                out.append(
                    ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=(gx, gy),
                        in0_block_w=k,
                        out_subblock_h=sh,
                        out_subblock_w=sw,
                        out_block_h=bh,
                        out_block_w=bw,
                        per_core_M=pm,
                        per_core_N=pn,
                        fuse_batch=True,
                        fused_activation=None,
                        mcast_in0=mcast_in0,
                        num_global_cb_receivers=0,
                    )
                )
    return out


def reduced_grid_candidates(case, grid, arch):
    """Each family on fewer cores than the full grid: 2D on sub-grids, 1D and Reuse on 1..all cores.

    in0_block_w is capped at 8. Used to check whether small matmuls run faster on fewer cores.
    """
    batch, M, K, N = case.mkn
    mt, kt, nt = math.ceil(M / 32), math.ceil(K / 32), math.ceil(N / 32)
    if math.prod(case.b_shape[:-2]) > 1:
        return []  # batched B: covered by candidates()
    mf = batch * mt
    gx, gy = grid.x, grid.y
    budget = cb_budget(case, grid)
    out = []

    def best_blocks(family, pm, pn, count):
        opts = [o for o in block_options(case, family, arch, budget, pm, pn, kt) if o[0] <= 8]
        return opts[:count]

    sub_grids = sorted({(max(1, gx // sx), max(1, gy // sy)) for sx in (1, 2, 4, 8) for sy in (1, 2, 4, 8)})
    for sgx, sgy in sub_grids:
        pm, pn = math.ceil(mf / sgy), math.ceil(nt / sgx)
        for k, bh, bw in best_blocks("2d", pm, pn, 1):
            sh, sw = subblock(bh, bw, case.fp32_acc)
            out.append(
                ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(gx, gy),
                    in0_block_w=k,
                    out_subblock_h=sh,
                    out_subblock_w=sw,
                    out_block_h=bh,
                    out_block_w=bw,
                    per_core_M=pm,
                    per_core_N=pn,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=True,
                )
            )
    for cores in (1, 2, 4, 8, 16, 32, 64, 128):
        if cores > gx * gy:
            break
        pn = math.ceil(nt / cores)
        for k, bh, bw in best_blocks("1d_in0", mf, pn, 1):
            sh, sw = subblock(bh, bw, case.fp32_acc)
            out.append(
                ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(gx, gy),
                    in0_block_w=k,
                    out_subblock_h=sh,
                    out_subblock_w=sw,
                    out_block_h=bh,
                    out_block_w=bw,
                    per_core_M=mf,
                    per_core_N=pn,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                    num_global_cb_receivers=0,
                )
            )
        # Reuse (B not batched): per_core_N = Nt, cores rows of the (fused) M split, restricted by the grid size
        if batch == 1:
            rows = [d for d in divisors(mt) if mt // d <= cores]
            if rows:
                pm = min(rows)
                ggx, ggy = min(cores, gx), max(1, min(gy, cores // gx))
                for k in reversed([k for k in k_options(kt) if k <= 8]):
                    if cb_bytes(case, "reuse", arch, pm, nt, k, pm, nt, kt) <= budget:
                        sh, sw = subblock(pm, nt, case.fp32_acc)
                        out.append(
                            ttnn.MatmulMultiCoreReuseProgramConfig(
                                compute_with_storage_grid_size=(ggx, ggy),
                                in0_block_w=k,
                                out_subblock_h=sh,
                                out_subblock_w=sw,
                                per_core_M=pm,
                                per_core_N=nt,
                            )
                        )
                        break
    unique = {}
    for config in out:
        unique.setdefault(repr(config), config)
    return list(unique.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="generated/matmul_oob/sweep.csv")
    parser.add_argument("--tiers", nargs="+", choices=list(TIERS), default=None)
    parser.add_argument("--cases-csv", default=None, help="take the cases from an earlier results CSV instead")
    parser.add_argument("--filter", default=None, help="regex on case name")
    parser.add_argument("--exclude-tags", nargs="*", default=[])
    parser.add_argument("--max-per-family", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--pcc-threshold", type=float, default=0.99)
    parser.add_argument("--pcc-max-flops", type=float, default=2e12)
    parser.add_argument("--resume", action="store_true", help="skip cases already in --out")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--reduced-grids", action="store_true", help="sweep each family on fewer cores instead of the full grid"
    )
    args = parser.parse_args()

    cases = cases_from_csv(args.cases_csv) if args.cases_csv else get_cases(args.tiers)
    if args.filter:
        cases = [c for c in cases if re.search(args.filter, c.name)]
    if args.exclude_tags:
        cases = [c for c in cases if not set(c.tags) & set(args.exclude_tags)]
    # Explicit configs here assume interleaved operands and no user core_grid
    cases = [
        c
        for c in cases
        if c.a_mem in ("dram", "l1") and c.b_mem in ("dram", "l1") and c.out_mem in ("dram", "l1") and not c.core_grid
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out_path.exists():
        with open(out_path) as f:
            done = {r["case"] for r in csv.DictReader(f)}
    elif out_path.exists():
        sys.exit(f"{out_path} exists; pass --resume to continue it or choose another --out")
    cases = [c for c in cases if c.name not in done]
    print(f"{len(cases)} cases -> {out_path}")

    git = git_rev()
    device = ttnn.open_device(device_id=args.device_id)
    arch = str(device.arch()).split(".")[-1].lower()
    grid = device.compute_with_storage_grid_size()
    seen_programs = set()
    write_header = not out_path.exists()
    try:
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                writer.writeheader()
            for i, case in enumerate(cases):
                t0 = time.time()
                fields = case_fields(case, arch, grid, git)
                try:
                    run = CaseRun(case, device, args, seen_programs)
                except (CaseRun.Infeasible, CaseRun.SetupError) as e:
                    status = "infeasible" if isinstance(e, CaseRun.Infeasible) else "setup_error"
                    writer.writerow({**fields, "mode": "oob", "status": status, "error": str(e)})
                    continue
                try:
                    base = run.measure("oob")
                    writer.writerow({**fields, **base, "mode": "oob"})
                    best = None
                    cands = (
                        reduced_grid_candidates(case, grid, arch)
                        if args.reduced_grids
                        else candidates(case, grid, arch, args.max_per_family)
                    )
                    for config in cands:
                        row = run.measure("oob", program_config=config)
                        writer.writerow({**fields, **row, "mode": "sweep"})
                        if row["status"] == "ok" and (best is None or row["device_ns"] < best["device_ns"]):
                            best = row
                        device.clear_program_cache()
                    f.flush()
                finally:
                    run.close()
                b_us = f"{base['device_ns'] / 1e3:9.1f}" if base.get("device_ns") else "      n/a"
                s_us = f"{best['device_ns'] / 1e3:9.1f}" if best else "      n/a"
                gain = f"{base['device_ns'] / best['device_ns']:5.2f}x" if best and base.get("device_ns") else "   n/a"
                print(
                    f"[{i + 1}/{len(cases)}] {case.name:44s} oob {b_us}us  best {s_us}us  {gain}  "
                    f"({len(cands)} cands, {time.time() - t0:.0f}s)  {best['config'][:110] if best else ''}",
                    flush=True,
                )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
