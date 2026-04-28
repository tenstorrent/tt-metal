#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Compare ttnvtop's host-poll sampling against the tt-metal device profiler
# (cycle-precise ground truth). Answers three questions:
#   1. Coverage   — what fraction of profiler-seen ops did ttnvtop catch?
#   2. Time       — does ttnvtop's "frames-seen" estimate correlate with
#                   profiler's actual op duration?
#   3. Sequence   — is the inferred dispatch order correct?
#
# Verdict is printed at the end. Used to decide whether Phase 2.1.c
# (on-chip sampler) is worth building.
#
# Usage:
#   1. Run a workload with both sources enabled:
#        TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
#        TTNVTOP_REGISTER_PROGRAMS=1 \
#          <run record.py at --hz 100 in a sidecar> \
#          <run workload>
#   2. After workload exits:
#        python compare.py \
#            --profiler generated/profiler/.logs/profile_log_device.csv \
#            --ttnvtop  /tmp/ttnvtop_run.csv

import argparse
import csv
import struct
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# Decision rule (locked in upfront — change here only if you really want to)
COVERAGE_THRESHOLD = 0.70  # ops by count (attribution: cycles_total > 0)
REG_COVERAGE_THRESHOLD = 0.99  # ops registered in registry (cycles>=0)
TIME_THRESHOLD = 0.90  # ops by total time
R2_THRESHOLD = 0.85  # time correlation


# ───────────────────────────────────────────────────────────────────────────
# Registry binary parser (v3)
# ───────────────────────────────────────────────────────────────────────────
# Schema: see tt_metal/tools/ttnvtop/common/program_registry.hpp
#   Header (48 B): magic[4], version u16, entry_size u16, capacity u32,
#                  writer_pid u32, epoch_us u64, write_cursor u32, reserved[4] u32
#   Entry (128 B): runtime_id u32, pid u32, epoch_us u64, name[96],
#                  cycles_in_window u64, cycles_total u64
def parse_registry(path: str) -> Dict[int, Dict]:
    """Read /dev/shm/tt_program_registry (v3). Returns
    dict {runtime_id: {"cycles_total": int, "name": str, "epoch_us": int,
                       "dispatch_count": int}}.

    The registrar's `register_program` is called from every wired dispatch
    site and claims a fresh slot via fetch_add(write_cursor) per call —
    so the count of entries with a given runtime_id is *exactly* the
    dispatch count for that program (modulo circular-buffer wrap at
    capacity=16384). We surface this as `dispatch_count`, separate from
    `cycles_total` (Hook B sampling) — useful when the user prefers
    catching every op over estimating per-op cycles.

    Returns empty dict if version mismatch or unreadable.
    """
    out: Dict[int, Dict] = {}
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 48:
        return out
    magic, version, entry_size, capacity, writer_pid, epoch_us, cursor = struct.unpack_from("<4sHHIIQI", data, 0)
    if magic != b"TPRG":
        print(f"[compare] WARNING: registry magic mismatch ({magic!r})", file=sys.stderr)
        return out
    if version != 3:
        print(f"[compare] WARNING: registry version {version} != 3 — refusing to parse", file=sys.stderr)
        return out
    if entry_size != 128:
        print(f"[compare] WARNING: entry_size {entry_size} != 128 — refusing to parse", file=sys.stderr)
        return out
    n = min(cursor, capacity)
    HDR = 48
    for i in range(n):
        off = HDR + i * 128
        runtime_id, pid = struct.unpack_from("<II", data, off)
        (ep,) = struct.unpack_from("<Q", data, off + 8)
        name = data[off + 16 : off + 16 + 96].split(b"\x00", 1)[0].decode("utf-8", "replace")
        (cycles_total,) = struct.unpack_from("<Q", data, off + 120)
        prev = out.get(runtime_id)
        if prev is None:
            out[runtime_id] = {"cycles_total": cycles_total, "name": name, "epoch_us": ep, "dispatch_count": 1}
        else:
            prev["dispatch_count"] += 1
            if cycles_total > prev["cycles_total"]:
                prev["cycles_total"] = cycles_total
                prev["epoch_us"] = ep
    return out


# ───────────────────────────────────────────────────────────────────────────
# Profiler parser
# ───────────────────────────────────────────────────────────────────────────
def parse_profiler(path: str):
    """Parse tt-metal's profile_log_device.csv. Schema:
       PCIe slot, core_x, core_y, RISC processor type, timer_id,
       time[cycles since reset], data, run host ID, trace id, trace id counter,
       zone name, type, source line, source file, meta data

    Returns dict {runtime_id: {"cycles": int, "first_cycle": int, "events": int}}
    aggregating across all kernel zones on all cores. Only counts the actual
    kernel zones (TRISC1-KERNEL, BRISC-KERNEL, NCRISC-KERNEL) — skips firmware
    boot zones since those don't represent op dispatch time.
    """
    # tt-metal profiler emits a single "TRISC-KERNEL" zone for all three
    # TRISCs — the math/unpack/pack thread is distinguished by the
    # `RISC processor type` column ("TRISC_0", "TRISC_1", "TRISC_2"). We
    # detect Hook B's catchable subset (programs with TRISC_1 work) by
    # combining zone name + processor type, not by zone name alone.
    KERNEL_ZONES = {
        "BRISC-KERNEL",
        "NCRISC-KERNEL",
        "TRISC-KERNEL",
        "ERISC-KERNEL",
    }

    # (chip, core_x, core_y, processor, zone_name, runtime_id) -> last START cycle
    open_zones: Dict[Tuple, int] = {}
    # runtime_id -> stats. `has_math` flips true the first time we see a TRISC1
    # KERNEL zone for this rid — used downstream to compute the structural
    # attributable subset (Hook B fires only on TRISC1, so programs with no
    # TRISC1 zone are unattributable by design).
    stats: Dict[int, Dict] = defaultdict(
        lambda: {"cycles": 0, "first_cycle": -1, "events": 0, "has_math": False, "trisc1_cycles": 0}
    )

    rows_total = 0
    rows_kept = 0

    with open(path, "r") as f:
        # Skip the architecture header line.
        first = f.readline()
        if not first.startswith("ARCH:"):
            # Older format — rewind to start.
            f.seek(0)
        reader = csv.reader(f)
        header = next(reader)
        # Build column index map. Strip whitespace.
        idx = {col.strip(): i for i, col in enumerate(header)}
        REQ = [
            "PCIe slot",
            "core_x",
            "core_y",
            "RISC processor type",
            "time[cycles since reset]",
            "run host ID",
            "zone name",
            "type",
        ]
        for r in REQ:
            if r not in idx:
                print(f"profiler CSV missing column: {r!r} (have: {list(idx)})", file=sys.stderr)
                sys.exit(2)

        for row in reader:
            rows_total += 1
            if not row or len(row) <= idx["type"]:
                continue
            zone = row[idx["zone name"]].strip()
            if zone not in KERNEL_ZONES:
                continue
            try:
                raw_id = int(row[idx["run host ID"]] or 0)
                cycle = int(row[idx["time[cycles since reset]"]])
            except ValueError:
                continue
            if raw_id == 0:
                # Firmware-only events with no program assigned — skip.
                continue
            # Decode to the same prog_id space record.py uses on the ttnvtop
            # side. host_assigned_id encodes (runtime_id << 10 | device_id) on
            # mesh dispatch paths; raw runtime_id on single-device paths. The
            # >>10 decode is correct for mesh (the only case where raw_id
            # reaches the >1M range we see in the profiler dump for decode).
            # For tiny single-device workloads where the value is already raw
            # and < 1024, the shift produces 0 which we skip — those workloads
            # need a different path but aren't the target use case here.
            runtime_id = (raw_id >> 10) & 0x1FFFFF
            if runtime_id == 0:
                continue

            ztype = row[idx["type"]].strip()
            chip = row[idx["PCIe slot"]].strip()
            cx = row[idx["core_x"]].strip()
            cy = row[idx["core_y"]].strip()
            risc = row[idx["RISC processor type"]].strip()
            key = (chip, cx, cy, risc, zone, runtime_id)

            if ztype == "ZONE_START":
                open_zones[key] = cycle
                rows_kept += 1
            elif ztype == "ZONE_END":
                start = open_zones.pop(key, None)
                if start is None:
                    continue
                dur = cycle - start
                if dur < 0:
                    continue
                s = stats[runtime_id]
                s["cycles"] += dur
                s["events"] += 1
                # Hook B fires from TRISC1 (math thread) — the profiler
                # encodes it as TRISC-KERNEL zone with processor "TRISC_1".
                if zone == "TRISC-KERNEL" and risc == "TRISC_1":
                    s["has_math"] = True
                    s["trisc1_cycles"] += dur
                if s["first_cycle"] == -1 or start < s["first_cycle"]:
                    s["first_cycle"] = start
                rows_kept += 1

    print(f"  profiler rows total: {rows_total}, kernel-zone events kept: {rows_kept}", file=sys.stderr)
    return stats


# ───────────────────────────────────────────────────────────────────────────
# ttnvtop recorder parser
# ───────────────────────────────────────────────────────────────────────────
def parse_ttnvtop(path: str):
    """Parse record.py CSV. Columns:
       t_us, chip_idx, asic_id, prog_id, name, cores, f_pct, s_pct, d_pct

    Returns:
        stats:        dict {prog_id: {"frames": int, "first_t_us": int, "core_frames": int}}
        frame_period_us: estimated render period (median dt between distinct t_us)
    """
    stats: Dict[int, Dict] = defaultdict(lambda: {"frames": 0, "first_t_us": -1, "core_frames": 0})
    frame_ts: List[int] = []
    last_t = -1

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = int(row["t_us"])
                pid = int(row["prog_id"])
                cores = int(row["cores"])
            except (KeyError, ValueError):
                continue
            if pid == 0:
                continue
            if t != last_t:
                frame_ts.append(t)
                last_t = t
            s = stats[pid]
            s["frames"] += 1
            s["core_frames"] += cores
            if s["first_t_us"] == -1:
                s["first_t_us"] = t

    # Estimate frame period from inter-frame deltas (median to ignore startup jitter).
    if len(frame_ts) > 1:
        deltas = sorted(frame_ts[i + 1] - frame_ts[i] for i in range(len(frame_ts) - 1))
        period = deltas[len(deltas) // 2]
    else:
        period = 250_000  # 4 Hz default if we can't tell

    print(f"  ttnvtop frames: {len(frame_ts)}, median period: {period} us", file=sys.stderr)
    return stats, period


# ───────────────────────────────────────────────────────────────────────────
# Stats helpers (no scipy — keep deps minimal)
# ───────────────────────────────────────────────────────────────────────────
def pearson_r2(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """Returns (R², slope) of OLS y ~ x. Returns (0, 0) if undefined."""
    n = len(xs)
    if n < 2:
        return 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return 0.0, 0.0
    slope = sxy / sxx
    r2 = (sxy**2) / (sxx * syy)
    return r2, slope


def spearman(a: List[int], b: List[int]) -> float:
    """Spearman rank correlation between two equal-length sequences of ids."""
    if len(a) != len(b) or len(a) < 2:
        return 0.0

    def ranks(xs):
        # Stable rank assignment (ties get average rank).
        sorted_xs = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(sorted_xs):
            j = i
            while j + 1 < len(sorted_xs) and xs[sorted_xs[j + 1]] == xs[sorted_xs[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[sorted_xs[k]] = avg
            i = j + 1
        return r

    ra = ranks(a)
    rb = ranks(b)
    n = len(a)
    mra = sum(ra) / n
    mrb = sum(rb) / n
    sxx = sum((r - mra) ** 2 for r in ra)
    syy = sum((r - mrb) ** 2 for r in rb)
    sxy = sum((x - mra) * (y - mrb) for x, y in zip(ra, rb))
    if sxx == 0 or syy == 0:
        return 0.0
    return sxy / (sxx**0.5 * syy**0.5)


# ───────────────────────────────────────────────────────────────────────────
# Main comparison
# ───────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiler", required=True, help="profile_log_device.csv from tt-metal device profiler")
    ap.add_argument("--ttnvtop", required=True, help="record.py output CSV (host-poll stream)")
    ap.add_argument(
        "--registry",
        default=None,
        help="tt_program_registry.bin (v3). If provided, attribution coverage is computed "
        "from registry cycles_total instead of host-poll CSV.",
    )
    ap.add_argument("--aiclk-mhz", type=int, default=1000, help="AICLK rate (MHz) for cycles→us conversion")
    args = ap.parse_args()

    print(f"\nparsing profiler: {args.profiler}", file=sys.stderr)
    prof = parse_profiler(args.profiler)
    print(f"parsing ttnvtop:  {args.ttnvtop}", file=sys.stderr)
    tt, period_us = parse_ttnvtop(args.ttnvtop)

    if not prof:
        print(
            "\nERROR: profiler captured zero kernel-zone events. Either "
            "TT_METAL_DEVICE_PROFILER wasn't enabled at workload runtime, "
            "or the workload didn't dispatch any programs.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not tt:
        print(
            "\nERROR: ttnvtop recorder captured zero programs. Check that "
            "ttnvtop-collector was running and TTNVTOP_REGISTER_PROGRAMS=1 "
            "was set in the workload's environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ─── Optional registry load (Phase 2.1.c.i v3) ────────────────────────
    reg = {}
    if args.registry:
        print(f"parsing registry: {args.registry}", file=sys.stderr)
        reg = parse_registry(args.registry)
        if not reg:
            print("[compare] registry empty or unreadable — falling back to host-poll-only metrics", file=sys.stderr)

    # ─── Coverage ─────────────────────────────────────────────────────────
    prof_ids = set(prof.keys())
    tt_ids = set(tt.keys())
    reg_ids = set(reg.keys())
    reg_attributed_ids = {k for k, v in reg.items() if v["cycles_total"] > 0}

    # Phase 2.1.c.i: structural attributable subset. The LLK Hook B fires
    # from TRISC1 (math thread) only, so programs whose profiler trace has
    # NO TRISC1-KERNEL zone (data-movement-only, fabric, dispatch glue) are
    # unattributable by design. Reporting attribution against the full
    # profiler set conflates two questions; we surface both numerators with
    # the structural subset as the denominator that actually measures Hook
    # B effectiveness.
    attributable_ids = {i for i in prof_ids if prof[i].get("has_math")}

    caught = prof_ids & tt_ids  # legacy: host-poll caught
    caught_reg = prof_ids & reg_ids  # registered (any cycles)
    caught_attr = prof_ids & reg_attributed_ids  # attributed (cycles_total > 0)
    caught_attr_in_attributable = attributable_ids & reg_attributed_ids
    missed_attr = prof_ids - caught_attr
    extra = tt_ids - prof_ids

    coverage_count = len(caught) / len(prof_ids) if prof_ids else 0.0
    coverage_reg = len(caught_reg) / len(prof_ids) if prof_ids else 0.0
    coverage_attr = len(caught_attr) / len(prof_ids) if prof_ids else 0.0
    coverage_attributable = len(caught_attr_in_attributable) / len(attributable_ids) if attributable_ids else 0.0

    cycles_per_us = args.aiclk_mhz
    total_cycles = sum(s["cycles"] for s in prof.values())
    attributable_cycles = sum(prof[i]["trisc1_cycles"] for i in attributable_ids)
    caught_cycles = sum(prof[i]["cycles"] for i in caught)
    caught_attr_cycles = sum(prof[i]["cycles"] for i in caught_attr)
    coverage_time = caught_cycles / total_cycles if total_cycles else 0.0
    coverage_attr_time = caught_attr_cycles / total_cycles if total_cycles else 0.0

    # ─── Time correlation (registry path if available, else host-poll) ────
    # Registry path (preferred when --registry is given): both quantities are
    # TOTAL CYCLES the program ran on the math thread. Profiler sums kernel-
    # zone cycles per program; registry's cycles_total is the collector's
    # monotonic wall_clock_l-delta accumulator per kernel_id since collector
    # start. Same dimension → expect slope ≈ 1, R² ≥ 0.85.
    xs, ys, joined_ids = [], [], []
    if reg:
        # Compare against TRISC1-only cycles, not all-RISC totals — Hook B
        # samples only the math thread, so the apples-to-apples comparison
        # is profiler.trisc1_cycles vs registry.cycles_total. Using the
        # all-RISC profiler total here would give a slope < 1 even on a
        # perfect sampler (just because most programs have BRISC + NCRISC
        # work that the registry by definition doesn't see).
        for i in caught_attr & attributable_ids:
            prof_cyc = prof[i]["trisc1_cycles"]
            reg_cyc = reg[i]["cycles_total"]
            if prof_cyc > 0 and reg_cyc > 0:
                xs.append(prof_cyc)
                ys.append(reg_cyc)
                joined_ids.append(i)
    else:
        # Legacy: host-poll core-frames × period_us vs profiler core-us
        for i in caught:
            prof_core_us = prof[i]["cycles"] / cycles_per_us
            tt_core_us = tt[i]["core_frames"] * period_us
            if prof_core_us > 0 and tt_core_us > 0:
                xs.append(prof_core_us)
                ys.append(tt_core_us)
                joined_ids.append(i)
    r2, slope = pearson_r2(xs, ys)

    # ─── Sequence ─────────────────────────────────────────────────────────
    # Order by first-seen timestamp in each source.
    common_in_prof_order = sorted(joined_ids, key=lambda i: prof[i]["first_cycle"])
    common_in_tt_order = sorted(joined_ids, key=lambda i: tt[i]["first_t_us"])
    # Spearman over rank positions.
    pos_in_prof = {i: k for k, i in enumerate(common_in_prof_order)}
    pos_in_tt = {i: k for k, i in enumerate(common_in_tt_order)}
    seq_corr = spearman(
        [pos_in_prof[i] for i in joined_ids],
        [pos_in_tt[i] for i in joined_ids],
    )

    # ─── Verdict ──────────────────────────────────────────────────────────
    # When --registry is given, the real Hook B effectiveness gate is on
    # `coverage_attributable` — fraction of profiler ops with a TRISC1-KERNEL
    # zone that the registry attributes. The all-ops attribution coverage
    # is shown for context but isn't a fair gate (most missed ops are
    # data-movement-only and structurally unattributable by Hook B alone).
    ATTRIBUTABLE_THRESHOLD = 0.95  # ≥95% of catchable subset
    if reg:
        pass_reg = coverage_reg >= REG_COVERAGE_THRESHOLD
        pass_attr = coverage_attributable >= ATTRIBUTABLE_THRESHOLD
        pass_r2 = r2 >= R2_THRESHOLD
        # Project priority: catch every op > estimate cycles accurately.
        # Verdict gates ONLY on registration coverage. Attribution coverage
        # and R² are informational — they describe Hook B's per-op cycle
        # measurement quality, which is structurally bounded at the
        # current 5ms sample period.
        overall = pass_reg
    else:
        pass_reg = pass_attr = False
        pass_time = coverage_time >= TIME_THRESHOLD
        pass_r2 = r2 >= R2_THRESHOLD
        overall = (coverage_count >= COVERAGE_THRESHOLD) and pass_time and pass_r2

    print()
    print("=" * 60)
    print("  ttnvtop vs device profiler — comparison report")
    print("=" * 60)
    print(f"  profiler unique ops:    {len(prof_ids)}")
    print(f"    with TRISC1-KERNEL (attributable):  {len(attributable_ids)}")
    print(f"    data-movement-only (Hook B blind):  {len(prof_ids) - len(attributable_ids)}")
    print(f"  ttnvtop unique ops:     {len(tt_ids)}")
    print(f"  registry entries:       {len(reg_ids)}  ({len(reg_attributed_ids)} with cycles_total>0)")
    if reg:
        # Cumulative dispatch counts across all programs in profiler set —
        # gives a feel for whether the registry's circular buffer wrapped.
        # If sum ≪ profiler events sum, we're losing dispatches to wrap.
        total_dispatch = sum(reg.get(i, {}).get("dispatch_count", 0) for i in prof_ids)
        total_prof_events = sum(prof[i]["events"] for i in prof_ids)
        print(f"  registry dispatch sum:  {total_dispatch:,}  (profiler events: {total_prof_events:,})")
    print(f"  ops in ttnvtop only:    {len(extra)}  (usually fabric/init)")
    print()
    print(f"  Host-poll coverage (legacy):     {coverage_count:6.1%}")
    if reg:
        print(
            f"  Registration coverage:           {coverage_reg:6.1%}   "
            f"{'PASS' if pass_reg else 'FAIL':4}  (threshold {REG_COVERAGE_THRESHOLD:.0%})"
        )
        print(
            f"  Attribution / all profiler ops:  {coverage_attr:6.1%}   "
            f"      (informational; Hook B can't catch data-only ops)"
        )
        print(
            f"  Attribution / attributable ops:  {coverage_attributable:6.1%}   "
            f"{'PASS' if pass_attr else 'FAIL':4}  (threshold {ATTRIBUTABLE_THRESHOLD:.0%})  ← Hook B effectiveness"
        )
        print(f"  Attribution coverage (time):     {coverage_attr_time:6.1%}   (informational)")
    else:
        print(
            f"  Coverage by time:                {coverage_time:6.1%}   "
            f"{'PASS' if pass_time else 'FAIL':4}  (threshold {TIME_THRESHOLD:.0%})"
        )
    print(
        f"  Time correlation R²:             {r2:6.3f}   "
        f"{'PASS' if pass_r2 else 'FAIL':4}  (threshold {R2_THRESHOLD:.2f})  (TRISC1 cycles only)"
    )
    print(f"  Time scaling slope:              {slope:6.3f}   " f"(1.0 ideal; >1 = ttnvtop overestimates)")
    print(f"  Sequence agreement:              {seq_corr:6.3f}   " f"(Spearman; 1.0 = perfect order match)")

    # Show worst misses (longest profiler-cycles ops that the registry did
    # not attribute, when --registry is given; else the legacy "host-poll
    # didn't see" set).
    miss_set = (prof_ids - caught_attr) if reg else (prof_ids - caught)
    if miss_set:
        worst = sorted(miss_set, key=lambda i: -prof[i]["cycles"])[:10]
        print()
        print("  Top 10 ops missed (by profiler cycles):")
        print(f"    {'runtime_id':>11}  {'cycles':>12}  {'~us':>10}  events  in_registry?")
        for i in worst:
            us = prof[i]["cycles"] / cycles_per_us
            in_reg = "registered" if i in reg_ids else "no"
            if i in reg_ids and reg[i]["cycles_total"] == 0:
                in_reg = "registered (no cycles)"
            print(f"    {i:>11}  {prof[i]['cycles']:>12}  {us:>10.1f}  {prof[i]['events']:>6}  {in_reg}")

    print()
    print("-" * 60)
    if overall:
        print(f"  VERDICT: PASS — ttnvtop attribution coverage is sufficient")
    else:
        print(f"  VERDICT: FAIL — ttnvtop attribution coverage gaps")
        print()
        if reg:
            if not pass_reg:
                print(
                    f"  → registration coverage {coverage_reg:.1%} below {REG_COVERAGE_THRESHOLD:.0%}: "
                    f"some profiler ops never reached the registrar (missing dispatch hook?)"
                )
            if not pass_attr:
                print(
                    f"  → attributable coverage {coverage_attributable:.1%} below {ATTRIBUTABLE_THRESHOLD:.0%}: "
                    f"of the {len(attributable_ids)} ops with TRISC1 work, Hook B caught only "
                    f"{len(caught_attr_in_attributable)}. Likely persistent matmuls bypassing "
                    f"_llk_math_wait_for_dest_available_, or kernels too short for the sample period."
                )
        if not pass_r2:
            print(
                f"  → time R² {r2:.2f} below {R2_THRESHOLD:.2f}: "
                f"per-op TRISC1-cycle estimates correlate weakly with profiler. Either sample "
                f"period is too coarse for short kernels, or wall_d cap is too tight/loose."
            )
    print("=" * 60)
    print()
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
