#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Analyze the Quasar device-profiler CSV (profile_log_device.csv) for the single-core add.

Produces, from the per-(core, RISC) zone start/end timestamps:
  * single mode  — a per-role table of zone durations (cycles) for one run.
  * sweep  mode  — per-role cycles/tile SLOPE (+ intercept, R²) across a tile-count sweep,
                   a memory-vs-compute-vs-dataflow bottleneck verdict, and a pipeline overlap check.

WHY this works without a global clock: every duration here is END - START on the SAME (core, RISC),
i.e. a same-core rdcycle delta. Those are the only clock-trustworthy quantities on Quasar (there is
no cross-core sync; WALL_CLOCK hangs the emulator). We never compare timestamps across cores, and we
never convert to ns (CHIP_FREQ is 0 on the emulator — cycles only).

Pipeline role map (compute kernel is one binary compiled 4x onto the TRISCs of a NEO):
    TRISC_0 = UNPACK  (L1 -> regs)      -> MEMORY
    TRISC_1 = MATH    (FPU add)         -> COMPUTE
    TRISC_2 = PACK    (regs -> L1)      -> MEMORY
    BRISC/NCRISC = DM reader/writer     -> DATAFLOW

CAVEATS (also printed in the report):
  * Cycles only (freq=0); never ns.
  * Emulator is single-sample (deterministic but one rep); slopes need >=3 points. R² shows fit quality.
  * The 4 TRISCs are issue THREADS (UNPACK/MATH/PACK/ISOLATE_SFPU), not the fixed-function engines
    they drive (3 unpackers / 2 packers / vector unit) — those engines emit NO rows, so we only ever
    see a thread's stall on its engine, never the engine itself.
  * The host collapses trisc_id 2 (PACK) and 3 (ISOLATE_SFPU) onto one TRISC_2 label (no TRISC_3 in
    tracy's enum), so a TRISC_2 group holds both the PACK thread and the (idle-for-add) SFPU thread.
    We take MAX per (role, zone) — the busy PACK sibling — and report the count. trisc_id is lost in
    the CSV, so true 2-vs-3 separation needs a device-side change (stamp trisc_id into the marker).
  * Raw per-role durations are STALL-CONTAMINATED (a role's cycles include time blocked on another
    thread / DFB credits, not just its own work). Therefore the SLOPE (cycles/tile) is the headline
    signal, not the single biggest raw number.
"""

import argparse
import json
import sys
from collections import defaultdict

import pandas as pd

# RISC identity -> (role, category). These are the 4 programmable TRISC ISSUE THREADS, not the
# fixed-function engines they drive (unpackers/packers/SFPU emit no profiler rows — we only ever see a
# thread's own time + its stall on its engine). trisc_id->role is source-verified
# (emulated_program_runner.cpp:118-119 "0=UNPACK 1=MATH 2=PACK 3=ISOLATE_SFPU"; LLK TRISC_ID consts;
# genfiles.cpp:391 compiles 4 variants in that order).
#
# RESOLVED identities (TRISC_0..TRISC_3): when the CSV is "stamped" (profiler.cpp writes the true
# trisc_id into the data column), we key off the true trisc_id, so TRISC_2 (PACK) and TRISC_3
# (ISOLATE_SFPU) are SEPARATE rows. On unstamped CSVs the host collapses 2&3 onto the TRISC_2 label
# (tracy enum has no TRISC_3) and we fall back to max-over-group.
#
# CATEGORY tags (MEMORY/COMPUTE/DATAFLOW) describe what the thread is DOING (issuing data-movement vs
# math), used by the verdict. UNVERIFIED: that UNPACK/PACK threads are the "memory" roles is the
# conventional split, not independently confirmed here — treat the category as interpretation.
# DM (Rocket data-movement) cores are labeled DM<n> by core index (mhartid), stamped into data.
# KNOWN GAP (verified dm.cc): only DM0 (hartid==0) runs the firmware path that emits the "DM0-FW"
# profiler zone; DM1+ run a SEPARATE subordinate path (dm.cc `if (hartid>0)`) that has NO profiler
# zone. So the device CSV currently shows ONLY DM0's firmware span — the reader kernel (DM1) and even
# DM0's writer KERNEL zone are absent. The DM-index stamp below is correct and ready, but until the
# subordinate DM path is instrumented, any "dataflow" number from this CSV is DM0-firmware-only and
# UNDERCOUNTS the reader+writer. (Per-DM-kernel timing does exist via the rdcycle kernel-timer DPRINT
# path, separate from this CSV.)
ROLE_MAP = {
    "BRISC": ("DM0 (data movement)", "DATAFLOW"),
    "NCRISC": ("DM (data movement)", "DATAFLOW"),
    "DM0": ("DM0 (data movement)", "DATAFLOW"),
    "DM1": ("DM1 (data movement)", "DATAFLOW"),
    "DM2": ("DM2 (data movement)", "DATAFLOW"),
    "DM3": ("DM3 (data movement)", "DATAFLOW"),
    "TRISC_0": ("UNPACK thread (trisc_id 0)", "MEMORY"),
    "TRISC_1": ("MATH thread (trisc_id 1)", "COMPUTE"),
    "TRISC_2": ("PACK thread (trisc_id 2)", "MEMORY"),
    "TRISC_3": ("ISOLATE_SFPU thread (trisc_id 3)", "SFPU"),
}

# Columns of profile_log_device.csv (line 1 is the ARCH banner, line 2 is this header).
COLS = [
    "chip",
    "core_x",
    "core_y",
    "risc",
    "timer_id",
    "cycle",
    "data",
    "run_host_id",
    "trace_id",
    "trace_id_counter",
    "zone_name",
    "type",
    "src_line",
    "src_file",
    "meta",
]


def read_device_csv(path):
    """Return (arch, freq, max_cores, DataFrame). DataFrame has the COLS above, skipping the banner."""
    with open(path) as f:
        banner = f.readline().strip()
    arch, freq, max_cores = "unknown", 0, 0
    if banner.startswith("ARCH"):
        # "ARCH: quasar, CHIP_FREQ[MHz]: 0, Max Compute Cores: 1"
        parts = [p.strip() for p in banner.split(",")]
        arch = parts[0].split(":")[-1].strip()
        freq = int(parts[1].split(":")[-1].strip())
        max_cores = int(parts[2].split(":")[-1].strip())
    df = pd.read_csv(path, skiprows=1, header=0, names=COLS, na_filter=False)
    # Coerce the numeric fields we use. 'data' carries the stamped trisc_id on Quasar TRISC rows.
    for c in ("core_x", "core_y", "cycle", "run_host_id", "data"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return arch, freq, max_cores, df


# trisc_id -> the de-collapsed RISC identity used for ROLE_MAP lookups.
_TRISC_ID_TO_LABEL = {0: "TRISC_0", 1: "TRISC_1", 2: "TRISC_2", 3: "TRISC_3"}


def csv_is_stamped(df):
    """A stamped CSV carries the true trisc_id in 'data' on TRISC rows; the unambiguous signature is a
    TRISC row with data==3 (unstamped CSVs have data==0 everywhere for these zones)."""
    trisc = df[df["risc"].astype(str).str.startswith("TRISC")]
    return bool((trisc["data"] == 3).any())


def effective_risc(row, stamped):
    """The RISC identity to attribute a row to. On a stamped CSV:
      - TRISC rows: true identity from the stamped trisc_id in 'data' (TRISC_2 splits into 2=PACK,
        3=ISOLATE_SFPU).
      - DM rows (BRISC/NCRISC labels): identity from the stamped DM core index in 'data' -> DM<n>,
        so the writer (DM0) and reader (DM1) are distinct instead of both reading as one BRISC row.
    Otherwise use the CSV's RISC label as-is."""
    risc = str(row.risc)
    if stamped and risc.startswith("TRISC") and row.data in _TRISC_ID_TO_LABEL:
        return _TRISC_ID_TO_LABEL[int(row.data)]
    if stamped and risc in ("BRISC", "NCRISC") and not pd.isna(row.data):
        return f"DM{int(row.data)}"
    return risc


def zone_durations(df):
    """
    Pair ZONE_START/ZONE_END per (core, effective-risc, zone_name) using a stack, returning dicts:
      {core, risc, zone, dur, role, category}.
    On a stamped CSV the effective-risc comes from the true trisc_id (data column), so PACK (2) and
    ISOLATE_SFPU (3) become distinct identities instead of colliding on the TRISC_2 label. The stack
    still tolerates genuine same-identity concurrency.
    """
    out = []
    stamped = csv_is_stamped(df)
    stacks = defaultdict(list)  # (core_x, core_y, eff_risc, zone_name) -> [start_cycle, ...]
    # Stable order: as written in the file (already time-ish ordered per risc).
    for r in df.itertuples(index=False):
        eff = effective_risc(r, stamped)
        key = (r.core_x, r.core_y, eff, r.zone_name)
        if r.type == "ZONE_START":
            stacks[key].append(r.cycle)
        elif r.type == "ZONE_END":
            if stacks[key]:
                start = stacks[key].pop()
                role, cat = ROLE_MAP.get(eff, (eff, "OTHER"))
                out.append(
                    {
                        "core": (int(r.core_x), int(r.core_y)),
                        "risc": eff,
                        "zone": r.zone_name,
                        "dur": int(r.cycle - start),
                        "role": role,
                        "category": cat,
                    }
                )
    return out


def aggregate_roles(durations):
    """
    Collapse to one row per (risc, zone), taking MAX duration (busy sibling under label collapse)
    and recording how many instances were merged. Returns list of dicts sorted by risc then zone.
    """
    groups = defaultdict(list)
    for d in durations:
        groups[(d["risc"], d["zone"])].append(d)
    rows = []
    for (risc, zone), items in groups.items():
        durs = [i["dur"] for i in items]
        role, cat = ROLE_MAP.get(risc, (risc, "OTHER"))
        rows.append(
            {
                "risc": risc,
                "zone": zone,
                "role": role,
                "category": cat,
                "dur_max": max(durs),
                "dur_min": min(durs),
                "instances": len(durs),
            }
        )
    rows.sort(key=lambda x: (x["risc"], x["zone"]))
    return rows


def kernel_dur_by_risc(rows):
    """Map risc -> max KERNEL-zone duration (the per-role work proxy used for slopes/verdict)."""
    out = {}
    for r in rows:
        if "KERNEL" in r["zone"]:
            out[r["risc"]] = max(out.get(r["risc"], 0), r["dur_max"])
    return out


def fit_line(xs, ys):
    """Least-squares slope/intercept/R² for ys = slope*xs + intercept. Pure-python (no numpy dep)."""
    n = len(xs)
    if n < 2:
        return None
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    # R²
    mean_y = sy / n
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"slope": slope, "intercept": intercept, "r2": r2}


CAVEATS = [
    "Cycles only (CHIP_FREQ=0 on the emulator); never interpret as ns.",
    "Emulator is single-sample (deterministic but one rep). Slopes need >=3 tile points; check R².",
    "TRISC labels are issue THREADS, not the engines they drive (unpackers/packers/SFPU emit no rows; "
    "a thread's time = its work + stall on its engine). STAMPED CSVs split all 4 threads via the true "
    "trisc_id in the data column (TRISC_3 = ISOLATE_SFPU shown separately). UNSTAMPED CSVs collapse "
    "trisc_id 2 (PACK) + 3 (SFPU) onto TRISC_2 and fall back to max-over-group (busy PACK sibling).",
    "Raw per-role cycles are stall-contaminated (include cross-thread/DFB blocking). The cycles/tile "
    "SLOPE is the trustworthy signal, not the largest single raw number.",
    "Timestamps are per-core rdcycle (no global clock); only same-core durations are valid — never "
    "compare start times across cores.",
]


def print_caveats():
    print("\nCAVEATS:")
    for c in CAVEATS:
        print(f"  - {c}")


def cmd_single(args):
    arch, freq, max_cores, df = read_device_csv(args.csv)
    rows = aggregate_roles(zone_durations(df))
    print(f"=== Quasar add profile (single run) ===")
    print(f"file: {args.csv}")
    print(f"arch={arch}  freq(MHz)={freq}  max_compute_cores={max_cores}  (durations in CYCLES)\n")
    print(f"{'RISC':<9} {'role':<22} {'category':<9} {'zone':<14} {'cycles':>8} {'n':>3}")
    print("-" * 72)
    for r in rows:
        print(
            f"{r['risc']:<9} {r['role']:<22} {r['category']:<9} {r['zone']:<14} "
            f"{r['dur_max']:>8} {r['instances']:>3}"
        )
    if args.json:
        print("\nJSON:")
        print(json.dumps({"arch": arch, "freq": freq, "rows": rows}, indent=2))
    print_caveats()
    return 0


def parse_points(point_strs):
    """["1:path", "4:path", ...] -> sorted [(tiles:int, path:str), ...]. Raises ValueError on bad form."""
    points = []
    for p in point_strs:
        tiles_str, _, path = p.partition(":")
        if not path:
            raise ValueError(f"malformed point '{p}', expected <tiles>:<csv>")
        points.append((int(tiles_str), path))
    points.sort()
    return points


def slopes_from_points(points):
    """
    Given [(tiles, csv_path), ...], return (arch, freq, series, fits) where:
      series[risc] = [(tiles, kernel_cycles), ...]
      fits[risc]   = {slope, intercept, r2} or None.
    """
    series = defaultdict(list)
    arch = freq = None
    for tiles, path in points:
        a, f, _mc, df = read_device_csv(path)
        arch, freq = a, f
        for risc, kc in kernel_dur_by_risc(aggregate_roles(zone_durations(df))).items():
            series[risc].append((tiles, kc))
    fits = {}
    for risc, pts in series.items():
        pts.sort()
        fits[risc] = fit_line([t for t, _ in pts], [c for _, c in pts])
    return arch, freq, series, fits


def cmd_sweep(args):
    try:
        points = parse_points(args.points)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    arch, freq, series, _fits = slopes_from_points(points)

    print("=== Quasar add profile (tile-count sweep) ===")
    print(f"arch={arch}  freq(MHz)={freq}  points={[t for t, _ in points]}  (CYCLES)\n")

    # Per-role KERNEL cycles table.
    all_riscs = sorted(series.keys())
    tile_vals = [t for t, _ in points]
    header = (
        f"{'RISC':<9} {'role':<22} {'category':<9}"
        + "".join(f"{('t=' + str(t)):>9}" for t in tile_vals)
        + f"{'slope':>10}{'intcpt':>9}{'R2':>7}"
    )
    print(header)
    print("-" * len(header))
    fits = {}
    for risc in all_riscs:
        role, cat = ROLE_MAP.get(risc, (risc, "OTHER"))
        by_tiles = dict(series[risc])
        xs = [t for t in tile_vals if t in by_tiles]
        ys = [by_tiles[t] for t in xs]
        fit = fit_line(xs, ys)
        fits[risc] = fit
        cells = "".join(f"{by_tiles.get(t, '-'):>9}" for t in tile_vals)
        if fit:
            print(f"{risc:<9} {role:<22} {cat:<9}{cells}{fit['slope']:>10.1f}{fit['intercept']:>9.0f}{fit['r2']:>7.3f}")
        else:
            print(f"{risc:<9} {role:<22} {cat:<9}{cells}{'n/a':>10}{'n/a':>9}{'n/a':>7}")

    # Verdict: largest KERNEL slope among compute roles.
    verdict = compute_verdict(fits)
    print("\n--- VERDICT ---")
    if verdict["bottleneck_risc"] is None:
        print("  Insufficient data for a slope-based verdict (need >=2 tile points with kernel zones).")
    else:
        b = verdict
        print(
            f"  Bottleneck role : {b['bottleneck_risc']} = {b['bottleneck_role']} "
            f"({b['bottleneck_slope']:.1f} cyc/tile, R²={b['bottleneck_r2']:.3f})"
        )
        print(f"  Bound           : {b['bound']}")
        print(
            f"  Overlap check   : pipeline≈{b['overlap']}  "
            f"(sum-of-role-slopes={b['sum_slopes']:.1f}, max-role-slope={b['max_slope']:.1f})"
        )
        print(f"  Interpretation  : {b['interpretation']}")

    if args.json:
        print("\nJSON:")
        print(json.dumps({"arch": arch, "freq": freq, "fits": fits, "verdict": verdict}, indent=2))
    print_caveats()
    return 0


def compute_verdict(fits):
    """From per-risc line fits, pick the bottleneck KERNEL role by slope and classify the bound."""
    cat_of = {risc: ROLE_MAP.get(risc, (risc, "OTHER"))[1] for risc in fits}
    role_of = {risc: ROLE_MAP.get(risc, (risc, "OTHER"))[0] for risc in fits}
    slopes = {risc: fit["slope"] for risc, fit in fits.items() if fit and fit["slope"] > 0}
    if not slopes:
        return {"bottleneck_risc": None}
    bottleneck = max(slopes, key=slopes.get)
    cat = cat_of[bottleneck]
    bound = {
        "COMPUTE": "COMPUTE-BOUND (FPU math is the per-tile rate limiter)",
        "MEMORY": "MEMORY-BOUND (L1<->register data movement is the per-tile rate limiter)",
        "DATAFLOW": "DATAFLOW-BOUND (NoC/DFB credit movement is the per-tile rate limiter)",
    }.get(cat, f"{cat}-BOUND")
    max_slope = slopes[bottleneck]
    sum_slopes = sum(slopes.values())
    # Overlap heuristic: if the max slope dominates the sum, stages overlap (true pipeline).
    overlap = "OVERLAPPED" if max_slope >= 0.6 * sum_slopes else "SERIALIZED"
    if overlap == "SERIALIZED":
        interp = (
            "Per-role slopes add up close to the total — stages are NOT overlapping well "
            "(pipeline stall). Fix the bubble before trusting the memory-vs-compute label."
        )
    else:
        interp = (
            "One role's slope dominates — stages overlap, so the bottleneck role genuinely sets "
            "the per-tile cost. Optimize that role."
        )
    return {
        "bottleneck_risc": bottleneck,
        "bottleneck_role": role_of[bottleneck],
        "bottleneck_slope": max_slope,
        "bottleneck_r2": fits[bottleneck]["r2"],
        "bound": bound,
        "overlap": overlap,
        "sum_slopes": sum_slopes,
        "max_slope": max_slope,
        "interpretation": interp,
    }


# Width-sweep classification thresholds: ratio of bf16 slope to bf8_b slope per role.
# bf16 moves ~2x the bytes of bf8_b for the same FLOPs. A role whose slope ~doubles tracks bytes
# (memory); a role whose slope stays ~flat tracks compute.
WIDTH_MEMORY_MIN = 1.5  # ratio >= this => byte-width-sensitive => MEMORY
WIDTH_COMPUTE_MAX = 1.2  # ratio <= this => byte-width-insensitive => COMPUTE
BYTE_RATIO = 2.0  # bf16 / bf8_b nominal bytes-per-element


def classify_width_ratio(ratio):
    if ratio is None:
        return "n/a", "insufficient data"
    if ratio >= WIDTH_MEMORY_MIN:
        return "MEMORY", f"slope scales with bytes (x{ratio:.2f} ~ byte ratio {BYTE_RATIO:.0f}) => data movement"
    if ratio <= WIDTH_COMPUTE_MAX:
        return "COMPUTE", f"slope ~flat across widths (x{ratio:.2f}) => FPU/compute, not bytes"
    return "MIXED", f"slope partly scales (x{ratio:.2f}) => between compute and memory"


def cmd_width(args):
    """
    Compare per-role cycles/tile slopes between two element widths (bf16 vs bf8_b). Same FLOPs/tile,
    ~2x bytes in bf16. Per role: ratio = slope_bf16 / slope_bf8b. ~2x => memory-bound; ~1x => compute.
    """
    try:
        pts16 = parse_points(args.bf16)
        pts8 = parse_points(args.bf8b)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    arch16, freq, _s16, fits16 = slopes_from_points(pts16)
    arch8, _f8, _s8, fits8 = slopes_from_points(pts8)

    print("=== Quasar add profile (element-WIDTH sweep: bf16 vs bf8_b) ===")
    print(f"arch={arch16}  freq(MHz)={freq}  (CYCLES; same FLOPs/tile, bf16≈2x bytes of bf8_b)\n")

    riscs = sorted(set(fits16) | set(fits8))
    header = f"{'RISC':<9} {'role':<22} {'category':<9}" f"{'bf16 s':>9}{'bf8b s':>9}{'ratio':>8}  {'reading':<10}"
    print(header)
    print("-" * len(header))
    rows = []
    for risc in riscs:
        role, cat = ROLE_MAP.get(risc, (risc, "OTHER"))
        f16 = fits16.get(risc)
        f8 = fits8.get(risc)
        s16 = f16["slope"] if f16 else None
        s8 = f8["slope"] if f8 else None
        ratio = (s16 / s8) if (s16 and s8 and s8 > 0) else None
        reading, _why = classify_width_ratio(ratio)
        rows.append(
            {
                "risc": risc,
                "role": role,
                "category": cat,
                "slope_bf16": s16,
                "slope_bf8b": s8,
                "ratio": ratio,
                "reading": reading,
            }
        )
        s16s = f"{s16:>9.1f}" if s16 is not None else f"{'n/a':>9}"
        s8s = f"{s8:>9.1f}" if s8 is not None else f"{'n/a':>9}"
        rs = f"{ratio:>8.2f}" if ratio is not None else f"{'n/a':>8}"
        print(f"{risc:<9} {role:<22} {cat:<9}{s16s}{s8s}{rs}  {reading:<10}")

    # Verdict: focus on the bottleneck role (largest bf16 slope) and read its width sensitivity.
    bott = max((r for r in rows if r["slope_bf16"]), key=lambda r: r["slope_bf16"], default=None)
    print("\n--- WIDTH VERDICT ---")
    if bott is None or bott["ratio"] is None:
        print("  Insufficient data (need >=2 tile points per dtype, with kernel zones, for both widths).")
    else:
        reading, why = classify_width_ratio(bott["ratio"])
        print(f"  Bottleneck role : {bott['risc']} = {bott['role']} (bf16 {bott['slope_bf16']:.1f} cyc/tile)")
        print(f"  Width ratio     : {bott['ratio']:.2f}  ({reading})")
        print(f"  Reading         : {why}")
        if reading == "MEMORY":
            print(
                "  => MEMORY-BOUND: the per-tile cost moves with bytes. Optimize the data path "
                "(pack/unpack throughput, L1 layout), not the FPU."
            )
        elif reading == "COMPUTE":
            print(
                "  => COMPUTE-BOUND: the per-tile cost is byte-insensitive. The FPU/math (or a "
                "fixed per-tile stall) is the wall, not data movement."
            )
        else:
            print(
                "  => MIXED: partial byte sensitivity — both data movement and a fixed/compute "
                "term matter. Split the contaminated region to separate them."
            )
        print(
            "  NOTE: this width test is the controlled discriminator — it holds FLOPs fixed and "
            "varies only bytes, so it is not fooled by the cross-thread stall contamination that "
            "makes the single-width slope ambiguous."
        )

    if args.json:
        print("\nJSON:")
        print(json.dumps({"arch": arch16, "rows": rows}, indent=2))
    print_caveats()
    return 0


# ---------------------------------------------------------------------------
# Roofline mode.
# ---------------------------------------------------------------------------
TILE_ELEMS = 1024  # 32x32 tile
ELEM_BYTES = {"bf16": 2, "bf8b": 1, "fp32": 4}  # bytes/element by dtype label
# An elementwise add does 1 add per output element, reading 2 inputs + writing 1 output.
ADD_FLOPS_PER_ELEM = 1
ADD_TENSORS_TOUCHED = 3  # a + b -> out (all resident in L1 for the sharded fast path)

# Placeholder Quasar ceilings (NOT measured on Quasar — see is_supported_quasar; repo has no Quasar
# eltwise FPU peak or L1 bandwidth). Defaults chosen to be GENEROUS to compute so the memory-bound
# verdict is conservative. Override with --peak-flops-per-cycle / --peak-bytes-per-cycle.
#   Compute: add's num_tiles_per_cycle=min(8,..) x 1024 elem/tile = 8192 add/cycle if the FPU retired
#            a full DST batch every cycle (an optimistic upper bound, labeled ASSUMED).
#   Memory:  WH L1 bisection is 512 B/cycle (operation.cpp:29); used as a placeholder for Quasar L1.
DEFAULT_PEAK_FLOPS_PER_CYCLE = 8192.0
DEFAULT_PEAK_BYTES_PER_CYCLE = 512.0

ROOFLINE_CAVEATS = [
    "Ceilings (peak FLOP/cycle, peak byte/cycle) are ASSUMED placeholders — Quasar's eltwise FPU peak "
    "and L1 bandwidth are not in the repo. Re-run with --peak-* once real Quasar numbers exist.",
    "FLOPs and bytes are ANALYTIC (1 add/elem; 2 inputs + 1 output resident in L1). DRAM is not "
    "modeled (sharded inputs are already L1-resident), so this is an L1<->register roofline.",
    "Achieved rate uses the critical-role cycles/tile slope, which is stall-contaminated, so the "
    "achieved FLOP/cycle & byte/cycle UNDERSTATE the kernel's clean capability (lower bound).",
]


def cmd_roofline(args):
    try:
        points = parse_points(args.points)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if args.dtype not in ELEM_BYTES:
        print(f"ERROR: --dtype must be one of {list(ELEM_BYTES)}", file=sys.stderr)
        return 2

    elem_bytes = ELEM_BYTES[args.dtype]
    flops_per_tile = TILE_ELEMS * ADD_FLOPS_PER_ELEM
    bytes_per_tile = ADD_TENSORS_TOUCHED * TILE_ELEMS * elem_bytes
    ai = flops_per_tile / bytes_per_tile  # FLOP/byte — exact, dtype-derived

    peak_flops = args.peak_flops_per_cycle
    peak_bytes = args.peak_bytes_per_cycle
    ridge = peak_flops / peak_bytes  # AI at which compute ceiling meets memory ceiling

    # Measured: critical per-tile cost = the bottleneck KERNEL role's cycles/tile slope.
    arch, freq, _series, fits = slopes_from_points(points)
    verdict = compute_verdict(fits)
    crit = verdict.get("bottleneck_slope") if verdict.get("bottleneck_risc") else None

    print("=== Quasar add ROOFLINE ===")
    print(f"arch={arch}  freq(MHz)={freq}  dtype={args.dtype}  (per-CYCLE units; freq=0 so never ns)\n")

    print("Operational intensity (exact, from shape + dtype):")
    print(f"  FLOPs/tile        = {flops_per_tile}        (1 add per 32x32=1024 elements)")
    print(f"  bytes/tile        = {bytes_per_tile}        (2 inputs + 1 output x 1024 x {elem_bytes} B)")
    print(f"  arithmetic int.   = {ai:.4f} FLOP/byte\n")

    print("Ceilings (ASSUMED placeholders — not Quasar-measured):")
    print(f"  peak compute      = {peak_flops:.1f} FLOP/cycle")
    print(f"  peak memory       = {peak_bytes:.1f} byte/cycle")
    print(f"  ridge intensity   = {ridge:.4f} FLOP/byte  (compute meets memory here)\n")

    # Roofline placement (intensity vs ridge).
    bound = "MEMORY" if ai < ridge else "COMPUTE"
    attainable_flops_per_cycle = min(peak_flops, ai * peak_bytes)  # roofline value at this AI

    print("Placement:")
    print(f"  op AI {ai:.4f}  {'<' if ai < ridge else '>='}  ridge {ridge:.4f}  =>  {bound}-BOUND")
    print(f"  attainable        = {attainable_flops_per_cycle:.2f} FLOP/cycle " f"(roofline value at this intensity)")

    if crit:
        achieved_flops_per_cycle = flops_per_tile / crit
        achieved_bytes_per_cycle = bytes_per_tile / crit
        util_mem = achieved_bytes_per_cycle / peak_bytes
        util_cmp = achieved_flops_per_cycle / peak_flops
        print(
            f"\nMeasured (critical role {verdict['bottleneck_risc']} = {verdict['bottleneck_role']}, "
            f"{crit:.1f} cyc/tile):"
        )
        print(
            f"  achieved compute  = {achieved_flops_per_cycle:.2f} FLOP/cycle  "
            f"({util_cmp*100:.1f}% of peak compute)"
        )
        print(
            f"  achieved memory   = {achieved_bytes_per_cycle:.2f} byte/cycle  " f"({util_mem*100:.1f}% of peak memory)"
        )
    else:
        achieved_flops_per_cycle = achieved_bytes_per_cycle = None
        print("\nMeasured: insufficient tile points for a critical slope (need >=2).")

    # Robustness: the verdict is invariant for any ridge above the op AI.
    print("\n--- ROOFLINE VERDICT ---")
    print(f"  {bound}-BOUND.  The add's arithmetic intensity is {ai:.3f} FLOP/byte.")
    print(
        f"  This stays MEMORY-bound for ANY ridge > {ai:.3f} FLOP/byte — i.e. any machine whose "
        f"peak-FLOP/cycle exceeds {ai:.3f}x its peak-byte/cycle."
    )
    print(
        f"  Even a 1:1 (ridge=1.0) machine is {1.0/ai:.1f}x past that, so the verdict does NOT depend "
        f"on the assumed Quasar ceilings — only an absurd ridge < {ai:.3f} would flip it."
    )
    if bound == "MEMORY":
        print(
            "  => Optimize data movement (pack/unpack throughput, L1 layout, fewer passes). The FPU "
            "is not the wall; an elementwise add is structurally memory-bound by its intensity."
        )

    if args.json:
        out = {
            "arch": arch,
            "dtype": args.dtype,
            "flops_per_tile": flops_per_tile,
            "bytes_per_tile": bytes_per_tile,
            "arithmetic_intensity": ai,
            "peak_flops_per_cycle": peak_flops,
            "peak_bytes_per_cycle": peak_bytes,
            "ridge": ridge,
            "bound": bound,
            "critical_cyc_per_tile": crit,
            "achieved_flops_per_cycle": achieved_flops_per_cycle,
            "achieved_bytes_per_cycle": achieved_bytes_per_cycle,
        }
        print("\nJSON:")
        print(json.dumps(out, indent=2))

    print("\nCAVEATS:")
    for c in ROOFLINE_CAVEATS + CAVEATS:
        print(f"  - {c}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Analyze Quasar add device-profiler CSV (cycles only).")
    sub = ap.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("single", help="per-role durations for one CSV")
    s.add_argument("csv")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_single)

    w = sub.add_parser("sweep", help="per-role cycles/tile slopes across <tiles>:<csv> points")
    w.add_argument("points", nargs="+", help="e.g. 1:sweep_1.csv 4:sweep_4.csv 16:sweep_16.csv")
    w.add_argument("--json", action="store_true")
    w.set_defaults(func=cmd_sweep)

    wd = sub.add_parser("width", help="memory-vs-compute via byte-width: compare bf16 vs bf8_b per-role slopes")
    wd.add_argument(
        "--bf16",
        nargs="+",
        required=True,
        help="<tiles>:<csv> points for bfloat16, e.g. 1:w_bf16_1.csv 16:w_bf16_16.csv",
    )
    wd.add_argument(
        "--bf8b",
        nargs="+",
        required=True,
        help="<tiles>:<csv> points for bfloat8_b, e.g. 1:w_bf8b_1.csv 16:w_bf8b_16.csv "
        "(NOTE: bf8_b is NOT a supported Quasar tile format — is_supported_quasar "
        "lacks Bfp8_b — so this mode cannot run on Quasar today; kept for when a "
        "second supported sub-2-byte width exists)",
    )
    wd.add_argument("--json", action="store_true")
    wd.set_defaults(func=cmd_width)

    rf = sub.add_parser("roofline", help="place the add against compute/memory ceilings via its arithmetic intensity")
    rf.add_argument("points", nargs="+", help="<tiles>:<csv> points, e.g. 1:sweep_1.csv 16:sweep_16.csv")
    rf.add_argument(
        "--dtype", default="bf16", choices=list(ELEM_BYTES), help="element dtype for byte/AI math (default bf16)"
    )
    rf.add_argument(
        "--peak-flops-per-cycle",
        type=float,
        default=DEFAULT_PEAK_FLOPS_PER_CYCLE,
        help=f"ASSUMED Quasar compute ceiling (default {DEFAULT_PEAK_FLOPS_PER_CYCLE})",
    )
    rf.add_argument(
        "--peak-bytes-per-cycle",
        type=float,
        default=DEFAULT_PEAK_BYTES_PER_CYCLE,
        help=f"ASSUMED Quasar L1 ceiling (default {DEFAULT_PEAK_BYTES_PER_CYCLE})",
    )
    rf.add_argument("--json", action="store_true")
    rf.set_defaults(func=cmd_roofline)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
