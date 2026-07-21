#!/usr/bin/env python3
# Parse the device-profiler CSV for the DIAG_ZONES causal-timing decomposition (regime_a_matmul).
# Captures ALL named zones (the shipping -KERNEL whole-RISC zones AND the custom DeviceZoneScopedN zones),
# grouped per physical core (x,y,risc) and zone name, computing per-instance durations (ZONE_END-ZONE_START)
# in cycles. Reports per-zone distributions across cores (min/median/max) and per-core zone breakdowns so
# imbalance and role differences are visible — NOT just whole-RISC spans.
#
# The profiler CSV columns (from tt_metal): row[1..3] = core (x,y,risc-ish), row[5] = time[cycles], row[10]
# = zone name, row[11] = type (ZONE_START/ZONE_END). Rows 0-1 are headers. Run 0 (warmup) is dropped.
import csv, statistics
from collections import defaultdict


def parse_zones(csv_path, drop_warmup=True):
    rows = list(csv.reader(open(csv_path)))
    # (x, y, risc, zone) -> list[(type, cycle)]
    ev = defaultdict(list)
    for row in rows[2:]:
        if len(row) < 12:
            continue
        zone = row[10].strip()
        if not zone:
            continue
        ev[(row[1].strip(), row[2].strip(), row[3].strip(), zone)].append((row[11].strip(), int(row[5])))
    # durations per (core,zone): pair START->END in order
    dur = {}
    for k, lst in ev.items():
        ds, st = [], None
        for t, c in lst:
            if t == "ZONE_START":
                st = c
            elif t == "ZONE_END" and st is not None:
                ds.append(c - st)
                st = None
        if ds:
            dur[k] = ds[1:] if (drop_warmup and len(ds) > 1) else ds
    return dur  # {(x,y,risc,zone): [cycles per timed run]}


def summarize(csv_path, freq_hz=1.0e9):
    dur = parse_zones(csv_path)
    # per-zone across cores: median cycles per core, then distribution across cores
    byzone = defaultdict(list)
    for (x, y, r, zone), ds in dur.items():
        byzone[zone].append((f"{x},{y}", statistics.median(ds)))
    us = lambda c: c / freq_hz * 1e6
    out = {}
    for zone, pairs in byzone.items():
        vals = sorted(v for _, v in pairs)
        out[zone] = {
            "ncores": len(pairs),
            "min_us": round(us(vals[0]), 2),
            "med_us": round(us(statistics.median(vals)), 2),
            "max_us": round(us(vals[-1]), 2),
            "spread_pct": round((vals[-1] - vals[0]) / vals[0] * 100, 1) if vals[0] else None,
        }
    return out


def print_summary(csv_path, freq_hz=1.0e9, title=""):
    s = summarize(csv_path, freq_hz)
    # order: custom Z_* zones first (sorted), then -KERNEL whole-RISC zones
    zones = sorted(s, key=lambda z: (not z.startswith("Z_"), z))
    print(f"=== ZONE DECOMPOSITION {title} (us; med across cores [min..max], spread=core imbalance) ===")
    print(f"{'zone':22}{'ncores':>7}{'min':>8}{'med':>8}{'max':>8}{'spread%':>9}")
    for z in zones:
        d = s[z]
        print(f"{z:22}{d['ncores']:>7}{d['min_us']:>8}{d['med_us']:>8}{d['max_us']:>8}{str(d['spread_pct']):>9}")
    return s


def parse_raw(csv_path):
    """Per (core,zone) -> list of ABSOLUTE (start_cycle, end_cycle) per instance (all iters, warmup incl).
    Absolute times let overlap + the critical path be reconstructed (whole-phase duration != exposed wall)."""
    rows = list(csv.reader(open(csv_path)))
    ev = defaultdict(list)
    for row in rows[2:]:
        if len(row) < 12 or not row[10].strip():
            continue
        ev[(row[1].strip(), row[2].strip(), row[3].strip(), row[10].strip())].append((row[11].strip(), int(row[5])))
    raw = {}
    for k, lst in ev.items():
        pairs, st = [], None
        for t, c in lst:
            if t == "ZONE_START":
                st = c
            elif t == "ZONE_END" and st is not None:
                pairs.append((st, c))
                st = None
        if pairs:
            raw[k] = pairs
    return raw


def roles_and_timeline(csv_path, freq_hz=1.0e9):
    """Infer per-core role from zone presence/magnitude and build a per-role timeline. Roles: reader (runs
    Z_IN1READ), slave (no Z_IN1READ but has a KERNEL span), reduction root (high Z_PHASE2) vs leaf (low)."""
    raw = parse_raw(csv_path)
    us = lambda c: c / freq_hz * 1e6
    # collect per-core (x,y): which zones, med duration, min start, max end
    cores = defaultdict(dict)
    for (x, y, r, zone), pairs in raw.items():
        durs = sorted(e - s for s, e in pairs)[1:] or [pairs[-1][1] - pairs[-1][0]]  # drop warmup
        med = durs[len(durs) // 2]
        cores[(x, y)][zone] = {"med_us": round(us(med), 2), "risc": r,
                               "start": min(s for s, _ in pairs), "end": max(e for _, e in pairs)}
    return cores


if __name__ == "__main__":
    import sys, json

    csv_path = sys.argv[1]
    freq = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0e9
    print_summary(csv_path, freq)
    if len(sys.argv) > 3:  # dump raw absolute-timestamp artifact for auditability
        raw = {f"{x},{y},{r},{z}": v for (x, y, r, z), v in parse_raw(csv_path).items()}
        json.dump(raw, open(sys.argv[3], "w"))
        print(f"wrote raw zone artifact -> {sys.argv[3]} ({len(raw)} core-zone series)")
