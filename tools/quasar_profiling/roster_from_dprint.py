#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Per-cluster / per-engine roster + timing table for the Quasar add, parsed from the DPRINT log.

QUASAR COMPUTE HIERARCHY (source-verified, tt_metal/impl/metal2_host_api/temp_quasar_api.hpp:17-22,
kernel.hpp QuasarComputeProcessor enum, quasar_32_arch.yaml):
  chip
   └─ CLUSTER  (one per (core_x,core_y) coordinate; 32 per chip on the 8x4 grid; "users target
       │        clusters"; all engines + DM in a cluster share one 4 MB L1)
       ├─ 8 DM (Rocket data-movement) cores                       -> labeled DM0..DM7
       └─ 4 Tensix ENGINES, a.k.a. NEO_0..NEO_3                    -> the DPRINT "N<n>" prefix
            │  (neo_id CSR 0xBC2 selects WHICH of the 4 engines in the cluster — NOT which cluster)
            └─ 4 TRISC processors (trisc_id 0..3)                  -> the DPRINT "TR<n>" prefix
                  0=UNPACK, 1=MATH, 2=PACK, 3=ISOLATE_SFPU
So the DPRINT prefix "N1TR2" = Tensix engine NEO_1, TRISC 2 (PACK), within the cluster at (cx,cy).
A compute kernel's num_threads (1,2,4) = how many of the cluster's 4 engines it runs on.

WHY the DPRINT log (not the device-profiler CSV): the DPRINT prefix carries the full identity the CSV
loses — the NEO engine id and TRISC id (NxTRy) and the DM core index (DMn). The CSV only has
core_x,core_y plus a collapsed TRISC_0/1/2 label, with no engine field. So for a per-engine view the
DPRINT stream is the richer, correct source.

Signals consumed (all are firmware/kernel DPRINT lines, tagged by the dprint server's prefix
`device:cx-cy:risc_name`):
  * "<risc>: TRISC-FW: initialized" / "DM0-FW: initialized"  -> ROSTER (the core booted; present)
  * "<risc>: hartid: N"                                      -> identity confirm (TRISC only)
  * "<risc>: [kernel-timer] slot=S cycles=C"                 -> RAN an instrumented add kernel
        slot 0 = reader (DM), 1 = compute (TRISC), 2 = writer (DM)

A core that booted (init line) but emitted no kernel-timer line is reported as "no output" — NOT
"idle". We only know the timer printed nothing for it; the kernel timer instruments only the 3 add
kernels (reader/compute/writer), so a core doing uninstrumented or non-add work would also show "no
output". "no output" states the observation; it does not claim the core did nothing.

CAVEATS:
  * Cycles are per-core rdcycle deltas: valid as "how long that core's kernel ran". NO shared clock
    across cores — do not read the table as a synchronized timeline or infer concurrency.
  * Roster completeness depends on TT_METAL_DPRINT_CORES=all (else some cores never print).
  * Single sample (n=1); emulator is deterministic but one rep.
"""

import argparse
import re
import sys
from collections import defaultdict

# 0:0-0:N0TR2: [kernel-timer] slot=1 cycles=3668
# 0:0-0:DM3: DM0-FW: initialized
# NOTE: the prefix is matched ANYWHERE in the line, not anchored at ^, because other stdout (e.g.
# pytest's "PASSED") can be concatenated onto the start of a DPRINT line with no separating newline
# (seen: "PASSED0:1-0:DM3: [kernel-timer]..."). Anchoring at ^ silently dropped such lines.
PREFIX = re.compile(r"(\d+):(\d+)-(\d+):([A-Za-z0-9]+):\s*(.*)$")
TIMER = re.compile(r"\[kernel-timer\]\s+slot=(\d+)\s+cycles=(\d+)")
INIT = re.compile(r"(TRISC-FW|DM0-FW):\s*initialized")
HARTID = re.compile(r"hartid:\s*(\d+)")
NEOTR = re.compile(r"^N(\d+)TR(\d+)$")
DM = re.compile(r"^DM(\d+)$")

SLOT_ROLE = {0: "reader", 1: "compute", 2: "writer"}
TRISC_ROLE = {0: "UNPACK", 1: "MATH", 2: "PACK", 3: "ISOLATE_SFPU"}


def parse_log(path):
    """Return cores[(cx,cy)] = {risc_name: {booted, hartid, slot, cycles, kind, ...}}."""
    cores = defaultdict(lambda: defaultdict(dict))
    with open(path, errors="replace") as f:
        for line in f:
            m = PREFIX.search(line.rstrip("\n"))
            if not m:
                continue
            _dev, cx, cy, risc, rest = m.groups()
            cell = cores[(int(cx), int(cy))][risc]
            if INIT.search(rest):
                cell["booted"] = True
            elif h := HARTID.search(rest):
                cell["hartid"] = int(h.group(1))
            elif t := TIMER.search(rest):
                cell["slot"] = int(t.group(1))
                cell["cycles"] = int(t.group(2))
                cell["active"] = True
            else:
                # Any other DPRINT line from this core is positive evidence it executed kernel code
                # (e.g. a test kernel's own prints). We can't time it without a kernel-timer line, but
                # "active" is stronger than "booted, no output" — the core demonstrably ran.
                cell["active"] = True
    return cores


def classify(risc):
    """Return (subgroup, subsort, label, role) for a risc name, WITHIN a cluster. A NEO engine and a
    DM pool are sub-groups of the cluster. The 'N<n>' prefix is Tensix engine NEO_<n> (one of the
    cluster's 4 compute engines), 'TR<n>' is the TRISC role within it; 'DM<n>' is a DM core."""
    if m := NEOTR.match(risc):
        neo, tr = int(m.group(1)), int(m.group(2))
        return (f"NEO_{neo} engine", (0, neo), f"TR{tr}", TRISC_ROLE.get(tr, "?"))
    if m := DM.match(risc):
        idx = int(m.group(1))
        return ("DM cores", (1, 0), f"DM{idx}", "data-movement")
    return ("other", (2, 0), risc, "?")


def build_table(cores):
    """Flatten to rows grouped by cluster (core_x,core_y) -> NEO engine / DM subgroup. Each row:
    group, label, role, status, cycles, hartid."""
    groups = defaultdict(list)
    group_sort = {}
    for (cx, cy), riscs in cores.items():
        for risc, cell in riscs.items():
            sub, gs, label, role = classify(risc)
            # Top-level group = the cluster; the NEO engine / DM pool is a sub-group within it.
            grp = f"cluster ({cx},{cy})  /  {sub}"
            group_sort[grp] = (cx, cy, gs)
            timed = "cycles" in cell
            active = cell.get("active", False)
            if timed:
                status = "ran (timed)"
            elif active:
                status = "active (no timer)"
            elif cell.get("booted"):
                status = "booted, no output"
            else:
                status = "seen, no output"
            groups[grp].append(
                {
                    "label": label,
                    "role": role,
                    "status": status,
                    "cycles": cell.get("cycles"),
                    "slot_role": SLOT_ROLE.get(cell.get("slot")) if timed else None,
                    "hartid": cell.get("hartid"),
                    "sort": label,
                }
            )
    return groups, group_sort


def render(groups, group_sort):
    out = []
    out.append("=== Quasar roster: cluster -> NEO engine -> TRISC / DM (from DPRINT log) ===")
    out.append("hierarchy: cluster (one per (x,y) coord) = 8 DM + 4 Tensix engines (NEO_0..3), one 4MB L1;")
    out.append("           each NEO engine = 4 TRISC (TR0=UNPACK TR1=MATH TR2=PACK TR3=ISOLATE_SFPU).")
    out.append("status: 'ran (timed)' = kernel-timer line (cycles); 'active (no timer)' = printed other")
    out.append("        output so demonstrably executed, but no timing; 'booted, no output' = booted,")
    out.append("        emitted nothing further (NOT necessarily idle — may run uninstrumented work).")
    out.append("(cycles are per-core rdcycle deltas — NOT a shared timeline; do not infer concurrency)\n")
    n_timed = n_active = n_total = 0
    for grp in sorted(group_sort, key=lambda g: group_sort[g]):
        rows = sorted(groups[grp], key=lambda r: r["sort"])
        out.append(grp)
        out.append(f"  {'core':<6}{'role':<14}{'status':<20}{'cycles':>8}  {'kernel':<8}")
        out.append("  " + "-" * 56)
        for r in rows:
            n_total += 1
            cyc = str(r["cycles"]) if r["cycles"] is not None else "-"
            kern = r["slot_role"] or ""
            if r["status"] == "ran (timed)":
                n_timed += 1
            elif r["status"] == "active (no timer)":
                n_active += 1
            out.append(f"  {r['label']:<6}{r['role']:<14}{r['status']:<20}{cyc:>8}  {kern:<8}")
        out.append("")
    n_executed = n_timed + n_active
    out.append(
        f"summary: {n_executed} of {n_total} booted cores show execution "
        f"({n_timed} timed, {n_active} active-no-timer, {n_total - n_executed} no output)."
    )
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Per-NEO/per-core roster+timing from a Quasar DPRINT log.")
    ap.add_argument("log", help="path to the run log containing DPRINT lines (TT_METAL_DPRINT_CORES=all)")
    args = ap.parse_args()
    cores = parse_log(args.log)
    if not cores:
        print("No DPRINT core lines found. Was TT_METAL_DPRINT_CORES=all set and the kernel timer on?", file=sys.stderr)
        return 1
    groups, group_sort = build_table(cores)
    print(render(groups, group_sort))
    return 0


if __name__ == "__main__":
    sys.exit(main())
