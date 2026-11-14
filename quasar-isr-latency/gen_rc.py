#!/usr/bin/env python3
"""Generate a Verdi export signal list (.rc) for a dfb_isr_timing_bench capture.

Starts from isr_tc.rc (DM0 ISR signals, tile-counter bank 0, Neo0-3 handshakes on tile gen_y[1]/gen_x[0])
and adds what the bench's configurations need, read from its manifest (JSON lines):
  * the retire trace (io_trace_0_valid/iaddr) of every DM the configurations use (producer/consumer stall,
    launch windows);
  * the overlay tile-counter bank g_LLK_IF_GEN[k] for every Tensix id k whose counters the ISR credits;
  * the same set for every Tensix core in the configurations (one tile per core).

The logical-core -> RTL tile mapping is not derivable from the manifest; give it with --tile, e.g.
  --tile 0,0=1,0 --tile 1,0=1,1     (logical x,y = RTL gen_y,gen_x)
The default maps only logical (0,0) to gen_y[1]/gen_x[0], as isr_tc.rc does.

Usage:
  gen_rc.py manifest.jsonl -o capture.rc [--base isr_tc.rc] [--tile X,Y=GY,GX ...] [--all-dms] [--all-banks]
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TILE_RE = re.compile(r"gen_y\[\d+\]/gen_x\[\d+\]")
OW = "overlay_noc_wrap/overlay_noc_niu_router/neo_overlay_wrapper/overlay_wrapper"
DM_TRACE = OW + "/cpu_cluster/{dom}/element_reset_domain_rockettile/core/csr/io_trace_0_{sig}"
BANK_SIGS = [
    "i_local_wren", "i_local_addr", "i_local_wrdata", "tile_counters/o_tiles_available",
    "o_remote_rts", "i_remote_rtr", "o_remote_counter_sel", "o_remote_idx", "o_remote_incr",
]


def dm_domain(dm):
    return "tile_prci_domain" if dm == 0 else f"tile_prci_domain_{dm}"


def load_manifest(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--base", default=os.path.join(HERE, "isr_tc.rc"))
    ap.add_argument("--tile", action="append", default=[], metavar="X,Y=GY,GX")
    ap.add_argument("--all-dms", action="store_true", help="trace DM0-DM7 regardless of the manifest")
    ap.add_argument("--all-banks", action="store_true", help="export tile-counter banks 0-3 regardless")
    a = ap.parse_args()

    tiles = {(0, 0): (1, 0)}
    for t in a.tile:
        lhs, rhs = t.split("=")
        x, y = (int(v) for v in lhs.split(","))
        gy, gx = (int(v) for v in rhs.split(","))
        tiles[(x, y)] = (gy, gx)

    rows = load_manifest(a.manifest)
    cores, dms, banks = set(), {0}, {0}
    for r in rows:
        for c in r["cores"]:
            cores.add(tuple(c["logical"]))
            for d in c["dfbs"]:
                for rc in d["risc_tcs"]:
                    if rc["risc"] < 8:
                        dms.add(rc["risc"])
                    for tensix, _ctr in rc["tcs"]:
                        banks.add(tensix)
    if a.all_dms:
        dms = set(range(8))
    if a.all_banks:
        banks = {0, 1, 2, 3}
    missing = sorted(c for c in cores if c not in tiles)
    if missing:
        sys.exit(f"no RTL tile for logical core(s) {missing}; pass --tile X,Y=GY,GX")

    base = open(a.base).read().splitlines()
    header = [l for l in base if not l.startswith("addSignal")]
    base_sigs = [l for l in base if l.startswith("addSignal")]
    template_tile = TILE_RE.search(base_sigs[0]).group(0)
    prefix = base_sigs[0].split(template_tile)[0] + "{tile}/gen_tensix/tt_tensix_with_l1/"

    out, seen = list(header), set()

    def add(line):
        if line not in seen:
            seen.add(line)
            out.append(line)

    for core in sorted(cores):
        gy, gx = tiles[core]
        tile = f"gen_y[{gy}]/gen_x[{gx}]"
        for l in base_sigs:
            add(l.replace(template_tile, tile))
        p = prefix.format(tile=tile)
        for dm in sorted(dms):
            for sig in ("valid", "iaddr"):
                add("addSignal -h 15 " + p + DM_TRACE.format(dom=dm_domain(dm), sig=sig))
        for k in sorted(banks):
            for s in BANK_SIGS:
                add("addSignal -h 15 " + p + f"{OW}/g_LLK_IF_GEN[{k}]/overlay_tile_counters_with_comparators/{s}")

    with open(a.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {a.out}: {len(out) - len(header)} signals, cores {sorted(cores)}, DMs {sorted(dms)}, banks {sorted(banks)}")


if __name__ == "__main__":
    main()
