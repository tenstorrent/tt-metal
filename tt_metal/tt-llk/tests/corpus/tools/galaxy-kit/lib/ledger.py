#!/usr/bin/env python3
"""galaxy-kit ledger emitter — build the self-describing replication TSVs
from a pulled results/ tree.

  REPLICATION-LEDGER.tsv    one line per (op, leg, chip, arm, rep)
  REPLICATION-PAIRS.tsv     one line per (op, leg, chip): same-chip pair
  REPLICATION-VERDICTS.tsv  one line per (op, leg): class + board match

Cycle extraction mirrors sweep_2x2.py _perf_value/_kernel_value: within one
post CSV, sum mean(<metric>) over rows whose marker matches; the KERNEL
cell is absolute (the verdict metric); the diagnostic cell divides by
tile_cnt when per_tile=1.  Solo perf sessions make attribution trivial;
anomalies (MULTICSV, NO-KERNEL-ROW) are flagged, never silently booked.
Band (sweep_2x2.py): WIN < -0.5 <= PARITY <= +0.5 < LOSS.
"""
import argparse
import csv
import math
import statistics
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True, help="workdir with ROWS.tsv/ARMS.tsv")
ap.add_argument("--results", required=True, help="pulled results/ tree")
ap.add_argument("--out", default="", help="output dir (default: workdir)")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument(
    "--headline", default="", help="free-text provenance line for the ledger header"
)
a = ap.parse_args()

WORK = Path(a.work)
RES = Path(a.results)
OUT = Path(a.out) if a.out else WORK
OUT.mkdir(parents=True, exist_ok=True)

rows_meta = {}
with (WORK / "ROWS.tsv").open() as f:
    for r in csv.DictReader(f, delimiter="\t"):
        rows_meta[(r["op"], r["leg"])] = r
arms_meta = {}
with (WORK / "ARMS.tsv").open() as f:
    for r in csv.DictReader(f, delimiter="\t"):
        arms_meta[(r["op"], r["leg"], r["arm"])] = r


def cells_of(sdir, marker, metric, per_tile):
    col = f"mean({metric})"
    kvals, dvals, matched = [], [], 0
    for post in sorted(sdir.glob("perf_data/*/*.post.csv")):
        ktot, dtot, tiles, kseen, dseen = 0.0, 0.0, 1.0, False, False
        with post.open() as f:
            for rec in csv.DictReader(f):
                if col not in rec:
                    continue
                if rec.get("marker") == "KERNEL":
                    ktot += float(rec[col])
                    kseen = True
                if rec.get("marker") == marker:
                    dtot += float(rec[col])
                    if not dseen:
                        try:
                            tiles = float(rec.get("tile_cnt", 1) or 1)
                        except ValueError:
                            tiles = 1.0
                    dseen = True
        if kseen or dseen:
            matched += 1
            if kseen:
                kvals.append(ktot)
            if dseen:
                dvals.append(dtot / (tiles or 1.0) if per_tile else dtot)
    note = ""
    if matched > 1:
        note = f"MULTICSV:{matched}"
    if not kvals:
        return None, (dvals[0] if dvals else None), note or "NO-KERNEL-ROW"
    return kvals[0], (dvals[0] if dvals else None), note


led = (OUT / "REPLICATION-LEDGER.tsv").open("w")
led.write(
    "# galaxy-kit replication ledger — CRAQ-SFPI sem vs expert hand arms, "
    "same-chip pairs, execute-only consumer runs of quietbox-compiled ELFs.\n"
    f"# provenance: {a.headline or 'see kit README'}\n"
    "# GALAXY CYCLES ARE NOT p150-CANON: same-chip sem/hand RATIOS are the "
    "replication statistic; the p150 board stays canon.\n"
)
led.write(
    "op\tleg\tset\tboard_class\tchip\tarm\trep\trc\tkernel_cycles\t"
    "diag_cycles\tmarker\tmetric\tper_tile\tflagskey\tnote\n"
)
pairs = (OUT / "REPLICATION-PAIRS.tsv").open("w")
pairs.write(
    "op\tleg\tset\tboard_class\tchip\tsem_kernel_median\thand_kernel_median\t"
    "vs_hand_pct\tsem_reps\thand_reps\tsem_spread\thand_spread\tnote\n"
)
for opdir in sorted(RES.iterdir()):
    if not opdir.is_dir() or "__" not in opdir.name:
        continue
    op, leg = opdir.name.rsplit("__", 1)
    meta = rows_meta.get((op, leg))
    if meta is None:
        continue
    marker, metric = meta["marker"], meta["metric"]
    per_tile = meta["per_tile"] == "1"
    for chipdir in sorted(opdir.iterdir()):
        if not chipdir.is_dir():
            continue
        chip = chipdir.name
        kcells = {"sem": [], "hand": []}
        notes = []
        for arm in ("sem", "hand"):
            am = arms_meta.get((op, leg, arm))
            if am is None:
                continue
            fk = am["flagskey"]
            cd = chipdir / f"{arm}-corr"
            if (cd / "rc.txt").is_file():
                corr_rc = (cd / "rc.txt").read_text().strip()
                led.write(
                    f"{op}\t{leg}\t{meta['set']}\t{meta['board_class']}\t{chip}"
                    f"\t{arm}\tcorr\t{corr_rc}\t\t\t{marker}\t{metric}\t"
                    f"{meta['per_tile']}\t{fk}\t\n"
                )
            for k in range(1, a.reps + 1):
                sdir = chipdir / f"{arm}-perf-r{k}"
                if not (sdir / "rc.txt").is_file():
                    continue
                rc = (sdir / "rc.txt").read_text().strip()
                kc, dc, note = cells_of(sdir, marker, metric, per_tile)
                if rc == "0" and kc is not None:
                    kcells[arm].append(kc)
                if note:
                    notes.append(f"{arm}-r{k}:{note}")
                led.write(
                    f"{op}\t{leg}\t{meta['set']}\t{meta['board_class']}\t{chip}"
                    f"\t{arm}\tr{k}\t{rc}\t{'' if kc is None else kc}\t"
                    f"{'' if dc is None else dc}\t{marker}\t{metric}\t"
                    f"{meta['per_tile']}\t{fk}\t{note}\n"
                )
        s, h = kcells["sem"], kcells["hand"]
        if s and h:
            sm, hm = statistics.median(s), statistics.median(h)
            vs = (sm - hm) / hm * 100.0 if hm else math.nan
            pairs.write(
                f"{op}\t{leg}\t{meta['set']}\t{meta['board_class']}\t{chip}\t"
                f"{sm}\t{hm}\t{vs:+.2f}\t{len(s)}\t{len(h)}\t"
                f"{max(s)-min(s)}\t{max(h)-min(h)}\t{';'.join(notes)}\n"
            )
led.close()
pairs.close()


def band(v):
    return "WIN" if v < -0.5 else ("PARITY" if v <= 0.5 else "LOSS")


per_op = {}
with (OUT / "REPLICATION-PAIRS.tsv").open() as f:
    for r in csv.DictReader(f, delimiter="\t"):
        per_op.setdefault((r["op"], r["leg"]), []).append(r)
tally = {"MATCH": 0, "MISMATCH": 0}
with (OUT / "REPLICATION-VERDICTS.tsv").open("w") as w:
    w.write(
        "op\tleg\tset\tboard_class\tchips\tvs_hand_median_pct\t"
        "vs_hand_min_pct\tvs_hand_max_pct\treplica_class\tmatch\tchips_list\n"
    )
    for (op, leg), cells in sorted(per_op.items()):
        vs = sorted(float(c["vs_hand_pct"]) for c in cells)
        med = statistics.median(vs)
        rc = band(med)
        bc = cells[0]["board_class"]
        match = (
            "MATCH"
            if rc == bc
            else ("MISMATCH" if bc in ("WIN", "PARITY", "LOSS") else "N/A")
        )
        if match in tally:
            tally[match] += 1
        w.write(
            f"{op}\t{leg}\t{cells[0]['set']}\t{bc}\t{len(cells)}\t{med:+.2f}\t"
            f"{vs[0]:+.2f}\t{vs[-1]:+.2f}\t{rc}\t{match}\t"
            f"{','.join(c['chip'] for c in cells)}\n"
        )
print(f"ledger written to {OUT} (verdict tally {tally})")
