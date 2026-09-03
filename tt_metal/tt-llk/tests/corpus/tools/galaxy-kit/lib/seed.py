#!/usr/bin/env python3
"""galaxy-kit queue seeder.  Each (op,leg) row gets LK_COPIES queue items;
workers put each copy on a DISTINCT chip.  --ops restricts to a subset.
Idempotent (existing items kept)."""
import argparse
import csv
import os
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--copies", type=int, default=int(os.environ.get("LK_COPIES", "4")))
ap.add_argument(
    "--special-copies", type=int, default=int(os.environ.get("LK_SPECIAL_COPIES", "8"))
)
ap.add_argument("--ops", default="", help="comma list filter (default all)")
a = ap.parse_args()

BASE = Path(os.environ.get("LK_BASE", "/data/nkapre/craq-laneLK"))
only = {t for t in re.split(r"[\s,]+", a.ops) if t} or None

qdir = BASE / "queue"
qdir.mkdir(exist_ok=True)
pairs = []
with (BASE / "ARMS.tsv").open() as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if only and r["op"] not in only:
            continue
        if (r["op"], r["leg"]) not in pairs:
            pairs.append((r["op"], r["leg"]))
n = 0
for op, leg in pairs:
    copies = a.copies if leg == "plain" else a.special_copies
    for k in range(copies):
        item = qdir / f"{op}__{leg}__c{k}"
        if not item.exists():
            item.mkdir()
            n += 1
print(f"seeded {n} new items ({len(pairs)} rows)")
