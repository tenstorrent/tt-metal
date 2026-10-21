#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""#58718 check: every counter bank of every core reports one ref cnt per op, i.e. it was frozen through the readout.

Reads the device log of the last tracy capture (or the path given) and fails if any (op, core, bank) has more than one
ref cnt among its records, or if the capture holds no counter records at all.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tt_metal" / "tt-llk" / "tools" / "python"))
from tt_llk_perf.headers import bank_tables  # noqa: E402

log = Path(sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv")
with log.open() as f:
    arch_line = f.readline().strip()
    rows = list(csv.DictReader(f, skipinitialspace=True))
print(arch_line)
arch = arch_line.split(",")[0].split(":")[1].strip()
# Every counter of a bank shares the bank's window; on tt-1xx each L1 mux group counts in a window of its own.
bank_of = {
    e.name: f"L1_{e.l1_mux}" if bank == "L1" else bank for bank, entries in bank_tables(arch).items() for e in entries
}

refs = defaultdict(set)
records = defaultdict(int)
for r in rows:
    r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
    if r.get("timer_id") != "9090" or not r.get("meta data"):
        continue
    md = json.loads(r["meta data"].replace(";", ",").replace("'", '"'))
    bank = bank_of.get(str(md["counter type"]), "UNKNOWN")
    op = (r["run host ID"], r["trace id counter"])
    core = (r["core_x"], r["core_y"], r["RISC processor type"])
    refs[(op, core, bank)].add(int(md["ref cnt"]))
    records[(op, core)] += 1

if not records:
    sys.exit(f"FAIL: {log} holds no counter records (timer_id 9090)")
ops = {op for op, _ in records}
cores = {core for _, core in records}
print(f"{sum(records.values())} counter records, {len(ops)} ops, {len(cores)} cores, {len(refs)} (op, core, bank) groups")
by_bank = defaultdict(lambda: [0, 0])
for (op, core, bank), values in sorted(refs.items()):
    by_bank[bank][0] += 1
    by_bank[bank][1] += len(values) > 1
for bank, (groups, bad) in sorted(by_bank.items()):
    print(f"  {bank:14s} {groups} groups, {bad} with more than one ref cnt")
bad = {k: v for k, v in refs.items() if len(v) > 1}
for (op, core, bank), values in list(sorted(bad.items()))[:20]:
    print(f"  op {op} core {core} {bank}: {len(values)} ref cnts {sorted(values)[:6]}")
if bad:
    sys.exit(f"FAIL: {len(bad)} of {len(refs)} (op, core, bank) groups kept counting through the readout")
print("PASS: every bank of every core reports one ref cnt per op")
