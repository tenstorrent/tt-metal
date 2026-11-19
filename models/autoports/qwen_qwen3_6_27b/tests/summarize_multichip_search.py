# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Read candidate results without sending large JSON records to the console."""

import argparse
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--prefix", default="family")
a = p.parse_args()
d = Path(__file__).resolve().parents[1] / "doc/optimized_multichip_decoder"
rows = []
for path in sorted(d.glob(a.prefix + "*.json")):
    r = json.loads(path.read_text())
    if not isinstance(r, dict) or "selected_policy" not in r:
        continue
    vals = [r[k] for k in ("prefill", "decode") if k in r]
    vals += list(r.get("contribution", {}).values())
    vals += [v for values in r.get("per_user_pcc", {}).values() for v in values]
    vals += [v for values in r.get("state", {}).values() for v in values]
    row = dict(
        candidate=path.stem,
        passed=r.get("pass"),
        minimum_pcc=min(vals) if vals else None,
        prefill_ms=r.get("timing", {}).get("prefill_ms"),
        decode_ms=r.get("timing", {}).get("decode_trace_ms"),
    )
    rows.append(row)
    print(row)
if rows:
    with (d / (a.prefix + "_summary.csv")).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
