# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Join a tracy capture of ``test_dflash_attn_matmul_sweep`` to its per-candidate signposts.

Each candidate in that sweep is bracketed by ``<case>__<cand>_start`` / ``_stop`` signposts, so a
candidate's cost is the sum of DEVICE KERNEL DURATION over the ops between its pair, divided by the
iteration count — no row counting, and a candidate that emits an extra reshard is charged for it.

Usage::

    python models/demos/blackhole/qwen36/tests/perf/dflash_mm_sweep_report.py \\
      generated/profiler/reports/<dir>/ops_perf_results_<dir>.csv [iters]
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict


def main(path: str, iters: int = 5) -> None:
    per_cand: dict[tuple[str, str], dict] = {}
    cur = None
    with open(path) as fh:
        for row in csv.DictReader(fh):
            code = (row.get("OP CODE") or "").strip()
            if (row.get("OP TYPE") or "").strip() == "signpost":
                if code.endswith("_start") and "__" in code:
                    case, cand = code[: -len("_start")].split("__", 1)
                    cur = (case, cand)
                    per_cand.setdefault(cur, {"ns": 0.0, "ops": 0, "codes": defaultdict(float), "devs": set()})
                elif code.endswith("_stop"):
                    cur = None
                continue
            if cur is None:
                continue
            try:
                ns = float(row["DEVICE KERNEL DURATION [ns]"])
            except (KeyError, TypeError, ValueError):
                continue
            e = per_cand[cur]
            e["ns"] += ns
            e["ops"] += 1
            e["codes"][code] += ns
            e["devs"].add((row.get("DEVICE ID") or "").strip())

    by_case: dict[str, list] = defaultdict(list)
    for (case, cand), e in per_cand.items():
        ndev = max(1, len(e["devs"]))
        # Every device runs the same replicated matmul, so per-device time is the wall cost.
        us = e["ns"] / 1e3 / iters / ndev
        by_case[case].append((us, cand, e["ops"] // (iters * ndev), dict(e["codes"])))

    for case, rows in by_case.items():
        rows.sort()
        base = next((us for us, cand, *_ in rows if cand == "auto"), None)
        print(f"\n=== {case} ===  (auto = {base:.1f} us)" if base else f"\n=== {case} ===")
        for us, cand, nops, codes in rows:
            delta = f"{100 * (us - base) / base:+6.1f} %" if base else "      -"
            extra = ""
            if nops > 1:
                parts = sorted(codes.items(), key=lambda kv: -kv[1])
                extra = "  [" + ", ".join(f"{k.replace('DeviceOperation','')} {v/1e3/5:.0f}us" for k, v in parts) + "]"
            print(f"  {us:7.1f} us  {delta}  {cand:40} ops/iter={nops}{extra}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
