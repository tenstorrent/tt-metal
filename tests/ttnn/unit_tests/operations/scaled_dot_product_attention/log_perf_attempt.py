#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Append one optimization attempt to agent_logs/perf_journal.jsonl.

Breadcrumb-style journal for the SDPA perf-juicing loop (see
OPTIMIZATION_PROTOCOL.md): one JSON line per attempt — hypothesis, the single
change, the keep/revert decision, and the measured numbers. Auto-attaches the
latest results parsed from generated/sdpa_juice_perf.txt (what the test writes),
so you don't retype them.

Usage:
    python3 log_perf_attempt.py \
        --hypothesis "d=512 c_kv=1 makes softmax fire per KV tile; raise c_kv" \
        --change "program_descriptor: set c_kv=4 (decouple from 16//Dt clamp)" \
        --decision keep \
        --files scaled_dot_product_attention_program_descriptor.py \
        --note "case4 ns 9.9e8 -> 4.1e8"
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]  # <repo>/tests/ttnn/unit_tests/operations/<op>/file -> <repo>
PERF_TXT = REPO / "generated" / "sdpa_juice_perf.txt"
JOURNAL = Path(__file__).resolve().parent / "agent_logs" / "perf_journal.jsonl"

# Each test line looks like:
# [favorable_h8_s4096_d64] cores=64/64 device_kernel_ns=12234200 achieved=2.81TFLOPs util=1.09% pcc=0.99999
_LINE = re.compile(
    r"\[(?P<case>[^\]]+)\].*?device_kernel_ns=(?P<ns>[\d.]+).*?util=(?P<util>[\d.]+)%.*?pcc=(?P<pcc>[-\d.]+)"
)


def _latest_results() -> dict:
    """Last measured value per case from generated/sdpa_juice_perf.txt."""
    out: dict[str, dict] = {}
    if PERF_TXT.exists():
        for line in PERF_TXT.read_text().splitlines():
            m = _LINE.search(line)
            if m:
                out[m["case"]] = {
                    "ns": float(m["ns"]),
                    "util": float(m["util"]),
                    "pcc": float(m["pcc"]),
                }
    return out


def _next_attempt() -> int:
    if not JOURNAL.exists():
        return 1
    n = 0
    for line in JOURNAL.read_text().splitlines():
        line = line.strip()
        if line:
            n += 1
    return n + 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hypothesis", required=True, help="The single falsifiable idea being tested")
    ap.add_argument("--change", required=True, help="The one change made")
    ap.add_argument("--decision", required=True, choices=["keep", "revert"], help="Outcome")
    ap.add_argument("--files", nargs="*", default=[], help="Files touched")
    ap.add_argument("--note", default="", help="Free-text note (numbers, why)")
    ap.add_argument("--attempt", type=int, default=None, help="Override attempt number")
    args = ap.parse_args()

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attempt": args.attempt if args.attempt is not None else _next_attempt(),
        "hypothesis": args.hypothesis,
        "change": args.change,
        "files": args.files,
        "results": _latest_results(),
        "decision": args.decision,
        "note": args.note,
    }
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"logged attempt #{entry['attempt']} ({args.decision}) -> {JOURNAL}")
    if entry["results"]:
        for case, r in entry["results"].items():
            print(f"  {case}: ns={r['ns']:.0f} util={r['util']:.2f}% pcc={r['pcc']:.5f}")
    else:
        print("  (no results in generated/sdpa_juice_perf.txt yet — run the test first)")


if __name__ == "__main__":
    main()
