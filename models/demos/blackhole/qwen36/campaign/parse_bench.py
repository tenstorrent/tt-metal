#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Turn a benchmark log into one-line summaries and append records to results.jsonl.

Usage:
    parse_bench.py LOG [--append /data/ayerofieiev/qwen38/results.jsonl]
                       [--ref SHA] [--node NAME]

Scans LOG for BENCH_JSON lines (emitted by bench_decode.py / bench_prefill.py),
prints a one-line summary per record, and appends the full records — enriched
with node and ref — as JSON lines. Exits nonzero if no records were found, so a
silently broken bench run fails the job. Stdlib only: runs anywhere.
"""

import argparse
import json
import os
import socket
import sys

MARKER = "BENCH_JSON "


def parse_log(path):
    records = []
    with open(path, errors="replace") as f:
        for line in f:
            at = line.find(MARKER)
            if at < 0:
                continue
            try:
                records.append(json.loads(line[at + len(MARKER) :]))
            except json.JSONDecodeError as e:
                print(f"warning: unparseable BENCH_JSON line skipped ({e})", file=sys.stderr)
    return records


def summarize(rec):
    c, m = rec.get("config", {}), rec.get("metrics", {})
    head = (
        f"{rec.get('kind', '?'):8s} ref={rec.get('ref', '?')} "
        f"node={rec.get('node', '?')} mesh={rec.get('mesh', '?')}"
    )
    if rec.get("kind") == "decode":
        step = m.get("step", {})
        parts = [
            f"b={c.get('batch')}",
            f"isl={c.get('isl')}",
            f"t/s/u={m.get('tsu_median')}",
            f"median={step.get('median_ms')}ms",
            f"p90={step.get('p90_ms')}ms",
            f"replay={m.get('replay_only_ms')}ms",
        ]
        if m.get("ttft_s") is not None:
            parts.append(f"ttft={m.get('ttft_s')}s")
        if c.get("synth_state"):
            parts.append("SYNTH")
        if c.get("mode") == "eager":
            parts.append("EAGER")
        if m.get("rows_identical") is False:
            parts.append("ROWS_DIVERGED")
        return f"{head} {' '.join(parts)}"
    if rec.get("kind") == "prefill":
        return (
            f"{head} isl={c.get('isl')} ms/tok={m.get('ms_per_token')} "
            f"tok/s={m.get('prefill_tok_s')} ttft={m.get('ttft_s')}s capture={m.get('capture_s')}s"
        )
    return f"{head} {json.dumps(m)[:120]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--append", metavar="JSONL", help="append enriched records to this file")
    ap.add_argument("--ref", help="override the ref recorded by the bench process")
    ap.add_argument("--node", help="node name (default: $SLURMD_NODENAME or hostname)")
    args = ap.parse_args()

    records = parse_log(args.log)
    if not records:
        print(f"parse_bench: no BENCH_JSON records in {args.log}", file=sys.stderr)
        return 1

    node = args.node or os.environ.get("SLURMD_NODENAME") or socket.gethostname()
    for rec in records:
        rec["node"] = node
        if args.ref:
            rec["ref"] = args.ref
        print(summarize(rec))

    if args.append:
        os.makedirs(os.path.dirname(os.path.abspath(args.append)), exist_ok=True)
        with open(args.append, "a") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"parse_bench: appended {len(records)} record(s) to {args.append}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
