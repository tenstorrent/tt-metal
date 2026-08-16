# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""How wide is the run-to-run band on the ranking metric?

The sweep ranks on traced teacher-forcing decode t/s/u, and the interesting
deltas turned out to be fractions of a percent. A frontier drawn through
differences smaller than the measurement's own spread is a frontier drawn
through noise, so the spread has to be *measured* rather than assumed.

This re-runs the **identical** config several times -- same JSON, same command,
same machine, fresh process each time, exactly as a sweep row runs -- and
reports the spread of the decode figure. Whatever band that produces is the
resolution limit of every comparison in ``sweep_results.json``: a candidate
whose gain over the default does not clear it has not been shown to be faster.

Run it on the default *and* on the leading contender, because there is no
guarantee the two have the same variance.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.doc.datatype_sweep.probes.candidates import by_id  # noqa: E402
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.doc.datatype_sweep.probes.sweep_runner import (  # noqa: E402
    run_row,
)

HERE = Path(__file__).resolve().parent
SWEEP_DIR = HERE.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="comma-separated config ids")
    ap.add_argument("--n", type=int, default=3, help="repetitions per id")
    ap.add_argument("--out", default="repeats.json")
    args = ap.parse_args()

    out_path = SWEEP_DIR / args.out
    data = json.loads(out_path.read_text()) if out_path.exists() else {}

    for cid in args.ids.split(","):
        cand = by_id(cid)
        samples = data.get(cid, {}).get("samples", [])
        for i in range(len(samples), args.n):
            print(f"{cid} repeat {i + 1}/{args.n}", flush=True)
            row = run_row(cand, {})
            if row["status"] != "ok":
                print(f"  -> FAILED: {row.get('error')}", flush=True)
                continue
            samples.append(
                {
                    "decode_tps_user": row["decode_tps_user"],
                    "ttft_ms": row["ttft_ms"],
                    "top1": row["top1"],
                    "top5": row["top5"],
                    "top100": row["top100"],
                }
            )
            print(f"  -> {row['decode_tps_user']} t/s/u  top1={row['top1']}", flush=True)
            data[cid] = summarise(cid, samples)
            out_path.write_text(json.dumps(data, indent=2) + "\n")

    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "samples"} for k, v in data.items()}, indent=2))


def summarise(cid: str, samples: list[dict]) -> dict:
    vals = [s["decode_tps_user"] for s in samples]
    top1s = [s["top1"] for s in samples]
    out = {
        "config_id": cid,
        "n": len(vals),
        "samples": samples,
        "decode_min": min(vals),
        "decode_max": max(vals),
        "decode_median": statistics.median(vals),
        "decode_mean": round(statistics.fmean(vals), 4),
        "decode_spread_abs": round(max(vals) - min(vals), 4),
        "decode_spread_pct": round((max(vals) - min(vals)) / statistics.fmean(vals) * 100, 4),
        "top1_values": top1s,
        "top1_stable": len(set(top1s)) == 1,
    }
    if len(vals) > 1:
        out["decode_stdev"] = round(statistics.stdev(vals), 4)
    return out


if __name__ == "__main__":
    main()
