# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarise test_expert_bw.py: per case, the expert ops' device time (sum of the ops in the tag, max over chips per
op) and the achieved weight-read bandwidth (bf4 weight bytes of the active experts / time), plus the op breakdown.

    python models/demos/mimo_v2_d_p/tests/perf/analyze_expert_bw.py <ops_perf_results.csv> [generated/mimo_expert_bw/cases.jsonl]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_tags import load, summarize  # noqa: E402

PEAK_GBPS = 512.0  # BH p150 GDDR6 nominal


def main():
    stats = {}
    try:
        for line in open(sys.argv[2] if len(sys.argv) > 2 else "generated/mimo_expert_bw/cases.jsonl"):
            s = json.loads(line)
            stats[s["tag"]] = s
    except FileNotFoundError:
        pass
    print(f"{'case':<28s} {'time us':>9s} {'weights MB':>10s} {'GB/s':>7s} {'% peak':>7s} {'TFLOP/s':>8s}  ops")
    for tag, iters in load(sys.argv[1]).items():
        total, ops = summarize(iters)
        s = stats.get(tag, {})
        wb, fl = s.get("weight_bytes", 0), s.get("flops", 0)
        gbps = wb / (total * 1e3) if total else 0
        breakdown = ", ".join(f"{k.split(' ', 1)[1].replace('DeviceOperation', '')} {v:.0f}" for k, v in ops.items())
        print(
            f"{tag:<28s} {total:9.1f} {wb / 1e6:10.0f} {gbps:7.1f} {100 * gbps / PEAK_GBPS:6.1f}% {fl / (total * 1e6):8.1f}  {breakdown}"
        )


if __name__ == "__main__":
    main()
