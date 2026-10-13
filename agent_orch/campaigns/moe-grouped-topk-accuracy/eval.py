#!/usr/bin/env python3
"""moe-grouped-topk-accuracy eval: optimize MoE gate expert-selection accuracy while holding device time within a margin.

1. Runs the Dream perf adapter on the accuracy test (device kernel time + DREAM_METRIC lines per case).
2. Runs the existing moe_grouped_topk correctness suites and the moe_hash_gate suite (shared compute code);
   any failure makes the attempt invalid.
3. Rewrites each case so the campaign minimizes accuracy, not time:
     value       = miss_pct + FLOOR   (% of fp32-golden experts not selected; FLOOR keeps ratios finite)
     duration_us = the adapter's device kernel time
     slowdown    = duration_us / perf_baseline.json[case]   (gated in dream.yaml: slowdown <= 1.05)

    python agent_orch/campaigns/moe-grouped-topk-accuracy/eval.py                 # writes $DREAM_RESULT
    python agent_orch/campaigns/moe-grouped-topk-accuracy/eval.py --record-perf   # (re)writes perf_baseline.json from this checkout
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PERF_BASELINE = HERE / "perf_baseline.json"
TEST = "tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/test_moe_grouped_topk_prefill_perf.py"
_SUITE_DIR = "tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/"
SUITES = [
    _SUITE_DIR + name
    for name in (
        "test_moe_grouped_topk.py",
        "test_moe_grouped_topk_stable.py",
        "test_moe_grouped_topk_ties.py",
        "test_moe_grouped_topk_state.py",
        "test_moe_grouped_topk_row_major_weights.py",
        "test_moe_hash_gate.py",
    )
]
OP_CODE = "MoeGroupedTopkDeviceOperation"
FLOOR = 0.001  # miss_pct floor: 1e-5 of the selected experts


def run_adapter(out: Path) -> dict:
    cmd = [sys.executable, "agent_orch/adapters/ttmetal_op_perf.py", "--test", TEST, "--op-code", OP_CODE]
    cmd += ["--warmup", "3", "--measured", "10", "--out", str(out)]
    subprocess.run(cmd, check=False)
    return json.loads(out.read_text())


def run_suite() -> tuple[bool, str]:
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *SUITES], capture_output=True, text=True
    )
    tail = "\n".join([l for l in (r.stdout + r.stderr).splitlines() if "FAILED" in l or "Error" in l][-30:])
    return r.returncode == 0, tail


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=os.environ.get("DREAM_RESULT"))
    ap.add_argument("--record-perf", action="store_true", help="write perf_baseline.json and exit")
    a = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        res = run_adapter(Path(td) / "adapter.json")

    if a.record_perf:
        if not res.get("valid"):
            sys.exit(f"adapter run invalid: {res.get('error')}")
        PERF_BASELINE.write_text(json.dumps({c: v["value"] for c, v in res["cases"].items()}, indent=2) + "\n")
        print(f"wrote {PERF_BASELINE}: {PERF_BASELINE.read_text()}")
        return

    if not a.out:
        ap.error("--out or $DREAM_RESULT is required")
    if res.get("valid"):
        base = json.loads(PERF_BASELINE.read_text())
        for cid, c in res["cases"].items():
            dur = c["value"]
            c["duration_us"] = dur
            c["slowdown"] = dur / base[cid] if cid in base else None
            c["value"] = float(c["miss_pct"]) + FLOOR if "miss_pct" in c else None
            if c["value"] is None or c["slowdown"] is None:
                res.update(valid=False, fail_class="infra", error=f"{cid}: missing miss_pct or perf baseline")
                break
    if res.get("valid"):
        ok, tail = run_suite()
        res["suite_passed"] = ok
        if not ok:
            res.update(valid=False, fail_class="accuracy_fail", error=f"correctness suites failed:\n{tail}")
    a.out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"[moe-grouped-topk-accuracy] wrote {a.out}: valid={res.get('valid')} error={res.get('error')}", flush=True)


if __name__ == "__main__":
    main()
