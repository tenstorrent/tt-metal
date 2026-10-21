#!/usr/bin/env python3
"""chunk-gdn-tp4-queue eval: device time of one chunk_gated_delta_rule call at the TP-4 prefill shapes, all ops included.

1. Per case (bh12, bh16) runs test_chunk_gdn_tp4_perf.py under Tracy and reads the ops perf CSV. The test puts a
   `gdn_call` signpost before every call; the device time of a call is the sum of DEVICE KERNEL DURATION over every
   device op between that signpost and the next one, so a path change (fused / phased / mono) or work moved into
   pre- or post-processing ops is counted. value = the mean over the MEASURED calls after WARMUP (us).
   The test's `DREAM_METRIC` line gives pcc_o, pcc_fs, max_abs_o, repeat_exact (gated in campaign.yaml).
2. Runs the chunk_gated_delta_rule correctness suites (fused == phased bit-exact, the geometry model mirror, the
   prims, the whole op); any failure makes the attempt invalid.

    python agent_orch/campaigns/chunk-gdn-tp4-queue/eval.py            # writes $DREAM_RESULT (or --out)
    python agent_orch/campaigns/chunk-gdn-tp4-queue/eval.py --no-suite # timing and accuracy only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

TEST = "tests/ttnn/unit_tests/operations/transformers/test_chunk_gdn_tp4_perf.py"
CASES = {"bh12": f"{TEST}::test_chunk_gdn_tp4_prefill[bh12]", "bh16": f"{TEST}::test_chunk_gdn_tp4_prefill[bh16]"}
_SUITE_DIR = "tests/ttnn/unit_tests/operations/transformers/"
SUITES = [
    _SUITE_DIR + name
    for name in (
        "test_chunk_gdn_fused_geometry.py",
        "test_chunk_gdn_fused.py",
        "test_chunk_gdn_prims.py",
        "test_chunk_gated_delta_rule.py",
    )
]
WARMUP, MEASURED = 3, 10
METRIC = "DEVICE KERNEL DURATION [ns]"
CALL_SIGNPOST = "gdn_call"
PAIR = re.compile(r"([A-Za-z_][\w.]*)=([-+0-9.eE]+|nan|inf)")


def excerpt(text: str, n: int = 40) -> str:
    lines = text.splitlines()
    idx = [i for i, l in enumerate(lines) if re.search(r"error|Error|FAILED|TT_FATAL|TT_THROW|Traceback", l)]
    s = max(0, idx[0] - 5) if idx else max(0, len(lines) - n)
    return "\n".join(lines[s : s + n])


def classify(text: str) -> str:
    if re.search(r":\d+:\d+: error:", text) or re.search(r"(?i)failed to (compile|build)|compile failed", text):
        return "jit_compile_error"
    if re.search(r"PCC .* < |not deterministic", text):
        return "accuracy_fail"
    return "runtime_error"


def metrics(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        if "DREAM_METRIC" in line:
            for k, v in PAIR.findall(line.split("DREAM_METRIC", 1)[1]):
                out[k] = float(v)
    return out


def per_call(csv: Path) -> tuple[list[dict] | None, str | None]:
    """Device time of each call: the ops between one `gdn_call` signpost and the next signpost, in timeline order."""
    import pandas as pd

    df = pd.read_csv(csv)
    if "OP TYPE" not in df.columns or METRIC not in df.columns:
        return None, f"ops CSV lacks 'OP TYPE' or '{METRIC}' ({csv})"
    calls, cur = [], None
    for _, r in df.iterrows():
        if r["OP TYPE"] == "signpost":
            cur = {"us": 0.0, "ops": {}, "missing": 0, "cores": 0} if r["OP CODE"] == CALL_SIGNPOST else None
            if cur is not None:
                calls.append(cur)
            continue
        if cur is None:
            continue
        dur = r[METRIC]
        if pd.isna(dur):
            if str(r["OP TYPE"]) != "python_fallback" and "device" in str(r["OP TYPE"]).lower():
                cur["missing"] += 1
            continue
        code = str(r["OP CODE"])
        cur["us"] += float(dur) * 1e-3
        cur["ops"][code] = cur["ops"].get(code, 0.0) + float(dur) * 1e-3
        if "CORE COUNT" in r and not pd.isna(r["CORE COUNT"]):
            cur["cores"] = max(cur["cores"], int(r["CORE COUNT"]))
    if len(calls) != WARMUP + MEASURED:
        return None, f"{len(calls)} '{CALL_SIGNPOST}' signposts in {csv}, expected {WARMUP + MEASURED}"
    bad = [i for i, c in enumerate(calls) if c["missing"] or c["us"] <= 0]
    if bad:
        return None, f"calls {bad} have device ops without a device time (profiler buffer?) in {csv}"
    return calls, None


def run_case(name: str, nodeid: str, work: Path, port: int) -> dict:
    out, log = work / name, work / f"{name}.log"
    cmd = [sys.executable, "-m", "tracy", "-t", str(port), "-o", str(out), "-r", "-p", "-v", "-m", "pytest"]
    cmd.append(shlex.quote(nodeid))  # tracy re-joins its args into one `sh -c` string
    print(f"[chunk-gdn-tp4-queue] case {name}: {nodeid}", flush=True)
    with open(log, "w") as f:
        rc = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "TRACY_NO_WEB_SERVER": "1"}
        ).returncode
    text = log.read_text(errors="replace")
    case = {"value": None, "pytest_rc": rc, **metrics(text)}
    if rc != 0:
        return {"case": case, "error": f"{name}: pytest exited {rc}\n" + excerpt(text), "fail_class": classify(text)}
    hits = sorted(glob.glob(str(out / "reports" / "*" / "ops_perf_results_*.csv")))
    if not hits:
        return {"case": case, "error": f"{name}: no ops perf CSV under {out}/reports", "fail_class": "infra"}
    calls, err = per_call(Path(hits[-1]))
    if err:
        return {"case": case, "error": f"{name}: {err}", "fail_class": "infra"}
    m = calls[WARMUP:]
    us = sorted(c["us"] for c in m)
    codes = sorted({k for c in m for k in c["ops"]})
    case.update(
        value=round(sum(us) / len(us), 3),
        p50=round(us[len(us) // 2], 3),
        min=round(us[0], 3),
        max=round(us[-1], 3),
        calls=len(m),
        cores=max(c["cores"] for c in m),
        ops={k: round(sum(c["ops"].get(k, 0.0) for c in m) / len(m), 3) for k in codes},
    )
    print(f"[chunk-gdn-tp4-queue]   {case['value']} us (mean of {len(m)} calls) {case}", flush=True)
    return {"case": case, "error": None, "fail_class": None}


def run_suite() -> tuple[bool, str]:
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *SUITES], capture_output=True, text=True
    )
    text = r.stdout + r.stderr
    tail = "\n".join([l for l in text.splitlines() if "FAILED" in l or "Error" in l][-30:])
    return r.returncode == 0, tail or excerpt(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=os.environ.get("DREAM_RESULT"))
    ap.add_argument("--work", type=Path, default=os.environ.get("DREAM_OUT") or "generated/chunk_gdn_tp4")
    ap.add_argument("--tracy-port", type=int, default=int(os.environ.get("DREAM_TRACY_PORT", 8086)))
    ap.add_argument("--no-suite", action="store_true", help="skip the correctness suites")
    a = ap.parse_args()
    if not a.out:
        ap.error("--out or $DREAM_RESULT is required")
    a.work.mkdir(parents=True, exist_ok=True)

    res = {"valid": True, "cases": {}, "error": None, "fail_class": None}
    try:
        for name, nodeid in CASES.items():
            r = run_case(name, nodeid, a.work, a.tracy_port)
            res["cases"][name] = r["case"]
            if r["error"]:
                res.update(valid=False, error=r["error"], fail_class=r["fail_class"])
                break
        if res["valid"] and not a.no_suite:
            ok, tail = run_suite()
            res["suite_passed"] = ok
            if not ok:
                res.update(valid=False, fail_class="accuracy_fail", error=f"correctness suites failed:\n{tail}")
    except Exception as e:  # never leave the engine without a result file
        res.update(valid=False, fail_class="infra", error=f"eval error: {type(e).__name__}: {e}")
    a.out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"[chunk-gdn-tp4-queue] wrote {a.out}: valid={res['valid']} error={res['error']}", flush=True)


if __name__ == "__main__":
    main()
