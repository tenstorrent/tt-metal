#!/usr/bin/env python3
"""Dream-RSI eval adapter: device kernel time of one tt-metal op, measured by a pytest test under Tracy.

    python agent_orch/adapters/ttmetal_op_perf.py --test tests/.../test_my_op.py --op-code MyOpDeviceOperation \\
        --warmup 3 [--measured 10] [-k EXPR] [--extra 'pcc=PCC: ([0-9.eE+-]+)'] [--metric "DEVICE KERNEL DURATION [ns]"]

Each test case (pytest node id) is one case of the result. For each case the adapter runs the test
alone under the Tracy profiler, reads the ops perf CSV, keeps the rows whose OP CODE is --op-code,
drops the first --warmup calls on every device, and reports the mean of --metric over the remaining
calls and devices (in µs for *[ns] metrics). The case is valid when its pytest run passes.

What the test must do: call the op (warmup + measured) times per case on the device(s), and check or
log correctness. Extra numbers for the campaign's gates come from the test's output, either as
`DREAM_METRIC name=value` lines (any number of pairs per line) or via --extra name=REGEX (first group,
last match wins). Case ids are the pytest parameter ids with the parts shared by all cases removed.

Writes the result JSON to $DREAM_RESULT (or --out); full Tracy output goes under $DREAM_OUT.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

DREAM_METRIC = re.compile(r"DREAM_METRIC\s+(.*)")
PAIR = re.compile(r"([A-Za-z_][\w.]*)=([-+0-9.eE]+|nan|inf)")


_COLLECTOR = """
import json, sys
import pytest

ids = []


class Collector:
    def pytest_collection_finish(self, session):
        ids.extend(item.nodeid for item in session.items)


rc = pytest.main(["--collect-only", "-p", "no:cacheprovider", *sys.argv[1:]], plugins=[Collector()])
print("DREAM_IDS " + json.dumps(ids))
sys.exit(0 if ids else (rc or 1))
"""


def collect(test: str, k: str | None, cwd: Path) -> list[str]:
    """Node ids of the selected tests, read from pytest itself (immune to addopts such as -vv)."""
    args = [test] + (["-k", k] if k else [])
    r = subprocess.run([sys.executable, "-c", _COLLECTOR, *args], cwd=cwd, capture_output=True, text=True)
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("DREAM_IDS ")), None)
    ids = json.loads(line[len("DREAM_IDS ") :]) if line else []
    if not ids:
        raise RuntimeError(
            f"no tests collected from {test} (rc {r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
        )
    return ids


def case_names(ids: list[str], case_regex: str | None) -> list[str]:
    params = [i.split("[", 1)[1][:-1] if "[" in i else i.rsplit("::", 1)[-1] for i in ids]
    if case_regex:
        out = []
        for p in params:
            m = re.search(case_regex, p)
            out.append(m.group(1) if m and m.groups() else (m.group(0) if m else p))
        return out
    if len(params) == 1:
        return [params[0] if len(params[0]) <= 60 else "default"]
    pre = os.path.commonprefix(params)
    suf = os.path.commonprefix([p[::-1] for p in params])[::-1]
    out = [p[len(pre) : len(p) - len(suf) if suf else None].strip("-_ ") for p in params]
    return out if len(set(out)) == len(out) and all(out) else params


def parse_extras(text: str, extra: list[tuple[str, re.Pattern]]) -> dict:
    vals: dict[str, float] = {}
    for m in DREAM_METRIC.finditer(text):
        for k, v in PAIR.findall(m.group(1)):
            vals[k] = float(v)
    for name, rx in extra:
        hits = rx.findall(text)
        if hits:
            h = hits[-1]
            try:
                vals[name] = float(h[0] if isinstance(h, tuple) else h)
            except ValueError:
                pass
    return vals


def find_csv(d: Path) -> Path | None:
    hits = sorted(glob.glob(str(d / "reports" / "*" / "ops_perf_results_*.csv")))
    return Path(hits[-1]) if hits else None


def measure(csv: Path, op_code: str, metric: str, warmup: int, measured: int | None) -> tuple[dict | None, str | None]:
    import pandas as pd

    df = pd.read_csv(csv)
    if "OP CODE" not in df.columns or metric not in df.columns:
        return None, f"ops CSV lacks 'OP CODE' or '{metric}' ({csv})"
    d = df[df["OP CODE"] == op_code].copy()
    if d.empty:
        codes = ", ".join(sorted(map(str, df["OP CODE"].unique()))[:20])
        return None, f"no rows with OP CODE {op_code!r} in {csv}; seen: {codes}"
    d = d.sort_values(["DEVICE ID", "GLOBAL CALL COUNT"])
    d["call"] = d.groupby("DEVICE ID").cumcount()
    per_dev = d.groupby("DEVICE ID")["call"].max() + 1
    if per_dev.nunique() != 1:
        return None, f"devices ran the op a different number of times: {per_dev.to_dict()}"
    n = int(per_dev.iloc[0]) - warmup
    if n <= 0 or (measured is not None and n != measured):
        return None, f"{int(per_dev.iloc[0])} calls per device, expected warmup {warmup} + measured {measured}"
    g = d[d["call"] >= warmup]
    scale = 1e-3 if metric.endswith("[ns]") else 1.0
    per_call_max = g.groupby("call")[metric].max() * scale
    out = {
        "value": round(float(g[metric].mean() * scale), 4),
        "chip_max_mean": round(float(per_call_max.mean()), 4),
        "p50": round(float(g[metric].median() * scale), 4),
        "min": round(float(g[metric].min() * scale), 4),
        "max": round(float(g[metric].max() * scale), 4),
        "calls": n,
        "devices": int(d["DEVICE ID"].nunique()),
    }
    if "CORE COUNT" in g:
        out["core_count"] = int(g["CORE COUNT"].iloc[0])
    return out, None


def classify(text: str) -> str:
    if re.search(r":\d+:\d+: error:", text) or re.search(r"(?i)failed to (compile|build)|compile failed", text):
        return "jit_compile_error"
    return "runtime_error"


def excerpt(text: str, n: int = 40) -> str:
    lines = text.splitlines()
    idx = [i for i, l in enumerate(lines) if re.search(r"error|Error|FAILED|TT_FATAL|TT_THROW|Traceback", l)]
    s = max(0, idx[0] - 5) if idx else max(0, len(lines) - n)
    return "\n".join(lines[s : s + n])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", required=True, help="pytest file or node id prefix, relative to the checkout root")
    ap.add_argument("--op-code", required=True, help="value of the OP CODE column for the op being optimized")
    ap.add_argument("--warmup", type=int, default=0, help="calls per device per case to drop")
    ap.add_argument("--measured", type=int, help="expected measured calls per device per case (checked)")
    ap.add_argument("--metric", default="DEVICE KERNEL DURATION [ns]")
    ap.add_argument("-k", dest="k", help="pytest -k expression to select cases")
    ap.add_argument("--extra", action="append", default=[], help="name=REGEX: extra value from the test output")
    ap.add_argument("--case-regex", help="regex on the pytest param id; group 1 (or the match) names the case")
    ap.add_argument("--tracy-port", type=int, default=int(os.environ.get("DREAM_TRACY_PORT", 8086)))
    ap.add_argument("--out", type=Path, default=os.environ.get("DREAM_RESULT"))
    ap.add_argument("--work", type=Path, default=os.environ.get("DREAM_OUT") or "generated/dream_adapter")
    a = ap.parse_args()
    if not a.out:
        ap.error("--out or $DREAM_RESULT is required")
    extra = []
    for e in a.extra:
        name, _, rx = e.partition("=")
        extra.append((name, re.compile(rx)))
    cwd = Path.cwd()
    a.work.mkdir(parents=True, exist_ok=True)

    try:
        result = run_cases(a, extra, cwd)
    except Exception as e:  # never leave the engine without a result file
        result = {
            "valid": False,
            "cases": {},
            "fail_class": "infra",
            "error": f"adapter error: {type(e).__name__}: {e}",
            "adapter": "ttmetal_op_perf",
        }
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[adapter] wrote {a.out}: valid={result['valid']}", flush=True)


def run_cases(a, extra: list, cwd: Path) -> dict:
    ids = collect(a.test, a.k, cwd)
    names = case_names(ids, a.case_regex)
    result = {"valid": True, "cases": {}, "error": None, "fail_class": None, "adapter": "ttmetal_op_perf"}
    for i, (nid, name) in enumerate(zip(ids, names)):
        out = a.work / f"case{i:02d}"
        log = a.work / f"case{i:02d}.log"
        cmd = [
            sys.executable,
            "-m",
            "tracy",
            "-t",
            str(a.tracy_port),
            "-o",
            str(out),
            "-r",
            "-p",
            "-v",
            "-m",
            "pytest",
            nid,
        ]
        print(f"[adapter] case {name}: {nid}", flush=True)
        with open(log, "w") as f:
            rc = subprocess.run(
                cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "TRACY_NO_WEB_SERVER": "1"}
            ).returncode
        text = log.read_text(errors="replace")
        case = {"pytest_rc": rc, "nodeid": nid}
        case.update(parse_extras(text, extra))
        if rc != 0:
            result.update(valid=False, fail_class=classify(text), error=f"{name}: pytest exited {rc}\n" + excerpt(text))
            result["cases"][name] = {"value": None, **case}
            break  # later cases would only cost time; the attempt is invalid either way
        csv = find_csv(out)
        m, err = (
            measure(csv, a.op_code, a.metric, a.warmup, a.measured)
            if csv
            else (None, f"no ops perf CSV under {out}/reports")
        )
        if err:
            result.update(valid=False, fail_class="infra", error=f"{name}: {err}")
            result["cases"][name] = {"value": None, **case}
            break
        result["cases"][name] = {**m, **case}
        print(f"[adapter]   {m['value']} (mean of {m['calls']} calls x {m['devices']} devices) {case}", flush=True)
    return result


if __name__ == "__main__":
    main()
