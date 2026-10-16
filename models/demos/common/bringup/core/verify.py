# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Re-verify passed tasks under the current framework, without touching the run (F82).

    python -m models.demos.common.bringup.core.verify --spec S --tasks 'C.*' S.swa.15 [--status PASS] [--out DIR]

Runs each matching task's gate command as its gate would (same env, the safe runner), but with the metrics going to
``runs/<run>/verify/<stamp>/results`` and the logs next to them: state.json, results/ and the commits are untouched,
so a step that passed under older limits keeps its status while the overseer sees whether it still passes (after a
change of limits, F76-F81). Prints one line per task and writes summary.json; exit 1 if any task fails. A failure is
the overseer's to act on (a fix job, overseer.py), never an automatic re-run of the task.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import time
from pathlib import Path

from models.demos.common.bringup.core import metrics as M
from models.demos.common.bringup.core.gate import check_metrics, gate_command, gate_env, run_name
from models.demos.common.bringup.core.ledger import Ledger


def verify(spec, patterns: list[str], status: str | None = "PASS", out: Path | None = None) -> dict:
    led = Ledger(spec.bringup_dir)
    state = led.state()
    tids = [
        t
        for t in led.topo_order()
        if any(fnmatch.fnmatchcase(t, p) for p in patterns)
        and (status is None or state.get(t, {}).get("status") == status)
    ]
    out = Path(out or spec.run_dir(run_name(led)) / "verify" / time.strftime("%Y%m%dT%H%M%S"))
    (out / "results").mkdir(parents=True, exist_ok=True)
    rows = []
    for tid in tids:
        task = led.task(tid)
        env = gate_env(spec, led, tid)
        env[M.RESULTS_ENV] = str(out / "results")
        log = out / f"{tid}.log"
        t0 = time.time()
        with open(log, "w") as f:
            cmd = gate_command(task)
            f.write(f"$ {cmd}\n")
            f.flush()
            rc = subprocess.run(cmd, shell=True, cwd=spec.repo, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        got = M.load(tid, out / "results")
        ok, lines = check_metrics(task["gate"].get("metrics", {}), got)
        verdict = "PASS" if rc == 0 and ok else "FAIL"
        rows.append({"task": tid, "verdict": verdict, "rc": rc, "s": round(time.time() - t0, 1), "lines": lines})
        print(f"{verdict} {tid} (rc {rc}, {time.time() - t0:.0f}s, log {log})", flush=True)
        for line in lines if verdict == "FAIL" else []:
            print(line, flush=True)
    summary = {"out": str(out), "tasks": rows, "failed": [r["task"] for r in rows if r["verdict"] != "PASS"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"verify: {len(rows) - len(summary['failed'])}/{len(rows)} pass; {out}/summary.json")
    return summary


def main(argv=None) -> int:
    import os

    from models.demos.common.bringup.core.spec import Spec

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=os.environ.get("BRINGUP_SPEC"))
    ap.add_argument("--tasks", nargs="+", required=True, help="task id globs")
    ap.add_argument("--status", default="PASS", help="only tasks in this status ('' for any)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    res = verify(Spec.load(a.spec), a.tasks, a.status or None, Path(a.out) if a.out else None)
    return 1 if res["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
