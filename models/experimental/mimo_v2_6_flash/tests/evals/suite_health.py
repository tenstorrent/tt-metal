# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Health gates of a run_suite.sh run, written to <dir>/status.json every --interval seconds (and appended to
<dir>/status.jsonl), until the suite process exits.

Per bench step: requests sent (raw-log records), answered (status 200 with content or reasoning), failed, empty
answers, 5xx attempts, 429 attempts, cost. With --server-dir (our serve_mm_eval.sh log dir): per engine the
"ENCODER FAILURE" count, "Application startup complete" count (> 1 = restarted), HTTP 5xx in its access log, the last
DRAM line, /health of each engine port; the gate fails when an engine is unhealthy or restarted, and then the suite's
process group gets SIGTERM (--stop-on-engine-death; the server and its logs are left as they are). With --budget-usd
(OpenRouter): the key's usage (GET /api/v1/key, OPENROUTER_API_KEY) is checked every tick and the suite is stopped
once usage reaches the budget.

  suite_health.py --dir RUN/tt --suite-pid PID [--server-dir RUN/tt/server --ports 8731,8732,8733,8734]
                  [--budget-usd 40] [--interval 1200]
"""

import argparse
import glob
import json
import os
import re
import signal
import time
import urllib.request

ACCESS_5XX = re.compile(r'"(?:POST|GET) [^"]*" 5\d\d')


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def bench_steps(run_dir):
    steps = {}
    for raw in sorted(glob.glob(os.path.join(run_dir, "*", "*", "journal.raw.jsonl"))):
        out = os.path.dirname(raw)
        name = os.path.relpath(out, run_dir)
        s = dict(requests=0, answered=0, failed=0, empty=0, attempts_5xx=0, attempts_429=0, cost_usd=0.0)
        s["finish"] = {}
        with open(raw) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                s["requests"] += 1
                for a in rec.get("attempts") or []:
                    st = a.get("status")
                    if isinstance(st, int) and st >= 500:
                        s["attempts_5xx"] += 1
                    if st == 429:
                        s["attempts_429"] += 1
                if rec.get("status") != 200:
                    s["failed"] += 1
                    continue
                body = rec.get("response") or {}
                choice = (body.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                finish = str(choice.get("finish_reason"))
                s["finish"][finish] = s["finish"].get(finish, 0) + 1
                s["cost_usd"] += float((body.get("usage") or {}).get("cost") or 0)
                text = (msg.get("content") or "").strip()
                reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
                if text or reasoning:
                    s["answered"] += 1
                else:
                    s["empty"] += 1
        s["cost_usd"] = round(s["cost_usd"], 4)
        s["finished"] = os.path.exists(os.path.join(out, "summary.json"))
        steps[name] = s
    return steps


def engines(server_dir, ports):
    out = {}
    status_file = os.path.join(server_dir, "status")
    server_status = open(status_file).read().strip() if os.path.exists(status_file) else None
    for i, log in enumerate(sorted(glob.glob(os.path.join(server_dir, "row*.log")))):
        e = dict(encoder_failures=0, startups=0, http_5xx=0, dram=None, errors=0)
        with open(log, errors="replace") as f:
            for line in f:
                if "ENCODER FAILURE" in line:
                    e["encoder_failures"] += 1
                if "Application startup complete" in line:
                    e["startups"] += 1
                if ACCESS_5XX.search(line):
                    e["http_5xx"] += 1
                if "DRAM" in line and "MiMo" in line:
                    e["dram"] = line.strip()[-300:]
                if "Traceback (most recent call last)" in line or "EngineDeadError" in line:
                    e["errors"] += 1
        port = ports[i] if i < len(ports) else None
        e["port"] = port
        e["health"] = None
        if port:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=20) as r:
                    e["health"] = r.status
            except Exception as exc:  # noqa: BLE001 - any failure is "unhealthy"
                e["health"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        out[os.path.basename(log)[:-4]] = e
    return server_status, out


def key_usage():
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/key", headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)["data"]
    return {k: data.get(k) for k in ("usage", "limit", "limit_remaining")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--suite-pid", type=int, required=True)
    ap.add_argument("--server-dir")
    ap.add_argument("--ports", default="")
    ap.add_argument("--budget-usd", type=float)
    ap.add_argument("--interval", type=float, default=1200)
    ap.add_argument("--budget-interval", type=float, default=60)
    ap.add_argument("--stop-on-engine-death", action="store_true")
    args = ap.parse_args()
    ports = [p for p in args.ports.split(",") if p]
    last_full = 0.0
    stopped = None
    while True:
        alive = _alive(args.suite_pid)
        now = time.time()
        usage = None
        if args.budget_usd is not None:
            try:
                usage = key_usage()
            except Exception as exc:  # noqa: BLE001 - keep watching; report it
                usage = {"error": str(exc)[:200]}
            if usage.get("usage") is not None and usage["usage"] >= args.budget_usd and alive and not stopped:
                stopped = f"key usage {usage['usage']:.2f} reached the budget {args.budget_usd}"
                os.killpg(os.getpgid(args.suite_pid), signal.SIGTERM)
        if now - last_full >= args.interval or not alive or stopped:
            last_full = now
            status = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "suite_alive": alive}
            status["steps"] = bench_steps(args.dir)
            gates = {
                "requests_eq_answered": all(
                    s["requests"] == s["answered"] for s in status["steps"].values() if s["finished"]
                ),
                "failed_requests": sum(s["failed"] for s in status["steps"].values()),
                "empty_answers": sum(s["empty"] for s in status["steps"].values()),
                "attempts_5xx": sum(s["attempts_5xx"] for s in status["steps"].values()),
            }
            if args.server_dir:
                server_status, engs = engines(args.server_dir, ports)
                status["server_status"] = server_status
                status["engines"] = engs
                gates["encoder_failures"] = sum(e["encoder_failures"] for e in engs.values())
                gates["engine_http_5xx"] = sum(e["http_5xx"] for e in engs.values())
                gates["engine_restarts"] = sum(max(0, e["startups"] - 1) for e in engs.values())
                dead = [n for n, e in engs.items() if e["health"] != 200]
                gates["engines_unhealthy"] = dead
                if (dead or gates["engine_restarts"]) and args.stop_on_engine_death and alive and not stopped:
                    stopped = f"engine(s) unhealthy {dead} / restarts {gates['engine_restarts']}: suite stopped"
                    os.killpg(os.getpgid(args.suite_pid), signal.SIGTERM)
            if usage is not None:
                status["openrouter_key"] = usage
            status["gates"] = gates
            status["stopped"] = stopped
            tmp = os.path.join(args.dir, "status.json.tmp")
            with open(tmp, "w") as f:
                json.dump(status, f, indent=1)
            os.replace(tmp, os.path.join(args.dir, "status.json"))
            with open(os.path.join(args.dir, "status.jsonl"), "a") as f:
                f.write(json.dumps(status) + "\n")
        if not alive:
            return
        time.sleep(min(args.interval, args.budget_interval if args.budget_usd is not None else args.interval))


if __name__ == "__main__":
    main()
