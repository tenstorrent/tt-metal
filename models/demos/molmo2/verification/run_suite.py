# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run the 105-video multiple-choice suite (test.jsonl) against one or more Molmo2 servers and score the
answers against the baseline (test_results.jsonl).

Requests are spread over the servers (one in-flight request per server; the servers are batch=1).
A test counts as correct when the first A-D letter of the response equals the baseline's; baseline
entries that errored (empty response) are excluded from the denominator.

    python models/demos/molmo2/verification/run_suite.py --ports 8001 8002 8003 --output /tmp/suite.jsonl
"""

import argparse
import json
import queue
import re
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent


def letter(text):
    m = re.search(r"\b([A-D])\b", (text or "").upper())
    return m.group(1) if m else None


def post(port, body, token=None, timeout=900):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", type=int, nargs="+", required=True)
    ap.add_argument("--tests", default=str(HERE / "test.jsonl"))
    ap.add_argument("--baseline", default=str(HERE / "test_results.jsonl"))
    ap.add_argument("--output", default="/tmp/molmo2_suite_results.jsonl")
    ap.add_argument("--model", default="allenai/Molmo2-8B")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--timeout", type=int, default=300, help="per-request timeout in seconds")
    ap.add_argument("--token")
    args = ap.parse_args()

    tests = [json.loads(line) for line in open(args.tests)]
    base = [json.loads(line) for line in open(args.baseline)]
    todo = queue.Queue()
    for i, t in enumerate(tests[: args.limit]):
        todo.put((i, t))
    results = {}
    lock = threading.Lock()

    def worker(port):
        while True:
            try:
                i, t = todo.get_nowait()
            except queue.Empty:
                return
            body = dict(t, model=args.model)
            t0 = time.time()
            try:
                resp, err = post(port, body, args.token, args.timeout), None
            except Exception as e:  # noqa: BLE001
                resp, err = "", str(e)
            with lock:
                results[i] = {
                    "idx": i,
                    "port": port,
                    "response": resp,
                    "error": err,
                    "secs": round(time.time() - t0, 2),
                }
                print(
                    f"[{len(results):3d}/{len(tests[: args.limit])}] idx={i:3d} port={port} {time.time() - t0:5.1f}s {resp!r} {err or ''}",
                    flush=True,
                )

    threads = [threading.Thread(target=worker, args=(p,)) for p in args.ports]
    t_start = time.time()
    [t.start() for t in threads]
    [t.join() for t in threads]

    correct = scored = errors = 0
    wrong = []
    with open(args.output, "w") as f:
        for i in sorted(results):
            r = results[i]
            exp = base[i]
            r["expected"], r["baseline_error"] = exp["response"], bool(exp["error"] or not exp["response"].strip())
            f.write(json.dumps(r) + "\n")
            if r["error"]:
                errors += 1
            if r["baseline_error"]:
                continue
            scored += 1
            ok = not r["error"] and letter(r["response"]) == letter(exp["response"])
            correct += ok
            if not ok:
                wrong.append((i, exp["response"].strip(), r["response"].strip(), r["error"]))
    print(
        f"\naccuracy vs baseline: {correct}/{scored} = {correct / max(scored, 1):.1%}  (request errors: {errors}, "
        f"wall {time.time() - t_start:.0f}s over {len(args.ports)} server(s))"
    )
    for w in wrong:
        print("  mismatch idx=%d baseline=%r got=%r %s" % w)


if __name__ == "__main__":
    main()
