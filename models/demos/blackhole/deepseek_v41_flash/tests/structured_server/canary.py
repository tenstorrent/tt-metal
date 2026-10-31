#!/usr/bin/env python3
"""Hang canary with structured rows mixed in: usage: canary.py <out.json> [iterations=30] [tokens=8000] [count=8]
Each iteration: `count` fresh needle prompts of ~tokens tokens at once (greedy, 16 new tokens) AND, started 1 s later, three structured chat requests
(json_schema, regex, tool_choice required) next to them. Every needle must be answered, every structured output valid, nothing may error or stall (a
request with no first byte for 600 s is a hang). Prints one line per iteration."""
import concurrent.futures as cf
import json
import sys
import threading
import time

sys.path.insert(0, "/mnt/tt-data/ssinghal/dsv4-logs/cand_stab/clients")
sys.path.insert(0, "/mnt/tt-data/ssinghal/dsv4-logs/structured/clients")
import l2lib as L  # noqa: E402
import matrix as MX  # noqa: E402

out, iters = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 30
N = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
cnt = int(sys.argv[4]) if len(sys.argv) > 4 else 8
seed0 = int(time.time()) % 100000
args = type("A", (), {"host": "127.0.0.1", "port": 8000, "max_tokens": 200})()
log = []
bad = 0
for it in range(iters):
    rows = [L.mkprompt(N, seed0 * 1000 + it * 20 + i) for i in range(cnt)]
    res = [{} for _ in rows]

    def go(i):
        L.sreq("/v1/completions", {"prompt": rows[i][0], "max_tokens": 16, "temperature": 0}, res[i], timeout=1800)

    def structured(k):
        time.sleep(1.0)
        name = ("json_schema", "regex", "required")[k]
        return MX.run_cell(args, name, bool(it % 2), 0.0 if k != 1 else 0.8)

    t0 = time.time()
    th = [threading.Thread(target=go, args=(i,)) for i in range(cnt)]
    [t.start() for t in th]
    with cf.ThreadPoolExecutor(3) as ex:
        sres = list(ex.map(structured, range(3)))
    [t.join() for t in th]
    el = time.time() - t0
    ok_n = sum(rows[i][1] in res[i].get("text", "") for i in range(cnt))
    errs = [r.get("err") for r in res if r.get("err")]
    sbad = [r for r in sres if r[1] != "PASS"]
    ttft = [round(r["tf"] - r["t0"], 1) for r in res if r.get("tf")]
    good = ok_n == cnt and not errs and not sbad
    bad += not good
    line = "iter %2d %s %.0fs needles %d/%d ttft mean %.1f max %.1f structured %s%s%s" % (
        it,
        "OK  " if good else "FAIL",
        el,
        ok_n,
        cnt,
        sum(ttft) / max(1, len(ttft)),
        max(ttft or [0]),
        [(r[0], round(r[3], 1)) for r in sres],
        (" errs=%s" % errs[:2]) if errs else "",
        (" sbad=%s" % [r[:3] for r in sbad]) if sbad else "",
    )
    print(line, flush=True)
    log.append(line)
json.dump(log, open(out, "w"), indent=1)
print("CANARY %d/%d iterations ok" % (iters - bad, iters))
sys.exit(1 if bad else 0)
