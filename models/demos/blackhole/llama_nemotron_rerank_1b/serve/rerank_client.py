# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Client for rerank_server.py, rerank_serve_fastapi.py and tt-media-server /v1/rerank (stdlib only).

  --check REF.json   score every case of REF (rerank_hf_ref.py output: [{"query", "documents", "logits"}]) and
                     compare with the reference logits: Pearson, mean / max abs error, top-1 agreement; exit 1 when
                     Pearson < --min-pearson (default 0.9995, the PR #63253 gate) or a request fails.
  --load N           closed loop: N requests, --concurrency in flight, --docs documents of --words words each;
                     latency percentiles, req/s, pairs/s, status counts.

    python rerank_client.py --url http://127.0.0.1:8111 --wait-ready 900 --check ref.json
    python rerank_client.py --url http://127.0.0.1:8111 --load 500 --concurrency 8 --docs 25
"""

import argparse
import json
import math
import sys
import threading
import time
import urllib.error
import urllib.request


def post(url, body, timeout=120.0, api_key=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {"detail": e.read().decode(errors="replace")[:300]}
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        return 0, {"detail": repr(e)}


def wait_ready(base, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except urllib.error.HTTPError as e:
            if e.code == 503 and b'"failed"' in e.read():
                print("server reports failed", file=sys.stderr)
                return False
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(2)
    return False


def logits_by_index(resp):
    """/v1/rerank results are sorted; put each logit back at its document index."""
    out = [None] * len(resp["results"])
    for r in resp["results"]:
        out[r["index"]] = r["logit"]
    return out


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va == 0 or vb == 0:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def check(args):
    cases = json.load(open(args.check))
    got_all, ref_all, top1, fails = [], [], 0, 0
    for c in cases:
        status, resp = post(
            f"{args.url}/v1/rerank", {"query": c["query"], "documents": c["documents"]}, api_key=args.api_key
        )
        if status != 200:
            fails += 1
            print(f"FAIL {status}: {resp.get('detail')}", file=sys.stderr)
            continue
        got = logits_by_index(resp)
        got_all += got
        ref_all += c["logits"]
        top1 += max(range(len(got)), key=got.__getitem__) == max(range(len(got)), key=c["logits"].__getitem__)
    if not got_all:
        print("no successful request")
        return 1
    err = [abs(g - r) for g, r in zip(got_all, ref_all)]
    p = pearson(got_all, ref_all)
    print(
        json.dumps(
            {
                "cases": len(cases),
                "failed": fails,
                "pairs": len(got_all),
                "pearson": round(p, 6),
                "mean_abs_err": round(sum(err) / len(err), 4),
                "max_abs_err": round(max(err), 4),
                "top1_agree": f"{top1}/{len(cases) - fails}",
            }
        )
    )
    return 0 if fails == 0 and p >= args.min_pearson else 1


def load(args):
    words = " ".join(["passage"] * args.words)
    body = {"query": "which passage answers the question?", "documents": [f"{i} {words}" for i in range(args.docs)]}
    lat, codes, lock, next_i = [], {}, threading.Lock(), [0]

    def run():
        while True:
            with lock:
                if next_i[0] >= args.load:
                    return
                next_i[0] += 1
            t = time.perf_counter()
            status, _ = post(f"{args.url}/v1/rerank", body, api_key=args.api_key)
            ms = (time.perf_counter() - t) * 1000
            with lock:
                codes[status] = codes.get(status, 0) + 1
                if status == 200:
                    lat.append(ms)

    t0 = time.perf_counter()
    ts = [threading.Thread(target=run) for _ in range(args.concurrency)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.perf_counter() - t0
    lat.sort()

    def pct(p):
        return round(lat[min(len(lat) - 1, int(p / 100 * (len(lat) - 1) + 0.5))], 2) if lat else None

    print(
        json.dumps(
            {
                "requests": args.load,
                "concurrency": args.concurrency,
                "docs": args.docs,
                "status": codes,
                "p50_ms": pct(50),
                "p90_ms": pct(90),
                "p99_ms": pct(99),
                "req_s": round(len(lat) / wall, 2),
                "pairs_s": round(len(lat) * args.docs / wall, 1),
            }
        )
    )
    return 0 if codes.get(200, 0) == args.load else 1


def main():
    ap = argparse.ArgumentParser(description="rerank server client: reference check and load test")
    ap.add_argument("--url", default="http://127.0.0.1:8111")
    ap.add_argument("--api-key", default=None, help="Bearer key (tt-media-server)")
    ap.add_argument("--wait-ready", type=float, default=0, help="seconds to wait for /health 200 first")
    ap.add_argument("--check", help="reference JSON from rerank_hf_ref.py")
    ap.add_argument("--min-pearson", type=float, default=0.9995)
    ap.add_argument("--load", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--docs", type=int, default=25)
    ap.add_argument("--words", type=int, default=60)
    args = ap.parse_args()
    if args.wait_ready and not wait_ready(args.url, args.wait_ready):
        print("server not ready", file=sys.stderr)
        return 2
    rc = 0
    if args.check:
        rc |= check(args)
    if args.load:
        rc |= load(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
