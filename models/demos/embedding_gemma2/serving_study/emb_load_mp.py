"""Closed-loop load on /v1/embeddings from several client processes (asyncio + aiohttp), for DP servers that outrun
the one-process threaded emb_load.py.

C connections in total, split over P processes; each connection sends its next request as soon as the last reply
lands. A request carries `--texts` strings (1 = single-text requests, like emb_load.py). Only requests that start
and finish inside the measured window count.

Usage: emb_load_mp.py URL OUT.json --tokens 128 --conc 32 256 --texts 1 --procs 8 [--seconds 30 --warmup 5]
Prints one JSON row per concurrency: req/s, texts/s, tokens/s, p50/p90/p99 ms, errors.
"""

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import time


def _client_proc(url, bodies, n_conn, t_start, t_end, out_q, seed):
    import aiohttp

    try:
        import uvloop

        uvloop.install()
    except ImportError:
        pass

    async def run():
        lat, errs, n_ok = [], 0, 0
        conn = aiohttp.TCPConnector(limit=0, force_close=False)
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as sess:

            async def one(i):
                nonlocal errs, n_ok
                k = seed * 7919 + i
                while time.time() < t_end:
                    body = bodies[k % len(bodies)]
                    k += 7
                    a = time.time()
                    try:
                        async with sess.post(url, data=body, headers={"Content-Type": "application/json"}) as r:
                            await r.read()
                            if r.status != 200:
                                raise RuntimeError(r.status)
                    except Exception:
                        errs += 1
                        await asyncio.sleep(0.01)
                        continue
                    b = time.time()
                    if a >= t_start and b <= t_end:
                        lat.append((b - a) * 1e3)

            await asyncio.gather(*(one(i) for i in range(n_conn)))
        return lat, errs

    lat, errs = asyncio.run(run())
    out_q.put((lat, errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--conc", type=int, nargs="+", default=[32])
    ap.add_argument("--texts", type=int, default=1, help="strings per request")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=float(os.environ.get("EMB_LOAD_SECONDS", "30")))
    ap.add_argument("--warmup", type=float, default=float(os.environ.get("EMB_LOAD_WARMUP_S", "5")))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    from emb_common import MODEL_ID, load_tokenizer, make_texts

    texts = make_texts(load_tokenizer(), a.tokens, 64)
    bodies = []
    for i in range(64):
        inp = texts[i] if a.texts == 1 else [texts[(i + j) % 64] for j in range(a.texts)]
        bodies.append(json.dumps({"model": MODEL_ID, "input": inp}).encode())
    url = a.url.rstrip("/") + "/v1/embeddings"

    rows = []
    ctx = mp.get_context("fork")
    for c in a.conc:
        procs = min(a.procs, c)
        per = [c // procs + (1 if i < c % procs else 0) for i in range(procs)]
        t0 = time.time() + 1.0  # let every process start before the clock
        t_start, t_end = t0 + a.warmup, t0 + a.warmup + a.seconds
        q = ctx.Queue()
        ps = [ctx.Process(target=_client_proc, args=(url, bodies, per[i], t_start, t_end, q, i)) for i in range(procs)]
        for p in ps:
            p.start()
        lat, errs = [], 0
        for _ in ps:
            l, e = q.get()
            lat += l
            errs += e
        for p in ps:
            p.join()
        lat.sort()
        pct = lambda p: round(lat[min(len(lat) - 1, int(len(lat) * p))], 2) if lat else None
        rps = len(lat) / a.seconds
        row = dict(
            tag=a.tag,
            tokens=a.tokens,
            texts_per_req=a.texts,
            conc=c,
            procs=procs,
            seconds=a.seconds,
            n=len(lat),
            req_s=round(rps, 1),
            texts_s=round(rps * a.texts, 1),
            tokens_s=round(rps * a.texts * a.tokens),
            p50_ms=pct(0.5),
            p90_ms=pct(0.9),
            p99_ms=pct(0.99),
            errors=errs,
        )
        print(json.dumps(row), flush=True)
        rows.append(row)
    json.dump(dict(url=a.url, rows=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
