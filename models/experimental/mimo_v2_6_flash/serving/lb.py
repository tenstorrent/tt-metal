# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Least-connections HTTP proxy for N independent vLLM servers (one per Galaxy row): requests go to the backend with
the fewest in flight (ties: round robin), responses stream back unchanged.
GET /metrics concatenates every backend's metrics; GET /lb/stats shows the in-flight and served counts.
Usage: lb.py --port 8000 --backends 127.0.0.1:8001,127.0.0.1:8002,..."""
import argparse
import itertools

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "content-encoding", "upgrade", "host"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--backends", required=True)
    a = ap.parse_args()
    backends = [b if b.startswith("http") else f"http://{b}" for b in a.backends.split(",")]
    inflight = {b: 0 for b in backends}
    served = {b: 0 for b in backends}
    rr = itertools.count()
    state = {}

    def pick():
        low = min(inflight.values())
        cands = [b for b in backends if inflight[b] == low]
        return cands[next(rr) % len(cands)]

    async def proxy(request: web.Request):
        b = pick()
        inflight[b] += 1
        served[b] += 1
        try:
            body = await request.read()
            headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
            async with state["session"].request(
                request.method, b + request.path_qs, data=body or None, headers=headers
            ) as up:
                resp = web.StreamResponse(
                    status=up.status, headers={k: v for k, v in up.headers.items() if k.lower() not in HOP}
                )
                await resp.prepare(request)
                async for chunk in up.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
                return resp
        finally:
            inflight[b] -= 1

    async def stats(_):
        return web.json_response({"inflight": inflight, "served": served})

    async def all_metrics(_):
        # Every backend's Prometheus text, concatenated: a client summing a counter's samples sums the engines.
        texts = []
        for b in backends:
            async with state["session"].get(b + "/metrics") as up:
                texts.append(f"# backend {b}\n" + await up.text())
        return web.Response(text="".join(texts), content_type="text/plain")

    async def on_start(app):
        state["session"] = ClientSession(timeout=ClientTimeout(total=None), connector=TCPConnector(limit=0))

    async def on_stop(app):
        await state["session"].close()

    app = web.Application(client_max_size=1 << 30)
    app.on_startup.append(on_start)
    app.on_cleanup.append(on_stop)
    app.router.add_get("/lb/stats", stats)
    app.router.add_get("/metrics", all_metrics)
    app.router.add_route("*", "/{tail:.*}", proxy)
    web.run_app(app, host="127.0.0.1", port=a.port, access_log=None)


if __name__ == "__main__":
    main()
