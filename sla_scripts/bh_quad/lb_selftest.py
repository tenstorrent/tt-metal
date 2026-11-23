#!/usr/bin/env python3
"""Self-test of lb.py against fake vLLM backends (CPU only, ~1 min). Exits 1 on any failure.

Fake backend: /v1/completions and /v1/chat/completions sleep prompt_len * PREFILL_S_PER_CHAR, then emit max_tokens
SSE chunks (or one JSON body) whose text encodes (backend, request id, token index), so the client can check every
byte arrived in order from one backend. It records its concurrency high-water mark and requests the client aborted.
"""
import asyncio
import collections
import json
import random
import sys
import time

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import lb  # noqa: E402

PREFILL_S_PER_CHAR = 2e-6
TOK_S = 0.004
FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


class Fake:
    def __init__(self, name, port):
        self.name, self.port = name, port
        self.cur = self.hwm = self.count = self.aborted = 0
        self.fail_5xx = 0
        self.die_mid_stream = False
        self.runner = None

    async def gen(self, request):
        body = await request.json()
        if self.fail_5xx:
            self.fail_5xx -= 1
            return web.json_response({"error": "boom"}, status=503)
        if body.get("max_tokens", 1) < 0:
            return web.json_response({"error": "bad max_tokens"}, status=400)
        prompt = body.get("prompt") or json.dumps(body.get("messages"))
        rid = body.get("user", "?")
        self.cur += 1
        self.count += 1
        self.hwm = max(self.hwm, self.cur)
        try:
            await asyncio.sleep(len(prompt) * PREFILL_S_PER_CHAR)
            n = body.get("max_tokens", 4)
            if not body.get("stream"):
                await asyncio.sleep(n * TOK_S)
                text = "".join(f"<{self.name}:{rid}:{i}>" for i in range(n))
                return web.json_response({"choices": [{"text": text}], "backend": self.name})
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            for i in range(n):
                if self.die_mid_stream and i == 2:
                    request.transport.close()
                    return resp
                await resp.write(
                    f"data: {json.dumps({'choices': [{'text': f'<{self.name}:{rid}:{i}>'}]})}\n\n".encode()
                )
                await asyncio.sleep(TOK_S)
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        except (asyncio.CancelledError, ConnectionResetError):
            self.aborted += 1
            raise
        finally:
            self.cur -= 1

    async def health(self, request):
        return web.Response(text="")

    async def models(self, request):
        return web.json_response({"object": "list", "data": [{"id": "llama70b", "object": "model"}]})

    async def start(self):
        app = web.Application()
        app.router.add_post("/v1/completions", self.gen)
        app.router.add_post("/v1/chat/completions", self.gen)
        app.router.add_get("/health", self.health)
        app.router.add_get("/v1/models", self.models)
        self.runner = web.AppRunner(app, handler_cancellation=True)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", self.port).start()

    async def stop(self):
        await self.runner.cleanup()
        self.runner = None

    def reset(self):
        self.hwm = self.count = self.aborted = 0


async def stream(s, url, rid, prompt="x" * 4000, n=20, chat=False):
    if chat:
        body = {"model": "llama70b", "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "stream": True}
        path = "/v1/chat/completions"
    else:
        body = {"model": "llama70b", "prompt": prompt, "max_tokens": n, "stream": True}
        path = "/v1/completions"
    body["user"] = str(rid)
    t0 = time.perf_counter()
    ttft = None
    texts = []
    async with s.post(url + path, json=body) as r:
        if r.status != 200:
            return {"status": r.status, "body": await r.text()}
        buf = b""
        async for chunk in r.content.iter_any():
            if ttft is None:
                ttft = time.perf_counter() - t0
            buf += chunk
        for line in buf.decode().split("\n\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                texts.append(json.loads(line[6:])["choices"][0]["text"])
        done = buf.decode().rstrip().endswith("data: [DONE]")
    toks = [t.strip("<>").split(":") for t in texts]
    backends = {t[0] for t in toks}
    ok = (
        done
        and len(toks) == n
        and all(t[1] == str(rid) and int(t[2]) == i for i, t in enumerate(toks))
        and len(backends) == 1
    )
    return {"status": 200, "ok": ok, "backend": backends.pop() if backends else None, "ttft": ttft}


async def main():
    fakes = [Fake(f"b{i}", 18100 + i) for i in range(4)]
    for f in fakes:
        await f.start()
    args = lb.parse(
        [
            "--backends",
            ",".join(f"http://127.0.0.1:{f.port}" for f in fakes),
            "--port",
            "18099",
            "--health-interval",
            "0.3",
        ]
    )
    bal = lb.Balancer(args)
    runner = web.AppRunner(bal.app(), handler_cancellation=True)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 18099).start()
    url = "http://127.0.0.1:18099"
    s = ClientSession(timeout=ClientTimeout(total=120), connector=TCPConnector(limit=0))

    # 1. 128 concurrent streams: exactly 32 per backend, all byte-exact
    res = await asyncio.gather(*(stream(s, url, i, n=30) for i in range(128)))
    per = {f.name: sum(1 for r in res if r.get("backend") == f.name) for f in fakes}
    check(
        all(r.get("ok") for r in res),
        f"128 concurrent streams all complete and in order ({sum(r.get('ok', 0) for r in res)}/128)",
    )
    check(all(v == 32 for v in per.values()), f"128 concurrent split 32/32/32/32: {per}")
    check(all(f.hwm <= 32 for f in fakes), f"no backend over 32 concurrent: {[f.hwm for f in fakes]}")

    # 2. 300 concurrent (over capacity): queue, never > 32 per backend, all complete, FIFO-ish start order
    for f in fakes:
        f.reset()
    res = await asyncio.gather(*(stream(s, url, i, n=10) for i in range(300)))
    check(all(r.get("ok") for r in res), "300 concurrent (2.3x capacity) all complete via the queue")
    check(all(f.hwm <= 32 for f in fakes), f"over capacity: per-backend high-water {[f.hwm for f in fakes]} <= 32")
    st = await (await s.get(url + "/lb/stats")).json()
    check(
        st["queued_max"] >= 100 and st["inflight"] == 0 and st["queued"] == 0,
        f"queue drained: max {st['queued_max']}, now {st['queued']}, inflight {st['inflight']}",
    )
    check(all(b["pending_prefill_tokens"] == 0 for b in st["backends"]), "pending-prefill accounting returns to 0")

    # 3. chat + non-streaming JSON passthrough; 4xx passes through without retry
    r = await stream(s, url, 7, chat=True, n=5)
    check(r.get("ok"), "chat completions stream")
    async with s.post(url + "/v1/completions", json={"prompt": "hi", "max_tokens": 3, "user": "j"}) as rr:
        j = await rr.json()
        check(rr.status == 200 and j["choices"][0]["text"].count("<") == 3, "non-streaming JSON passthrough")
    before = sum(f.count for f in fakes)
    async with s.post(url + "/v1/completions", json={"prompt": "hi", "max_tokens": -1}) as rr:
        check(rr.status == 400 and sum(f.count for f in fakes) == before, "4xx passed through, not retried")
    async with s.get(url + "/v1/models") as rr:
        check(rr.status == 200 and (await rr.json())["data"][0]["id"] == "llama70b", "/v1/models proxied")

    # 4. prefill-aware routing: a long prompt in flight on one backend steers the next requests elsewhere
    for f in fakes:
        f.reset()
    long_task = asyncio.create_task(stream(s, url, 900, prompt="L" * 400000, n=2))
    await asyncio.sleep(0.05)
    busy = [f.name for f in fakes if f.cur]
    res = await asyncio.gather(*(stream(s, url, 901 + i, n=2) for i in range(6)))
    check(
        len(busy) == 1 and all(r["backend"] != busy[0] for r in res),
        f"short requests avoid the backend prefilling 100K tokens ({busy})",
    )
    await long_task

    # 5. prefix affinity: same 8K-char prefix -> same backend while load is balanced; identical prompts in a burst
    #    still spread (the cache discount is worth less than a full backend's backlog)
    pre = "".join(random.choice("abcdefgh ") for _ in range(8000))
    res = [await stream(s, url, 950 + i, prompt=pre + f" q{i}", n=2) for i in range(8)]
    check(
        len({r["backend"] for r in res}) == 1,
        f"shared-prefix requests stick to one backend: {[r['backend'] for r in res]}",
    )
    res = await asyncio.gather(*(stream(s, url, 970 + i, prompt=pre, n=30) for i in range(8)))
    per = collections.Counter(r["backend"] for r in res)
    check(max(per.values()) <= 3, f"burst of 8 identical prompts spreads over backends: {dict(per)}")

    # 6. 5xx before any byte -> retried elsewhere, client sees success
    for f in fakes:
        f.reset()
    fakes[0].fail_5xx = 1000
    await asyncio.sleep(0.4)
    res = await asyncio.gather(*(stream(s, url, 1000 + i, n=5) for i in range(40)))
    check(all(r.get("ok") for r in res), "backend returning 503: all 40 requests succeed via retry")
    check(all(r["backend"] != "b0" for r in res), "no response came from the failing backend")
    fakes[0].fail_5xx = 0
    await asyncio.sleep(1.0)
    st = await (await s.get(url + "/lb/stats")).json()
    check(st["backends"][0]["healthy"], "backend marked up again after health checks pass")

    # 7. backend process dies: new requests go elsewhere; comes back -> used again
    await fakes[1].stop()
    res = await asyncio.gather(*(stream(s, url, 1100 + i, n=5) for i in range(40)))
    check(all(r.get("ok") and r["backend"] != "b1" for r in res), "dead backend: 40 requests all served by the others")
    await fakes[1].start()
    await asyncio.sleep(1.0)
    res = await asyncio.gather(*(stream(s, url, 1200 + i, n=20) for i in range(128)))
    check(
        all(r.get("ok") for r in res) and sum(r["backend"] == "b1" for r in res) == 32,
        "restarted backend rejoins (gets 32 of 128)",
    )

    # 8. mid-stream backend failure: the client sees a truncated stream (not a silent splice); others unaffected
    fakes[2].die_mid_stream = True
    c0 = fakes[2].count
    res = await asyncio.gather(
        *(stream(s, url, 1300 + i, prompt=f"{i} " + "y" * 4000, n=10) for i in range(8)), return_exceptions=True
    )
    fakes[2].die_mid_stream = False
    print(
        "INFO mid-stream batch:",
        fakes[2].count - c0,
        "sent to b2;",
        [type(r).__name__ if isinstance(r, Exception) else (r.get("backend"), r.get("ok")) for r in res],
    )
    bad = [r for r in res if isinstance(r, Exception) or not r.get("ok")]
    check(
        len(bad) >= 1 and all(isinstance(r, Exception) for r in bad),
        f"mid-stream backend death -> client gets an error, never a clean short answer ({len(bad)} errored: {set(type(r).__name__ for r in bad)})",
    )
    check(all(r.get("ok") for r in res if not isinstance(r, Exception)), "other streams in that batch are complete")
    await asyncio.sleep(1.0)

    # 9. client disconnect mid-stream aborts the upstream request and frees the slot
    for f in fakes:
        f.reset()

    async def abandon(i):
        async with s.post(
            url + "/v1/completions", json={"prompt": "x", "max_tokens": 500, "stream": True, "user": str(i)}
        ) as r:
            await r.content.readany()
        # leaving the context closes the connection

    await asyncio.gather(*(abandon(2000 + i) for i in range(16)))
    await asyncio.sleep(0.5)
    st = await (await s.get(url + "/lb/stats")).json()
    check(
        sum(f.aborted for f in fakes) == 16, f"16 abandoned streams aborted upstream ({sum(f.aborted for f in fakes)})"
    )
    check(st["inflight"] == 0 and all(f.cur == 0 for f in fakes), "slots freed after client disconnects")

    # 10. drain a backend via admin API
    async with s.post(url + "/lb/backend", json={"url": f"http://127.0.0.1:{fakes[3].port}", "enabled": False}) as rr:
        assert rr.status == 200
    res = await asyncio.gather(*(stream(s, url, 3000 + i, n=3) for i in range(30)))
    check(all(r.get("ok") and r["backend"] != "b3" for r in res), "disabled backend gets no traffic")
    await s.post(url + "/lb/backend", json={"url": f"http://127.0.0.1:{fakes[3].port}", "enabled": True})

    # 11. overhead: 1000 tiny streams through the LB vs direct
    async def burst(u):
        t = time.perf_counter()
        await asyncio.gather(*(stream(s, u, 5000 + i, prompt="x", n=1) for i in range(1000)))
        return time.perf_counter() - t

    direct = await burst(f"http://127.0.0.1:{fakes[0].port}")
    via = await burst(url)
    print(f"INFO 1000 tiny requests: direct one backend {direct:.2f}s, via LB over 4 {via:.2f}s")

    await s.close()
    await runner.cleanup()
    for f in fakes:
        if f.runner:
            await f.stop()
    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
