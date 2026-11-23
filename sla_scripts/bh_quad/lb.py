#!/usr/bin/env python3
"""OpenAI-compatible load balancer over several vLLM servers (one Llama-3.3-70B server per BH Galaxy).

Clients talk to the balancer exactly as to one vLLM server (same paths, same JSON, SSE streaming passed through
byte for byte). Each backend serves at most --slots requests at a time (32 = one Galaxy's DP4 x max_num_seqs 8);
requests past the total wait in one FIFO queue and go to the first backend that frees a slot, so 4 Galaxies give
128 concurrent users with no backend over-subscribed.

Routing (generation requests): among healthy backends with a free slot, pick the one with the least outstanding
work = prompt tokens still waiting for their first output token (prefill backlog, sets TTFT) + --decode-weight
x running requests (decode batch, sets TPOT) + this prompt's own prefill. Requests that share a long prefix go to a backend that
saw it when that is cheaper: a backend's cost counts only the prompt tokens it would have to prefill beyond the
prefix it probably still caches (tracked by hashing the prompt at --prefix-levels lengths).

Failures: a backend that refuses connections or returns 5xx before sending any bytes is marked down and the
request is retried on another backend (--retries). A stream that already started cannot be replayed: the client
sees the connection end, same as with a direct server. Health: GET /health every --health-interval s; down
after 2 misses, up again on the first success. A client that disconnects closes the upstream connection, which
makes vLLM abort the request and free its slot.

Endpoints of the balancer itself: GET /health (200 if any backend is up), GET /lb/stats (JSON),
POST /lb/backend {"url": ..., "enabled": false|true} (drain / re-enable one backend).

usage: lb.py --backends http://10.82.97.43:8000,http://10.82.97.44:8000,... [--port 9000] [--slots 32]
"""
import argparse
import asyncio
import collections
import hashlib
import json
import logging
import time

from aiohttp import ClientError, ClientSession, ClientTimeout, TCPConnector, web

log = logging.getLogger("lb")
# hop-by-hop / framing headers that must not be copied between the two connections
HOP = {
    "host",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
}
GEN_PATHS = ("/v1/completions", "/v1/chat/completions", "/v1/responses", "/v1/embeddings", "/pooling", "/score")


class Backend:
    def __init__(self, url, slots, idx):
        self.url, self.slots, self.idx = url.rstrip("/"), slots, idx
        self.healthy = False
        self.enabled = True
        self.misses = 0
        self.inflight = 0
        self.pending = 0  # estimated prompt tokens of requests that have not produced their first byte yet
        self.served = self.errors = self.retried = 0
        self.ttft_ewma = None
        self.last_error = ""
        self.model_ids = []

    def usable(self):
        return self.healthy and self.enabled and self.inflight < self.slots

    def stats(self):
        return {
            "url": self.url,
            "healthy": self.healthy,
            "enabled": self.enabled,
            "inflight": self.inflight,
            "slots": self.slots,
            "pending_prefill_tokens": self.pending,
            "served": self.served,
            "errors": self.errors,
            "retried_away": self.retried,
            "ttft_ewma_s": None if self.ttft_ewma is None else round(self.ttft_ewma, 4),
            "last_error": self.last_error,
            "models": self.model_ids,
        }


def estimate_tokens(data):
    """Rough prompt-token count from the request JSON (4 chars per token for text, exact for token-id lists)."""
    if not isinstance(data, dict):
        return 0
    p = data.get("prompt")
    if isinstance(p, str):
        return len(p) // 4 + 1
    if isinstance(p, list):
        if p and isinstance(p[0], int):
            return len(p)
        return sum(len(x) if isinstance(x, list) else len(str(x)) // 4 + 1 for x in p)
    msgs = data.get("messages") or data.get("input")
    if isinstance(msgs, list):
        n = 0
        for m in msgs:
            c = m.get("content") if isinstance(m, dict) else m
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            n += len(str(c or "")) // 4 + 4
        return n
    if isinstance(msgs, str):
        return len(msgs) // 4 + 1
    return 0


def prefix_keys(data, levels):
    """[(chars, hash)] of the prompt's first `chars` characters for each level the prompt is long enough for.

    A backend that served a request with the same hash at a level likely still holds those tokens in its prefix
    cache. Token-id prompts count 4 chars per token. Chat requests hash the JSON of the message list, so a
    conversation's earlier turns form a prefix of its later requests.
    """
    if not levels or not isinstance(data, dict):
        return []
    p = data.get("prompt")
    if isinstance(p, list) and p and isinstance(p[0], int):
        text = ",".join(map(str, p[: max(levels) // 4]))
        n = len(p) * 4
    elif isinstance(p, str):
        text, n = p, len(p)
    elif isinstance(data.get("messages"), list):
        text = json.dumps(data["messages"], ensure_ascii=False)
        n = len(text)
    else:
        return []
    if isinstance(p, list):
        # token ids: cut on whole ids, one level per 4*k tokens
        ids = p[: max(levels) // 4]
        return [
            (lv, hashlib.blake2b(",".join(map(str, ids[: lv // 4])).encode(), digest_size=12).digest())
            for lv in levels
            if lv <= n
        ]
    return [
        (lv, hashlib.blake2b(text[:lv].encode("utf-8", "surrogatepass"), digest_size=12).digest())
        for lv in levels
        if lv <= n
    ]


class Balancer:
    def __init__(self, args):
        self.a = args
        self.backends = [Backend(u, args.slots, i) for i, u in enumerate(args.backends.split(","))]
        self.queue = collections.deque()  # [future, est, keys, excluded] in arrival order
        self.affinity = collections.OrderedDict()  # prefix hash -> {backend idx} that served it (LRU)
        self.levels = sorted(int(x) for x in args.prefix_levels.split(",") if x.strip() and int(x) > 0)
        self.session = None
        self.rr = 0
        self.t0 = time.time()
        self.total = self.failed = self.queued_max = 0

    # ---- slot accounting -------------------------------------------------------------------------------------
    def score(self, b):
        return b.pending + self.a.decode_weight * b.inflight

    def cached_tokens(self, b, keys):
        best = 0
        for lv, h in keys:
            if b.idx in self.affinity.get(h, ()):
                best = lv // 4
        return best

    def pick(self, est, keys, excluded):
        """(backend, prefill tokens it will actually compute) or None if no slot is free.

        cost = backlog already queued on the backend + this prompt's tokens it would have to prefill (minus what it
        probably has cached). Lowest cost wins; this spreads bursts, steers away from backends busy with long
        prefills, and keeps conversations on the backend holding their history when that is actually cheaper.
        """
        cands = [b for b in self.backends if b.usable() and b.idx not in excluded]
        if not cands:
            return None
        n = len(self.backends)
        eff = {b.idx: max(est - self.cached_tokens(b, keys), 0) for b in cands} if keys else {b.idx: est for b in cands}
        # ties (e.g. all idle) rotate so bursts spread evenly instead of piling onto backend 0
        best = min(cands, key=lambda b: (self.score(b) + eff[b.idx], b.inflight, (b.idx - self.rr) % n))
        for _, h in keys:
            s = self.affinity.get(h)
            if s is None:
                s = self.affinity[h] = set()
            s.add(best.idx)
            self.affinity.move_to_end(h)
        while len(self.affinity) > self.a.prefix_entries:
            self.affinity.popitem(last=False)
        self.rr = (best.idx + 1) % n
        return best, eff[best.idx]

    def reserve(self, b, est):
        b.inflight += 1
        b.pending += est

    def first_byte(self, b, est):
        b.pending -= est

    def release(self, b, est, got_first):
        b.inflight -= 1
        if not got_first:
            b.pending -= est
        self.drain()

    def drain(self):
        while self.queue:
            fut, est, keys, excluded = self.queue[0]
            if fut.done():
                self.queue.popleft()
                continue
            got = self.pick(est, keys, excluded)
            if got is None:
                return
            self.queue.popleft()
            self.reserve(*got)
            fut.set_result(got)

    async def acquire(self, est, keys, excluded):
        """Wait (FIFO) for a slot; returns (backend, prefill tokens reserved on it)."""
        if not self.queue:
            got = self.pick(est, keys, excluded)
            if got is not None:
                self.reserve(*got)
                return got
        fut = asyncio.get_running_loop().create_future()
        item = [fut, est, keys, excluded]
        self.queue.append(item)
        self.queued_max = max(self.queued_max, len(self.queue))
        try:
            return await asyncio.wait_for(fut, self.a.queue_timeout) if self.a.queue_timeout else await fut
        except BaseException:
            # cancelled (client gone) or timed out: give back a slot that was handed over at the same moment
            if fut.done() and not fut.cancelled() and fut.exception() is None:
                self.release(*fut.result(), False)
            try:
                self.queue.remove(item)
            except ValueError:
                pass
            raise

    # ---- health ----------------------------------------------------------------------------------------------
    def mark_down(self, b, why):
        b.last_error = why
        if b.healthy:
            log.warning("backend %s DOWN: %s", b.url, why)
        b.healthy = False

    async def check(self, b):
        try:
            async with self.session.get(b.url + "/health", timeout=ClientTimeout(total=self.a.health_timeout)) as r:
                ok = r.status == 200
                await r.read()
        except Exception as e:  # noqa: BLE001 - any failure is a miss
            ok, why = False, f"health: {type(e).__name__} {e}"
        else:
            why = f"health: HTTP {r.status}"
        if ok:
            b.misses = 0
            if not b.healthy:
                await self.fetch_models(b)
                log.warning("backend %s UP models=%s", b.url, b.model_ids)
                b.healthy = True
                self.drain()
        else:
            b.misses += 1
            if b.misses >= 2 or not b.healthy:
                self.mark_down(b, why)

    async def fetch_models(self, b):
        try:
            async with self.session.get(b.url + "/v1/models", timeout=ClientTimeout(total=10)) as r:
                b.model_ids = [m["id"] for m in (await r.json())["data"]]
        except Exception as e:  # noqa: BLE001
            b.model_ids = [f"? {type(e).__name__}"]
        ids = {tuple(x.model_ids) for x in self.backends if x.model_ids and x is not b} | {tuple(b.model_ids)}
        if len(ids) > 1:
            log.error("backends serve different models: %s", ids)

    async def health_loop(self):
        while True:
            await asyncio.gather(*(self.check(b) for b in self.backends))
            await asyncio.sleep(self.a.health_interval)

    # ---- proxy -----------------------------------------------------------------------------------------------
    @staticmethod
    def fwd_headers(h):
        return {k: v for k, v in h.items() if k.lower() not in HOP}

    async def light(self, request):
        """Non-generation requests (models, tokenize, version...): any healthy backend, no slot."""
        body = await request.read()
        order = sorted(self.backends, key=lambda b: (not (b.healthy and b.enabled), b.inflight))
        for b in order:
            try:
                async with self.session.request(
                    request.method,
                    b.url + str(request.rel_url),
                    data=body or None,
                    headers=self.fwd_headers(request.headers),
                ) as up:
                    data = await up.read()
                    return web.Response(status=up.status, body=data, headers=self.fwd_headers(up.headers))
            except (ClientError, asyncio.TimeoutError, OSError) as e:
                self.mark_down(b, f"{type(e).__name__} {e}")
        return web.json_response({"error": "no backend available"}, status=503)

    async def handle(self, request):
        path = request.path
        if request.method != "POST" or not path.startswith(GEN_PATHS):
            return await self.light(request)
        body = await request.read()
        try:
            data = json.loads(body) if body else None
        except ValueError:
            data = None  # let the backend produce the 400
        est, keys = estimate_tokens(data), prefix_keys(data, self.levels)
        self.total += 1
        excluded = set()
        headers = self.fwd_headers(request.headers)
        url_tail = str(request.rel_url)
        for attempt in range(self.a.retries + 1):
            b, eff = await self.acquire(est, keys, excluded)
            t0 = time.perf_counter()
            got_first = False
            resp = None
            try:
                async with self.session.post(b.url + url_tail, data=body, headers=headers) as up:
                    if up.status >= 500 and attempt < self.a.retries:
                        b.errors += 1
                        b.retried += 1
                        excluded.add(b.idx)
                        self.mark_down(b, f"HTTP {up.status}: {(await up.read())[:200]!r}")
                        continue
                    resp = web.StreamResponse(status=up.status, reason=up.reason, headers=self.fwd_headers(up.headers))
                    it = up.content.iter_any()
                    while True:
                        try:
                            chunk = await it.__anext__()
                        except StopAsyncIteration:
                            break
                        if not got_first:
                            got_first = True
                            self.first_byte(b, eff)
                            dt = time.perf_counter() - t0
                            b.ttft_ewma = dt if b.ttft_ewma is None else 0.9 * b.ttft_ewma + 0.1 * dt
                            await resp.prepare(request)
                        await resp.write(chunk)  # raises if the client went away -> upstream closed below
                    if not got_first:
                        await resp.prepare(request)
                    await resp.write_eof()
                    b.served += 1
                    return resp
            except (ClientError, asyncio.TimeoutError, OSError) as e:
                client_gone = request.transport is None or request.transport.is_closing()
                if client_gone:
                    raise
                b.errors += 1
                if resp is None or not resp.prepared:
                    self.mark_down(b, f"{type(e).__name__} {e}")
                    if attempt < self.a.retries:
                        b.retried += 1
                        excluded.add(b.idx)
                        continue
                    self.failed += 1
                    return web.json_response({"error": f"backend {b.url} failed: {type(e).__name__} {e}"}, status=502)
                # mid-stream backend failure: cannot replay what the client already received
                self.failed += 1
                log.warning("stream from %s broke after first byte: %s %s", b.url, type(e).__name__, e)
                self.mark_down(b, f"mid-stream {type(e).__name__} {e}")
                # cut the client connection without the final chunk, so the client sees an error exactly as if it
                # had been talking to the dead server (a clean EOF would look like a complete, short answer)
                if request.transport is not None:
                    request.transport.abort() if hasattr(request.transport, "abort") else request.transport.close()
                return resp
            finally:
                self.release(b, eff, got_first)
        self.failed += 1
        return web.json_response({"error": "all retries failed"}, status=502)

    # ---- admin -----------------------------------------------------------------------------------------------
    async def health(self, request):
        ok = any(b.healthy and b.enabled for b in self.backends)
        return web.json_response({"ok": ok}, status=200 if ok else 503)

    async def stats(self, request):
        return web.json_response(
            {
                "uptime_s": round(time.time() - self.t0, 1),
                "requests": self.total,
                "failed": self.failed,
                "queued": len(self.queue),
                "queued_max": self.queued_max,
                "inflight": sum(b.inflight for b in self.backends),
                "capacity": sum(b.slots for b in self.backends if b.healthy and b.enabled),
                "backends": [b.stats() for b in self.backends],
            }
        )

    async def set_backend(self, request):
        d = await request.json()
        for b in self.backends:
            if b.url == d["url"].rstrip("/"):
                b.enabled = bool(d.get("enabled", True))
                if "slots" in d:
                    b.slots = int(d["slots"])
                self.drain()
                return web.json_response(b.stats())
        return web.json_response({"error": "unknown backend"}, status=404)

    async def on_startup(self, app):
        self.session = ClientSession(
            connector=TCPConnector(limit=0, ttl_dns_cache=300, keepalive_timeout=60),
            timeout=ClientTimeout(total=None, connect=self.a.connect_timeout, sock_connect=self.a.connect_timeout),
            auto_decompress=False,
            read_bufsize=1 << 20,
        )
        await asyncio.gather(*(self.check(b) for b in self.backends))
        app["health_task"] = asyncio.create_task(self.health_loop())

    async def on_cleanup(self, app):
        app["health_task"].cancel()
        await self.session.close()

    def app(self):
        app = web.Application(client_max_size=1 << 28)
        app.router.add_get("/health", self.health)
        app.router.add_get("/lb/stats", self.stats)
        app.router.add_post("/lb/backend", self.set_backend)
        app.router.add_route("*", "/{tail:.*}", self.handle)
        app.on_startup.append(self.on_startup)
        app.on_cleanup.append(self.on_cleanup)
        return app


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backends", required=True, help="comma-separated base URLs")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--slots", type=int, default=32, help="max concurrent requests per backend")
    p.add_argument("--decode-weight", type=float, default=256.0, help="tokens of backlog one running request counts as")
    p.add_argument(
        "--prefix-levels",
        default="2048,8192,32768,131072",
        help="prompt prefix lengths (chars) tracked for prefix-cache-aware routing ('' = off)",
    )
    p.add_argument("--prefix-entries", type=int, default=100000)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--queue-timeout", type=float, default=0, help="seconds a request may wait for a slot (0 = forever)")
    p.add_argument("--health-interval", type=float, default=2.0)
    p.add_argument("--health-timeout", type=float, default=5.0)
    p.add_argument("--connect-timeout", type=float, default=10.0)
    p.add_argument("--log-level", default="WARNING")
    return p.parse_args(argv)


def main():
    a = parse()
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(message)s")
    try:
        import uvloop

        uvloop.install()
    except ImportError:
        pass
    web.run_app(Balancer(a).app(), host=a.host, port=a.port, access_log=None, backlog=4096, print=None)


if __name__ == "__main__":
    main()
