# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""HTTP proxy for N independent vLLM servers (one per Galaxy row), prefix-cache aware.

Each engine has its own prefix cache, so a request only hits on the engine that already holds its prefix. The proxy
keys each request by its prompt (chat messages and tools, a completion prompt or its token ids, ``cache_salt``) in
``--chunk-chars`` chunks and remembers which engines served which chunks (an LRU index of chained chunk hashes):

- a request whose longest known prefix is at least ``--min-match-chars`` long goes to the least loaded engine that
  holds it, unless that engine has more than ``--slack`` requests in flight above the least loaded engine;
- any other request goes to the least loaded engine (fewest in flight; ties round robin).

So requests sharing a long prefix (a document, a long system prompt, a multi-turn conversation) gather on the engines
that cached it, and spill to another engine (which then caches it too) instead of queueing on one. Responses stream
back unchanged.

GET /health is 200 only when every engine answers its own /health. GET /metrics merges every engine's metrics into one
exposition, each sample labeled with its engine's index (``merge_metrics``).
GET /lb/stats shows in-flight and served counts and the prefix routing counters. A request whose engine refuses the
connection goes to another engine (the refusing one is skipped for ``--down-s`` seconds).

Usage: lb.py --port 8000 --backends 127.0.0.1:8001,127.0.0.1:8002,...
"""
import argparse
import hashlib
import itertools
import json
import re
import time
from collections import OrderedDict

from aiohttp import ClientConnectorError, ClientSession, ClientTimeout, TCPConnector, web

HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "content-encoding", "upgrade", "host"}
PREFIX_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/responses")
SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?(\s.*)$")  # name, labels, value [timestamp]
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def prefix_text(path, body):
    """The text that determines a request's prompt prefix, in prompt order, or None (no prefix routing)."""
    try:
        req = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(req, dict):
        return None
    head = json.dumps(
        {
            "model": req.get("model"),
            "salt": req.get("cache_salt"),
            "tools": req.get("tools"),
            "template": req.get("chat_template_kwargs"),
        },
        sort_keys=True,
    )
    if path == "/v1/chat/completions":
        messages = req.get("messages")
        if not isinstance(messages, list):
            return None
        return head + "".join(json.dumps(m, sort_keys=True, ensure_ascii=False) for m in messages)
    if path == "/v1/completions":
        prompt = req.get("prompt")
        if isinstance(prompt, list) and len(prompt) == 1:
            prompt = prompt[0]  # one prompt in a list
        if isinstance(prompt, str):
            return head + prompt
        if isinstance(prompt, list) and all(isinstance(t, int) for t in prompt):
            return head + ",".join(map(str, prompt)) + ","
        return None  # a batch of prompts: no single prefix
    if path == "/v1/responses":
        inp = req.get("input")
        text = inp if isinstance(inp, str) else json.dumps(inp, sort_keys=True, ensure_ascii=False)
        return head + json.dumps(req.get("instructions")) + text
    return None


def chunk_hashes(text, chunk_chars):
    """Chained hashes of the text's full chunks: hash i covers chunks 0..i (a partial last chunk is not keyed)."""
    out, h = [], b""
    data = text.encode("utf-8", "surrogatepass")
    step = chunk_chars  # bytes; equal to characters for ASCII prompts
    for start in range(0, len(data) - step + 1, step):
        h = hashlib.blake2b(h + data[start : start + step], digest_size=16).digest()
        out.append(h)
    return out


def merge_metrics(texts):
    """The engines' Prometheus texts (in backend order) as one exposition.

    Each engine is its own vLLM server, so all of them label their samples ``engine="0"``: concatenated, every series
    and every family's HELP / TYPE would appear once per engine, which a Prometheus parser rejects or collapses. Here
    every sample gets ``engine="<index of its engine>"``, as vLLM labels the engines of one data-parallel server, and
    each family is written once: HELP / TYPE from the first engine that has them, then every engine's samples."""
    families = OrderedDict()  # family name -> {"HELP": line, "TYPE": line, "samples": [lines]}
    for index, text in enumerate(texts):
        family = None
        for line in text.splitlines():
            if line.startswith("#"):
                parts = line.split(None, 3)
                if len(parts) >= 3 and parts[1] in ("HELP", "TYPE"):
                    family = parts[2]
                    families.setdefault(family, {"samples": []}).setdefault(parts[1], line)
                continue
            match = SAMPLE.match(line)
            if match is None:
                continue
            name, labels, rest = match.groups()
            if family is None or not (name == family or name.startswith(family + "_")):
                family = name  # an untyped sample: its own family
            labels = [(k, v) for k, v in LABEL.findall(labels or "") if k != "engine"] + [("engine", str(index))]
            sample = name + "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}" + rest
            families.setdefault(family, {"samples": []})["samples"].append(sample)
    out = []
    for lines in families.values():
        out += [lines[key] for key in ("HELP", "TYPE") if key in lines]
        out += lines["samples"]
    return "\n".join(out) + "\n"


class PrefixIndex:
    """Chunk hash -> the engines that served a prompt with that prefix, as an LRU of at most ``capacity`` hashes. A
    request's hashes are touched longest first, so eviction drops the tails of old prompts before their heads (as an
    engine's prefix cache does) and every prefix of a kept hash stays kept."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.entries = OrderedDict()

    def match(self, hashes, backends):
        """{backend: matched chunks} for the backends holding at least the first chunk."""
        alive = set(backends)
        lengths = {}
        for i, h in enumerate(hashes):
            holders = self.entries.get(h)
            now = alive & holders if holders else set()
            for b in alive - now:
                if i:
                    lengths[b] = i
            alive = now
            if not alive:
                break
        for b in alive:
            lengths[b] = len(hashes)
        return lengths

    def record(self, hashes, backend):
        for h in reversed(hashes):
            holders = self.entries.pop(h, None) or set()
            holders.add(backend)
            self.entries[h] = holders
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)


class Router:
    def __init__(self, backends, chunk_chars=256, min_match_chars=1024, slack=2, capacity=262144, down_s=5.0):
        self.backends = list(backends)
        self.chunk_chars = chunk_chars
        self.min_match_chunks = max(1, -(-min_match_chars // chunk_chars))
        self.slack = slack
        self.down_s = down_s
        self.index = PrefixIndex(capacity)
        self.inflight = {b: 0 for b in self.backends}
        self.served = {b: 0 for b in self.backends}
        self.down_until = {b: 0.0 for b in self.backends}
        self.rr = itertools.count()
        self.stats = {"prefix_routed": 0, "prefix_spilled": 0, "least_loaded": 0, "matched_chars": 0, "retried": 0}

    def up(self, exclude=()):
        now = time.monotonic()
        ups = [b for b in self.backends if b not in exclude and self.down_until[b] <= now]
        return ups or [b for b in self.backends if b not in exclude]

    def least_loaded(self, cands):
        low = min(self.inflight[b] for b in cands)
        ties = [b for b in cands if self.inflight[b] == low]
        return ties[next(self.rr) % len(ties)]

    def pick(self, hashes, exclude=()):
        cands = self.up(exclude)
        if not cands:
            return None
        if hashes:
            lengths = self.index.match(hashes, cands)
            best = max(lengths.values(), default=0)
            if best >= self.min_match_chunks:
                holder = self.least_loaded([b for b, n in lengths.items() if n == best])
                if self.inflight[holder] <= min(self.inflight[b] for b in cands) + self.slack:
                    self.stats["prefix_routed"] += 1
                    self.stats["matched_chars"] += best * self.chunk_chars
                    return holder
                self.stats["prefix_spilled"] += 1
        self.stats["least_loaded"] += 1
        return self.least_loaded(cands)

    def mark_down(self, backend):
        self.down_until[backend] = time.monotonic() + self.down_s

    def snapshot(self):
        return {
            "inflight": self.inflight,
            "served": self.served,
            "prefix": dict(self.stats, indexed_chunks=len(self.index.entries)),
        }


def make_app(router, session_factory=None):
    state = {}

    async def proxy(request: web.Request):
        body = await request.read()
        hashes = []
        if request.method == "POST" and request.path in PREFIX_PATHS:
            text = prefix_text(request.path, body)
            if text is not None:
                hashes = chunk_hashes(text, router.chunk_chars)
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        tried = []
        while True:
            b = router.pick(hashes, exclude=tried)
            if b is None:
                return web.json_response({"error": "no backend accepted the request"}, status=503)
            router.inflight[b] += 1
            if hashes:
                # Now, not when the response starts: a request sent while this one prefills (or a non-streaming one,
                # whose headers come at its end) should find the prefix where it is being computed.
                router.index.record(hashes, b)
            try:
                async with state["session"].request(
                    request.method, b + request.path_qs, data=body or None, headers=headers
                ) as up:
                    router.served[b] += 1
                    resp = web.StreamResponse(
                        status=up.status, headers={k: v for k, v in up.headers.items() if k.lower() not in HOP}
                    )
                    await resp.prepare(request)
                    async for chunk in up.content.iter_any():
                        await resp.write(chunk)
                    await resp.write_eof()
                    return resp
            except ClientConnectorError:
                # Nothing reached the client yet: try another engine.
                router.mark_down(b)
                tried.append(b)
                router.stats["retried"] += 1
                if len(tried) == len(router.backends):
                    return web.json_response({"error": "every backend refused the connection"}, status=503)
            finally:
                router.inflight[b] -= 1

    async def stats(_):
        return web.json_response(router.snapshot())

    async def all_metrics(_):
        texts = []
        for b in router.backends:
            async with state["session"].get(b + "/metrics") as up:
                texts.append(await up.text())
        return web.Response(text=merge_metrics(texts), content_type="text/plain")

    async def health(_):
        bad = []
        for b in router.backends:
            try:
                async with state["session"].get(b + "/health", timeout=ClientTimeout(total=10)) as up:
                    if up.status != 200:
                        bad.append(b)
            except Exception:  # noqa: BLE001 - any failure means not healthy
                bad.append(b)
        if bad:
            return web.json_response({"unhealthy": bad}, status=503)
        return web.Response(status=200)

    async def on_start(app):
        state["session"] = (session_factory or ClientSession)(
            timeout=ClientTimeout(total=None), connector=TCPConnector(limit=0)
        )

    async def on_stop(app):
        await state["session"].close()

    app = web.Application(client_max_size=1 << 30)
    app.on_startup.append(on_start)
    app.on_cleanup.append(on_stop)
    app.router.add_get("/lb/stats", stats)
    app.router.add_get("/metrics", all_metrics)
    app.router.add_get("/health", health)
    app.router.add_route("*", "/{tail:.*}", proxy)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--backends", required=True)
    ap.add_argument("--chunk-chars", type=int, default=256, help="prefix key granularity")
    ap.add_argument("--min-match-chars", type=int, default=1024, help="shortest prefix worth routing by")
    ap.add_argument("--slack", type=int, default=2, help="in-flight requests a prefix holder may run above the least")
    ap.add_argument("--index-chunks", type=int, default=262144, help="LRU capacity of the prefix index")
    ap.add_argument("--down-s", type=float, default=5.0, help="how long an engine that refused a connection is skipped")
    a = ap.parse_args()
    backends = [b if b.startswith("http") else f"http://{b}" for b in a.backends.split(",")]
    router = Router(backends, a.chunk_chars, a.min_match_chars, a.slack, a.index_chunks, a.down_s)
    web.run_app(make_app(router), host=a.host, port=a.port, access_log=None)


if __name__ == "__main__":
    main()
