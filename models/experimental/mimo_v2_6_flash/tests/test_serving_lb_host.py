# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""serving/lb.py on the host: prefix keys, the prefix index, routing decisions, and the proxy over fake engines
(affinity, spill, retry past a refusing engine, /health and /metrics)."""

import asyncio
import json
import socket

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

from models.experimental.mimo_v2_6_flash.serving import lb

DOC = "".join(f"Paragraph {i}: the quick brown fox jumps over the lazy dog. " for i in range(200))  # ~12K chars


def chat(doc, question, salt=None):
    body = {"model": "m", "messages": [{"role": "user", "content": doc + "\n\nQuestion: " + question}]}
    if salt is not None:
        body["cache_salt"] = salt
    return json.dumps(body).encode()


def hashes_of(body, path="/v1/chat/completions", chunk=256):
    return lb.chunk_hashes(lb.prefix_text(path, body), chunk)


# ---------------------------------------------------------------- prefix keys and the index


def test_prefix_text_forms():
    assert lb.prefix_text("/v1/completions", json.dumps({"prompt": "abc"}).encode()).endswith("abc")
    one = lb.prefix_text("/v1/completions", json.dumps({"prompt": ["abc"]}).encode())
    assert one == lb.prefix_text("/v1/completions", json.dumps({"prompt": "abc"}).encode())
    ids = lb.prefix_text("/v1/completions", json.dumps({"prompt": [1, 2, 3]}).encode())
    assert ids.endswith("1,2,3,")
    assert lb.prefix_text("/v1/completions", json.dumps({"prompt": ["a", "b"]}).encode()) is None
    assert lb.prefix_text("/v1/chat/completions", b"not json") is None
    assert lb.prefix_text("/v1/chat/completions", json.dumps({"messages": "x"}).encode()) is None
    # The prompt prefix comes first: a longer conversation extends a shorter one's key.
    short = lb.prefix_text(
        "/v1/chat/completions", json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
    )
    longer = lb.prefix_text(
        "/v1/chat/completions",
        json.dumps({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}).encode(),
    )
    assert longer.startswith(short)
    # cache_salt separates namespaces.
    assert hashes_of(chat(DOC, "a", salt="s1"))[0] != hashes_of(chat(DOC, "a", salt="s2"))[0]


def test_chunk_hashes_chain():
    a = hashes_of(chat(DOC, "first question " * 40))
    b = hashes_of(chat(DOC, "a different question " * 40))
    common = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
    assert len(DOC) // 256 - 1 <= common <= len(DOC) // 256 + 1  # the document's chunks match, the questions' do not
    assert lb.chunk_hashes("x" * 255, 256) == []


def test_index_match_and_lru_keeps_heads():
    idx = lb.PrefixIndex(capacity=10)
    h = [bytes([i]) * 16 for i in range(8)]
    idx.record(h, "A")
    idx.record(h[:3], "B")
    assert idx.match(h, ["A", "B", "C"]) == {"A": 8, "B": 3}
    # Filling the LRU evicts the oldest tails first: A keeps a prefix of h, never a hole.
    idx.record([bytes([100 + i]) * 16 for i in range(4)], "C")
    m = idx.match(h, ["A", "B"])["A"]
    assert 0 < m < 8
    assert all(h[i] in idx.entries for i in range(m))


# ---------------------------------------------------------------- routing decisions


def test_router_affinity_short_prompts_and_spill():
    backends = ["A", "B", "C", "D"]
    r = lb.Router(backends, chunk_chars=256, min_match_chars=1024, slack=2)
    first = r.pick(hashes_of(chat(DOC, "q0")))
    r.index.record(hashes_of(chat(DOC, "q0")), first)
    # Same document, another question: the holder, while it is within the slack.
    for q in range(3):
        assert r.pick(hashes_of(chat(DOC, f"q{q + 1}"))) == first
        r.inflight[first] += 1
    # The holder now runs 3 above the others: spill to the least loaded.
    other = r.pick(hashes_of(chat(DOC, "q9")))
    assert other != first
    assert r.stats["prefix_spilled"] == 1
    # Short prompts (below min_match_chars) are load balanced only.
    r.inflight = {b: 0 for b in backends}
    short = [r.pick(hashes_of(chat("short", f"q{i}"))) for i in range(4)]
    assert sorted(short) == backends
    # Among several holders, the least loaded one.
    second = next(b for b in backends if b != first)
    r.index.record(hashes_of(chat(DOC, "q0")), second)
    r.inflight = {b: 0 for b in backends}
    r.inflight[first] = 1
    assert r.pick(hashes_of(chat(DOC, "q5"))) == second


def test_router_skips_a_down_backend():
    r = lb.Router(["A", "B"], down_s=60)
    r.mark_down("A")
    assert {r.pick([]) for _ in range(4)} == {"B"}
    assert r.pick([], exclude=["B"]) == "A"  # every other engine excluded: try the down one again
    assert r.pick([], exclude=["A", "B"]) is None


# ---------------------------------------------------------------- the proxy over fake engines


# What every engine (a DP-1 vLLM server) exposes: the same families and the same engine="0" label.
ENGINE_METRICS = """# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{engine="0",model_name="m"} 5.0
# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.5",model_name="m"} 3.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf",model_name="m"} 4.0
vllm:time_to_first_token_seconds_count{engine="0",model_name="m"} 4.0
vllm:time_to_first_token_seconds_sum{engine="0",model_name="m"} 1.5
process_open_fds 12.0
"""


def test_merge_metrics_labels_each_engine_and_writes_each_family_once():
    merged = lb.merge_metrics([ENGINE_METRICS, ENGINE_METRICS, 'odd{path="a}b\\"c"} 1\n'])
    lines = merged.splitlines()
    assert lines.count("# TYPE vllm:generation_tokens_total counter") == 1
    assert lines.count("# TYPE vllm:time_to_first_token_seconds histogram") == 1
    samples = [line for line in lines if not line.startswith("#")]
    assert len(samples) == len(set(samples)) == 2 * 6 + 1  # every series once
    assert 'vllm:generation_tokens_total{model_name="m",engine="0"} 5.0' in samples
    assert 'vllm:generation_tokens_total{model_name="m",engine="1"} 5.0' in samples
    assert 'process_open_fds{engine="1"} 12.0' in samples
    assert 'odd{path="a}b\\"c",engine="2"} 1' in samples  # a value with } and an escaped quote
    # A family's samples stay together, right after its TYPE line.
    start = lines.index("# TYPE vllm:time_to_first_token_seconds histogram")
    assert all(line.startswith("vllm:time_to_first_token_seconds_") for line in lines[start + 1 : start + 9])


def fake_engine(name, healthy=True):
    async def completions(request):
        body = await request.json()
        delay = float(request.headers.get("x-delay", "0"))
        await asyncio.sleep(delay)
        return web.json_response({"backend": name, "n": len(json.dumps(body))})

    async def health(_):
        return web.Response(status=200 if healthy else 503)

    async def metrics(_):
        return web.Response(text=ENGINE_METRICS)

    app = web.Application(client_max_size=1 << 26)
    app.router.add_post("/v1/chat/completions", completions)
    app.router.add_post("/v1/completions", completions)
    app.router.add_get("/health", health)
    app.router.add_get("/metrics", metrics)
    return app


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def with_proxy(engines, fn, dead=0, **router_kw):
    servers = [TestServer(app, host="127.0.0.1") for app in engines]
    for s in servers:
        await s.start_server()
    backends = [f"http://127.0.0.1:{s.port}" for s in servers]
    backends += [f"http://127.0.0.1:{free_port()}" for _ in range(dead)]  # nothing listens there
    router = lb.Router(backends, **router_kw)
    proxy = TestServer(lb.make_app(router), host="127.0.0.1")
    await proxy.start_server()
    try:
        async with ClientSession() as session:
            return await fn(session, f"http://127.0.0.1:{proxy.port}", router, backends)
    finally:
        await proxy.close()
        for s in servers:
            await s.close()


async def post(session, url, body, delay=0.0):
    async with session.post(
        url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json", "x-delay": str(delay)}
    ) as r:
        assert r.status == 200, await r.text()
        return (await r.json())["backend"]


def test_proxy_documents_stay_on_their_engine():
    docs = [f"Document {d}. " + DOC for d in range(4)]

    async def run(session, url, router, backends):
        where = {}
        for q in range(3):
            for d, doc in enumerate(docs):
                where.setdefault(d, set()).add(await post(session, url, chat(doc, f"question {q}")))
        return where, router.snapshot()

    where, snap = asyncio.run(with_proxy([fake_engine(n) for n in "ABCD"], run))
    assert all(len(engines) == 1 for engines in where.values()), where
    assert len(set().union(*where.values())) == 4  # the four documents went to four engines
    assert snap["prefix"]["prefix_routed"] == 8 and snap["prefix"]["least_loaded"] == 4


def test_proxy_spills_a_hot_prefix_and_balances():
    async def run(session, url, router, backends):
        got = await asyncio.gather(*(post(session, url, chat(DOC, f"q{i}"), delay=0.3) for i in range(16)))
        return got, router.snapshot()

    got, snap = asyncio.run(with_proxy([fake_engine(n) for n in "ABCD"], run, slack=2))
    counts = sorted(got.count(n) for n in "ABCD")
    assert counts[0] >= 2 and counts[-1] <= 6, counts  # spread, not all on the first holder
    assert snap["prefix"]["prefix_spilled"] > 0


def test_proxy_retries_past_a_refusing_engine():
    async def run(session, url, router, backends):
        got = [await post(session, url, chat(f"short {i}", "q")) for i in range(6)]
        return got, router.snapshot()

    got, snap = asyncio.run(with_proxy([fake_engine("A"), fake_engine("B")], run, dead=1))
    assert set(got) == {"A", "B"}
    assert snap["prefix"]["retried"] >= 1


def test_proxy_health_and_metrics():
    async def run(session, url, router, backends):
        async with session.get(url + "/health") as r:
            status = r.status
        async with session.get(url + "/metrics") as r:
            text = await r.text()
        return status, text

    status, text = asyncio.run(with_proxy([fake_engine("A"), fake_engine("B")], run))
    assert status == 200
    assert text == lb.merge_metrics([ENGINE_METRICS, ENGINE_METRICS])
    status, _ = asyncio.run(with_proxy([fake_engine("A"), fake_engine("B", healthy=False)], run))
    assert status == 503


def test_proxy_metrics_without_a_dead_or_hung_engine(monkeypatch):
    monkeypatch.setattr(lb, "PROBE_TIMEOUT_S", 0.5)

    async def hung_metrics(_):
        await asyncio.sleep(30)

    hung = web.Application()
    hung.router.add_get("/metrics", hung_metrics)

    async def run(session, url, router, backends):
        async with session.get(url + "/metrics") as r:
            return r.status, await r.text()

    status, text = asyncio.run(with_proxy([fake_engine("A"), hung], run, dead=1))
    assert status == 200
    assert text.startswith("# engine 1 (")  # the hung engine
    assert "# engine 2 (" in text  # the one nothing listens for
    assert 'vllm:generation_tokens_total{model_name="m",engine="0"} 5.0' in text


@pytest.mark.parametrize("dead", [1])
def test_proxy_health_with_a_dead_engine(dead):
    async def run(session, url, router, backends):
        async with session.get(url + "/health") as r:
            return r.status

    assert asyncio.run(with_proxy([fake_engine("A")], run, dead=dead)) == 503
