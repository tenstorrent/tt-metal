# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""serving/lb.py on the host: prefix keys, the prefix index, routing decisions, the multimodal request shim, and the
proxy over fake engines (affinity, spill, retry past a refusing engine, /health and /metrics, rewrites, 422 -> 400,
413)."""

import asyncio
import base64
import hashlib
import io
import json
import socket
import wave

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


# ---------------------------------------------------------------- the multimodal request shim


def legacy_prefix_text(path, body):
    """prefix_text before the shim (media keyed as text), frozen: text-only requests must key exactly like this."""
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
            prompt = prompt[0]
        if isinstance(prompt, str):
            return head + prompt
        if isinstance(prompt, list) and all(isinstance(t, int) for t in prompt):
            return head + ",".join(map(str, prompt)) + ","
        return None
    if path == "/v1/responses":
        inp = req.get("input")
        text = inp if isinstance(inp, str) else json.dumps(inp, sort_keys=True, ensure_ascii=False)
        return head + json.dumps(req.get("instructions")) + text
    return None


TOOL = {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"x": {}}}}}
TEXT_ONLY = [
    ("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": DOC + "Q?"}]}),
    (
        "/v1/chat/completions",
        {
            "model": "m",
            "messages": [
                {"role": "system", "content": "Be brief. ünïcödé"},
                {"role": "user", "content": [{"type": "text", "text": DOC}, {"type": "text", "text": "Q?"}]},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "1", "content": "42"},
            ],
            "tools": [TOOL],
            "chat_template_kwargs": {"enable_thinking": False},
            "cache_salt": "s",
            "stream": True,
            "include_reasoning": False,
            "reasoning_effort": "low",
        },
    ),
    ("/v1/completions", {"model": "m", "prompt": DOC}),
    ("/v1/completions", {"model": "m", "prompt": list(range(600))}),
    ("/v1/responses", {"model": "m", "input": DOC, "instructions": "x"}),
    ("/v1/responses", {"model": "m", "input": [{"role": "user", "content": [{"type": "input_text", "text": DOC}]}]}),
]


@pytest.mark.parametrize("path,req", TEXT_ONLY)
def test_text_only_requests_are_forwarded_and_keyed_as_before(path, req):
    body = json.dumps(req).encode()
    assert lb.prefix_text(path, body) == legacy_prefix_text(path, body)
    out, hashes = lb.prepare(path, body, lb.Shim(), 256)
    assert out is body  # the original bytes, not a re-serialization
    assert hashes == lb.chunk_hashes(legacy_prefix_text(path, body), 256)


def image_part(payload, detail=None):
    part = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + payload}}
    if detail:
        part["image_url"]["detail"] = detail
    return part


def test_media_parts_are_keyed_by_digest():
    big = base64.b64encode(bytes(range(256)) * 8000).decode()  # ~2.7 MB of base64
    other = base64.b64encode(bytes(range(255, -1, -1)) * 8000).decode()

    def req(payload, question):
        return {
            "model": "m",
            "messages": [{"role": "user", "content": [image_part(payload, "low"), {"type": "text", "text": question}]}],
        }

    original = req(big, "What is this?")
    snapshot = json.dumps(original)
    text = lb.prefix_text_of("/v1/chat/completions", original)
    assert json.dumps(original) == snapshot  # the request itself is untouched
    assert big not in text and lb.media_key("data:image/png;base64," + big) in text
    assert '"detail": "low"' in text
    assert len(lb.media_key("x")) == lb.MEDIA_KEY_CHARS
    hashes = lb.chunk_hashes(text, 256)
    assert len(hashes) < 10  # instead of ~10K chunks of base64
    # Same image, another question: the image's chunks match; another image: nothing after the head does.
    same = lb.chunk_hashes(lb.prefix_text_of("/v1/chat/completions", req(big, "And now?")), 256)
    diff = lb.chunk_hashes(lb.prefix_text_of("/v1/chat/completions", req(other, "What is this?")), 256)
    common = lambda a, b: next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))  # noqa: E731
    assert common(hashes, same) >= lb.MEDIA_KEY_CHARS // 256
    assert common(hashes, diff) < common(hashes, same)
    # Every media form OpenRouter / vLLM use, and the Responses API's input_image.
    view = lb.media_key_view(
        [
            {"type": "input_audio", "input_audio": {"data": big, "format": "wav"}},
            {"type": "video_url", "video_url": {"url": "https://x/v.mp4"}},
            {"type": "input_video", "video_url": {"url": "data:video/mp4;base64," + big}},
            {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + big}},
            {"type": "input_image", "image_url": "data:image/png;base64," + big},
        ]
    )
    assert big not in json.dumps(view) and "https://x/v.mp4" not in json.dumps(view)
    assert view[0]["input_audio"]["format"] == "wav"


def wav_params(data):
    with wave.open(io.BytesIO(data)) as w:
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()


def test_pcm_to_wav(expect_error):
    assert wav_params(lb.pcm_to_wav(b"\x01\x02" * 480, 2, 24000)) == (1, 2, 24000, 480)
    assert wav_params(lb.pcm_to_wav(b"\x01\x02\x03" * 480, 3, 24000)) == (1, 3, 24000, 480)
    with expect_error(lb.ShimError, "whole number"):
        lb.pcm_to_wav(b"\x01\x02\x03", 2, 24000)
    with expect_error(lb.ShimError, "whole number"):
        lb.pcm_to_wav(b"", 2, 24000)


def audio_req(data, fmt):
    return {
        "model": "m",
        "messages": [
            {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": data, "format": fmt}}]}
        ],
    }


@pytest.mark.parametrize("fmt,width", [("pcm16", 2), ("pcm24", 3), ("PCM16", 2)])
def test_rewrite_pcm_input_audio_to_wav(fmt, width, expect_error):
    raw = bytes(range(240)) * width * 10
    req = audio_req(base64.b64encode(raw).decode(), fmt)
    out = lb.rewrite_chat(req, lb.Shim(pcm_sample_rate=16000))
    audio = out["messages"][0]["content"][0]["input_audio"]
    assert audio["format"] == "wav"
    wav = base64.b64decode(audio["data"])
    assert wav[:4] == b"RIFF" and wav_params(wav) == (1, width, 16000, len(raw) // width) and wav.endswith(raw)
    assert req["messages"][0]["content"][0]["input_audio"]["format"] == fmt  # the input is not modified
    with expect_error(lb.ShimError, "base64"):
        lb.rewrite_chat(audio_req("not base64!!", "pcm16"), lb.Shim())


def test_rewrite_legacy_input_video():
    url = {"url": "data:video/mp4;base64,AAAA"}
    out = lb.rewrite_chat(
        {"messages": [{"role": "user", "content": [{"type": "input_video", "video_url": url, "x": 1}]}]}, lb.Shim()
    )
    assert out["messages"][0]["content"][0] == {"type": "video_url", "video_url": url, "x": 1}
    out = lb.rewrite_chat(
        {"messages": [{"role": "user", "content": [{"type": "input_video", "video_url": "https://v"}]}]}, lb.Shim()
    )
    assert out["messages"][0]["content"][0] == {"type": "video_url", "video_url": {"url": "https://v"}}


def test_rewrite_leaves_supported_forms_alone():
    wav = base64.b64encode(lb.pcm_to_wav(b"\x00\x01" * 100, 2, 24000)).decode()
    req = {
        "model": "m",
        "messages": [
            {"role": "user", "content": [image_part("AAAA"), {"type": "video_url", "video_url": {"url": "https://v"}}]},
            {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": wav, "format": "wav"}}]},
            {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": wav, "format": "mp3"}}]},
        ],
        "reasoning_effort": "none",
    }
    assert lb.rewrite_chat(req, lb.Shim()) is req


@pytest.mark.parametrize(
    "fields,want_kwargs,want_include",
    [
        ({"reasoning": {"enabled": False}}, {"enable_thinking": False}, None),
        ({"reasoning": {"enabled": True}}, {"enable_thinking": True}, None),
        ({"reasoning": {"effort": "none"}}, {"enable_thinking": False}, None),
        ({"reasoning": {"effort": "high"}}, {"enable_thinking": True}, None),
        ({"reasoning": {"max_tokens": 0}}, {"enable_thinking": False}, None),
        ({"reasoning": {"exclude": True}}, None, False),
        ({"reasoning": {"enabled": False}, "chat_template_kwargs": {"x": 1}}, {"x": 1, "enable_thinking": False}, None),
        # An explicit switch wins; the top-level reasoning_effort is vLLM's own.
        (
            {"reasoning": {"enabled": False}, "chat_template_kwargs": {"enable_thinking": True}},
            {"enable_thinking": True},
            None,
        ),
        ({"reasoning": {"enabled": False}, "reasoning_effort": "high"}, None, None),
        ({"reasoning": {"exclude": True}, "include_reasoning": True}, None, True),
        ({"include_reasoning": False}, None, False),  # passes through: hides the reasoning, does not switch it off
    ],
)
def test_map_reasoning(fields, want_kwargs, want_include):
    req = dict({"model": "m", "messages": []}, **fields)
    out = lb.map_reasoning(req)
    assert out.get("chat_template_kwargs") == want_kwargs
    assert out.get("include_reasoning") == want_include
    assert req == dict({"model": "m", "messages": []}, **fields)  # not modified


@pytest.mark.parametrize(
    "fields,want",
    [
        ({"temperature": 0}, (1.0, 0.95)),
        ({"temperature": 0.0, "top_p": 0.5}, (1.0, 0.95)),
        ({"temperature": 0, "chat_template_kwargs": {"enable_thinking": True}}, (1.0, 0.95)),
        ({"temperature": 0, "reasoning": {"enabled": True}}, (1.0, 0.95)),
        # Thinking off, a sampled request, or no temperature (vLLM's generation-config default): unchanged.
        ({"temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, None),
        ({"temperature": 0, "reasoning": {"enabled": False}}, None),
        ({"temperature": 0, "reasoning": {"effort": "none"}}, None),
        ({"temperature": 0.7}, None),
        ({}, None),
        ({"temperature": False}, None),
    ],
)
def test_map_greedy_thinking(fields, want):
    req = dict({"model": "m", "messages": [{"role": "user", "content": "hi"}]}, **fields)
    snapshot = json.loads(json.dumps(req))
    out = lb.rewrite_chat(req, lb.Shim(thinking_greedy=(1.0, 0.95)))
    if want is None:
        assert out.get("temperature") == req.get("temperature") and out.get("top_p") == req.get("top_p")
    else:
        assert (out["temperature"], out["top_p"]) == want
    assert req == snapshot  # not modified
    # Off by default: the request is forwarded as is (the same object when nothing else applies).
    assert lb.rewrite_chat(req, lb.Shim()).get("temperature") == req.get("temperature")
    assert lb.map_greedy_thinking(req, None) is req


@pytest.mark.parametrize("bad", ["0,0.95", "1.0,0", "1.0,1.5", "1.0", "a,b", "1,0.9,3"])
def test_parse_thinking_greedy(bad, expect_error):
    assert lb.Shim.parse_sampling("1.0,0.95") == (1.0, 0.95)
    assert lb.Shim.parse_sampling("") is None and lb.Shim.parse_sampling(None) is None
    with expect_error(ValueError, "want T,P"):
        lb.Shim.parse_sampling(bad)


def test_map_error():
    vllm_422 = json.dumps(
        {
            "error": {
                "message": "Failed to fetch media from URL: HTTP 404 error",
                "type": "UnprocessableEntityError",
                "param": "image_url",
                "code": 422,
            }
        }
    ).encode()
    status, body = lb.map_error(422, vllm_422)
    err = json.loads(body)["error"]
    assert status == 400 and err["code"] == 400 and err["type"] == "BadRequestError"
    assert err["message"] == "Failed to fetch media from URL: HTTP 404 error" and err["param"] == "image_url"
    assert lb.map_error(422, b"plain") == (400, b"plain")
    for status in (200, 400, 404, 500):
        assert lb.map_error(status, vllm_422) == (status, vllm_422)


def test_fix_flac_length():
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("av")
    import numpy as np

    buf = io.BytesIO()
    sf.write(buf, (np.sin(np.arange(24000 * 2) / 7) * 0.3).astype(np.float32), 24000, format="FLAC")
    good = buf.getvalue()
    assert lb.flac_total_samples(good) == 48000 and lb.fix_flac_length(good) is None
    piped = bytearray(good)  # what an encoder writing to a pipe leaves: length unknown
    piped[18:26] = (int.from_bytes(good[18:26], "big") & ~((1 << 36) - 1)).to_bytes(8, "big")
    piped = bytes(piped)
    assert lb.flac_total_samples(piped) == 0
    fixed = lb.fix_flac_length(piped)
    assert fixed == good
    with sf.SoundFile(io.BytesIO(fixed)) as f:
        assert f.frames == 48000
    assert lb.flac_total_samples(b"RIFF....WAVE" * 4) is None and lb.fix_flac_length(b"ID3" * 20) is None
    out = lb.rewrite_chat(audio_req(base64.b64encode(piped).decode(), "flac"), lb.Shim())
    assert base64.b64decode(out["messages"][0]["content"][0]["input_audio"]["data"]) == good


# ---------------------------------------------------------------- the shim in the proxy


def echo_engine():
    """An engine that answers what it received (sha256 of the raw body, and the parsed body), and 422 for a media URL
    it cannot fetch, as vLLM 0.30 does."""

    async def completions(request):
        raw = await request.read()
        body = json.loads(raw)
        if "no-such-image" in raw.decode(errors="replace"):
            err = {
                "message": "Failed to fetch media from URL: HTTP 404 error",
                "type": "UnprocessableEntityError",
                "param": "image_url",
                "code": 422,
            }
            return web.json_response({"error": err}, status=422)
        return web.json_response({"backend": "E", "sha256": hashlib.sha256(raw).hexdigest(), "body": body})

    app = web.Application(client_max_size=1 << 30)
    app.router.add_post("/v1/chat/completions", completions)
    return app


async def raw_post(session, url, body):
    async with session.post(url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}) as r:
        return r.status, await r.json()


def test_proxy_shim_end_to_end():
    text_body = json.dumps(TEXT_ONLY[1][1]).encode()
    pcm = base64.b64encode(b"\x00\x10" * 24000).decode()
    big_image = base64.b64encode(b"\x89PNG" + bytes(600_000)).decode()  # above OFFLOAD_BYTES: the worker-thread path
    bodies = {
        "text": text_body,
        "pcm": json.dumps(audio_req(pcm, "pcm16")).encode(),
        "image": json.dumps(
            {"model": "m", "messages": [{"role": "user", "content": [image_part(big_image)]}]}
        ).encode(),
        "404": json.dumps(
            {
                "model": "m",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "https://example.com/no-such-image.png"}}
                        ],
                    }
                ],
            }
        ).encode(),
        "bad": json.dumps(audio_req("%%%", "pcm24")).encode(),
        "huge": json.dumps({"model": "m", "messages": [{"role": "user", "content": "x" * 3_000_000}]}).encode(),
    }

    async def run(session, url, router, backends):
        return {name: await raw_post(session, url, body) for name, body in bodies.items()}, router.snapshot()

    async def with_shim():
        servers = [TestServer(echo_engine(), host="127.0.0.1")]
        await servers[0].start_server()
        router = lb.Router([f"http://127.0.0.1:{servers[0].port}"])
        proxy = TestServer(lb.make_app(router, shim=lb.Shim(max_request_bytes=2_000_000)), host="127.0.0.1")
        await proxy.start_server()
        try:
            async with ClientSession() as session:
                return await run(session, f"http://127.0.0.1:{proxy.port}", router, None)
        finally:
            await proxy.close()
            await servers[0].close()

    got, snap = asyncio.run(with_shim())
    status, out = got["text"]
    assert status == 200 and out["sha256"] == hashlib.sha256(text_body).hexdigest()  # byte for byte
    status, out = got["pcm"]
    audio = out["body"]["messages"][0]["content"][0]["input_audio"]
    assert status == 200 and audio["format"] == "wav" and base64.b64decode(audio["data"])[:4] == b"RIFF"
    status, out = got["image"]
    assert status == 200 and out["sha256"] == hashlib.sha256(bodies["image"]).hexdigest()
    status, out = got["404"]
    assert status == 400 and out["error"]["message"].startswith("Failed to fetch media from URL")
    status, out = got["bad"]
    assert status == 400 and "base64" in out["error"]["message"]
    status, out = got["huge"]
    assert status == 413 and out["error"]["code"] == 413
    assert snap["prefix"]["errors_mapped"] == 1 and snap["prefix"]["shim_rejected"] == 1
    assert snap["prefix"]["too_large"] == 1


# ---------------------------------------------------------------- downloading http(s) media in the proxy

PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 10  # 2.5 KB, below the 4096-byte limit used below


def media_host(hits=None):
    """A local media server: an image, a file without a media Content-Type, a web page, 404, a slow answer, two files
    above the size limit (one declaring its length, one streamed), and redirects (to the image, to the page, to
    YouTube, to another loopback address, to cloud metadata, to a file: URL, and to itself). ``hits`` collects (path, Host header) of every request.
    """
    hits = [] if hits is None else hits

    def logged(handler):
        async def wrapped(request):
            hits.append((request.path, request.headers.get("Host")))
            return await handler(request)

        return wrapped

    async def png(_):
        return web.Response(body=PNG, content_type="image/png")

    async def octet(_):
        return web.Response(body=PNG, content_type="application/octet-stream")

    async def slow(_):
        await asyncio.sleep(3)
        return web.Response(body=PNG, content_type="image/png")

    async def big(_):
        return web.Response(body=bytes(5000), content_type="video/mp4")

    async def streamed(request):
        resp = web.StreamResponse(headers={"Content-Type": "video/mp4"})
        await resp.prepare(request)
        for _ in range(10):
            await resp.write(bytes(1000))
        await resp.write_eof()
        return resp

    async def page(_):
        return web.Response(text="<html><body>watch</body></html>", content_type="text/html", charset="utf-8")

    def redirect(location):
        async def handler(request):
            port = request.url.port
            raise web.HTTPFound(location.format(port=port))

        return handler

    app = web.Application()
    for path, handler in (
        ("/cat.png", png),
        ("/photo.jpg", octet),
        ("/slow.png", slow),
        ("/big.mp4", big),
        ("/streamed.mp4", streamed),
        ("/watch", page),
        ("/hop", redirect("/cat.png")),
        ("/to-page", redirect("/watch")),
        ("/to-youtube", redirect("https://www.youtube.com/watch?v=abc")),
        ("/to-other-loopback", redirect("http://127.0.0.2:{port}/cat.png")),
        ("/to-metadata", redirect("http://169.254.169.254/latest/meta-data/")),
        ("/to-file", redirect("file:///etc/passwd")),
        ("/loop", redirect("/loop")),
    ):
        app.router.add_get(path, logged(handler))
    return app


@pytest.mark.parametrize(
    "host,want",
    [
        ("www.youtube.com", "youtube.com"),
        ("youtube.com", "youtube.com"),
        ("M.YouTube.com.", "youtube.com"),
        ("music.youtube.com", "youtube.com"),
        ("youtu.be", "youtu.be"),
        ("www.youtube-nocookie.com", "youtube-nocookie.com"),
        ("notyoutube.com", None),
        ("youtube.com.example.org", None),
        ("example.org", None),
        (None, None),
    ],
)
def test_page_host(host, want):
    assert lb.page_host(host) == want


def test_fetch_refuses_page_links_and_page_answers():
    """OpenRouter's input-youtube-url request: a YouTube link is refused by name before anything is resolved or
    downloaded, and a URL that answers a web page is refused instead of being handed to the engine's video decoder
    (which says "Could not open video stream")."""
    hits = []

    async def run():
        host = TestServer(media_host(hits), host="127.0.0.1")
        await host.start_server()
        base = f"http://127.0.0.1:{host.port}"
        out = {}
        try:
            for name, url in (
                ("youtube", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
                ("short", "https://youtu.be/dQw4w9WgXcQ"),
                ("page", base + "/watch"),
                ("to-page", base + "/to-page"),
                ("to-youtube", base + "/to-youtube"),
            ):
                try:
                    # The default policy for the YouTube links: refused before their host is resolved.
                    policy = only_127_0_0_1 if url.startswith(base) else None
                    await lb.fetch_media([url], 2.0, 4096, address_policy=policy)
                except lb.ShimError as exc:
                    out[name] = str(exc)
        finally:
            await host.close()
        return out

    out = asyncio.run(run())
    for name in ("youtube", "to-youtube"):
        assert "youtube.com links are not supported" in out[name], out[name]
    assert "youtu.be links are not supported" in out["short"]
    assert "dQw4w9WgXcQ" in out["youtube"]  # the message names the URL
    for name in ("page", "to-page"):
        assert "answered a web page (text/html), not a media file" in out[name], out[name]
    assert all(msg.startswith("Failed to fetch media from URL: ") for msg in out.values())
    assert [path for path, _ in hits] == ["/watch", "/to-page", "/watch", "/to-youtube"]


def only_127_0_0_1(ip):
    """The local test server's address is allowed; every other address gets the default verdict."""
    return None if ip == "127.0.0.1" else lb.blocked_reason(ip)


def chat_with(*parts):
    return {"model": "m", "messages": [{"role": "user", "content": [*parts, {"type": "text", "text": "Describe."}]}]}


def test_remote_media_urls_and_inline_media():
    req = chat_with(
        {"type": "image_url", "image_url": {"url": "https://h/a.png", "detail": "low"}},
        {"type": "video_url", "video_url": {"url": "HTTP://h/v.mp4"}},
        {"type": "image_url", "image_url": {"url": "https://h/a.png"}},  # a repeat: fetched once
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "audio_url", "audio_url": "https://h/s.wav"},
        {"type": "image_url", "image_url": {"url": "file:///etc/passwd"}},  # never fetched by the proxy
    )
    assert lb.remote_media_urls(req) == ["https://h/a.png", "HTTP://h/v.mp4", "https://h/s.wav"]
    snapshot = json.dumps(req)
    out = lb.inline_media(
        req, {"https://h/a.png": "data:image/png;base64,QQ==", "https://h/s.wav": "data:audio/wav;base64,UQ=="}
    )
    parts = out["messages"][0]["content"]
    assert parts[0]["image_url"] == {"url": "data:image/png;base64,QQ==", "detail": "low"}
    assert parts[1]["video_url"] == {"url": "HTTP://h/v.mp4"} and parts[2]["image_url"]["url"].startswith("data:")
    assert (
        parts[4]["audio_url"] == "data:audio/wav;base64,UQ==" and parts[5]["image_url"]["url"] == "file:///etc/passwd"
    )
    assert json.dumps(req) == snapshot
    assert lb.remote_media_urls({"model": "m", "messages": [{"role": "user", "content": "https://h/a.png"}]}) == []


def test_media_type():
    assert lb.media_type("image/png; charset=binary", "https://h/x") == "image/png"
    assert lb.media_type("application/octet-stream", "https://h/x.jpg?sig=1") == "image/jpeg"
    assert lb.media_type(None, "https://h/x") == "application/octet-stream"


def test_fetch_media_inlines_and_fails_with_400_messages():
    async def run():
        host = TestServer(media_host(), host="127.0.0.1")
        await host.start_server()
        base = f"http://127.0.0.1:{host.port}"
        out = {}
        try:
            got = await lb.fetch_media(
                [base + "/cat.png", base + "/photo.jpg", base + "/hop"], 2.0, 4096 * 4, address_policy=only_127_0_0_1
            )
            out["ok"] = got
            for name in ("missing.png", "slow.png", "big.mp4", "streamed.mp4", "loop", "to-file"):
                try:
                    await lb.fetch_media(
                        [base + "/cat.png", base + "/" + name], 0.5, 4096, address_policy=only_127_0_0_1
                    )
                except lb.ShimError as exc:
                    out[name] = str(exc)
            try:
                await lb.fetch_media(
                    [f"http://127.0.0.1:{free_port()}/x.png"], 2.0, 4096, address_policy=only_127_0_0_1
                )
            except lb.ShimError as exc:
                out["refused"] = str(exc)
        finally:
            await host.close()
        return base, out

    base, out = asyncio.run(run())
    assert out["ok"][base + "/cat.png"] == "data:image/png;base64," + base64.b64encode(PNG).decode()
    assert out["ok"][base + "/photo.jpg"].startswith("data:image/jpeg;base64,")
    assert out["ok"][base + "/hop"] == out["ok"][base + "/cat.png"]  # one redirect, followed
    assert "HTTP 404 error" in out["missing.png"] and "missing.png" in out["missing.png"]
    assert "no answer within 0.5 s" in out["slow.png"]
    assert "larger than 4096 bytes" in out["big.mp4"] and "larger than 4096 bytes" in out["streamed.mp4"]
    assert "more than 3 redirects" in out["loop"]
    assert "only http(s) URLs" in out["to-file"]
    assert out["refused"].startswith("Failed to fetch media from URL: ClientConnectorError")


BLOCKED = [
    ("127.0.0.1", "loopback"),
    ("127.8.9.10", "loopback"),
    ("::1", "loopback"),
    ("10.1.2.3", "private"),
    ("172.16.0.1", "private"),
    ("192.168.1.1", "private"),
    ("fc00::1", "private"),
    ("fd12:3456::1", "private"),
    ("169.254.169.254", "link-local"),  # cloud metadata
    ("fe80::1", "link-local"),
    ("fe80::1%eth0", "link-local"),
    ("224.0.0.1", "multicast"),
    ("ff02::1", "multicast"),
    ("0.0.0.0", "unspecified"),
    ("::", "unspecified"),
    ("::ffff:127.0.0.1", "loopback"),  # IPv4-mapped
    ("::ffff:169.254.169.254", "link-local"),
    ("2002:a00:1::1", "private"),  # 6to4 around 10.0.0.1
    ("100.64.0.1", "not globally routable"),  # carrier-grade NAT
    ("192.0.2.1", ""),  # documentation (private or not global, depending on the Python version)
    ("240.0.0.1", ""),  # reserved
]


@pytest.mark.parametrize("ip,label", BLOCKED)
def test_blocked_reason_refuses_non_public_addresses(ip, label):
    reason = lb.blocked_reason(ip)
    assert reason, ip
    assert label in reason


@pytest.mark.parametrize("ip", ["8.8.8.8", "93.184.216.34", "2606:4700:4700::1111", "::ffff:8.8.8.8"])
def test_blocked_reason_allows_public_unicast(ip):
    assert lb.blocked_reason(ip) is None


def test_fetch_refuses_non_public_targets_before_connecting(monkeypatch):
    """Every refusal happens before a connection: the local server never sees a request."""
    hits = []

    async def run():
        host = TestServer(media_host(hits), host="127.0.0.1")
        await host.start_server()
        port = host.port
        urls = [
            f"http://127.0.0.1:{port}/cat.png",  # loopback literal
            f"http://localhost:{port}/cat.png",  # a name that resolves to loopback
            f"http://[::1]:{port}/cat.png",
            f"http://[::ffff:127.0.0.1]:{port}/cat.png",
            f"http://0.0.0.0:{port}/cat.png",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.1/x.png",
            "http://192.168.0.10/x.png",
            "http://[fd00::1]/x.png",
            "http://[fe80::1]/x.png",
            "http://224.0.0.1/x.png",
            "ftp://example.com/x.png",
        ]
        out = {}
        try:
            for url in urls:
                try:
                    await lb.fetch_media([url], 2.0, 1 << 20)  # the default policy
                    out[url] = "fetched"
                except lb.ShimError as exc:
                    out[url] = str(exc)
        finally:
            await host.close()
        return out

    out = asyncio.run(run())
    for url, message in out.items():
        assert message.startswith("Failed to fetch media from URL: "), (url, message)
        assert "refused" in message or "only http(s)" in message, (url, message)
    assert hits == []


def test_fetch_checks_every_redirect_hop():
    hits = []

    async def run():
        host = TestServer(media_host(hits), host="127.0.0.1")
        await host.start_server()
        base = f"http://127.0.0.1:{host.port}"
        out = {}
        try:
            for name in ("to-other-loopback", "to-metadata"):
                try:
                    await lb.fetch_media([f"{base}/{name}"], 2.0, 1 << 20, address_policy=only_127_0_0_1)
                except lb.ShimError as exc:
                    out[name] = str(exc)
        finally:
            await host.close()
        return out

    out = asyncio.run(run())
    assert "127.0.0.2 (loopback); refused" in out["to-other-loopback"]
    assert "169.254.169.254 (link-local); refused" in out["to-metadata"]
    # Only the first hops reached the server; the refused targets never got a connection.
    assert [path for path, _ in hits] == ["/to-other-loopback", "/to-metadata"]


def test_fetch_connects_only_to_the_checked_address(monkeypatch):
    """DNS rebinding: the name is resolved once, checked, and the connection goes to that address. ``media.test``
    never resolves through DNS (.test is reserved), so the download works only through the pinned address."""
    hits, lookups = [], []

    async def fake_resolve(host, port):
        lookups.append(host)
        return {"media.test": ["127.0.0.1"], "mixed.test": ["93.184.216.34", "10.0.0.1"]}[host]

    monkeypatch.setattr(lb, "resolve_host", fake_resolve)

    async def run():
        host = TestServer(media_host(hits), host="127.0.0.1")
        await host.start_server()
        out, port = {}, host.port
        try:
            url = f"http://media.test:{port}/cat.png"
            out["pinned"] = await lb.fetch_media([url], 2.0, 1 << 20, address_policy=only_127_0_0_1)
            try:
                await lb.fetch_media([f"http://mixed.test:{port}/cat.png"], 2.0, 1 << 20)
            except lb.ShimError as exc:
                out["mixed"] = str(exc)
        finally:
            await host.close()
        return port, out

    port, out = asyncio.run(run())
    assert list(out["pinned"].values()) == ["data:image/png;base64," + base64.b64encode(PNG).decode()]
    assert hits == [("/cat.png", f"media.test:{port}")]  # the name in Host, the checked address on the wire
    assert lookups == ["media.test", "mixed.test"]  # one lookup per hop, none by aiohttp
    assert "10.0.0.1 (private); refused" in out["mixed"]  # one non-public answer refuses the name


def test_shim_from_env(monkeypatch):
    for name in (
        "TT_PROXY_REQUEST_SHIM",
        "TT_PROXY_FETCH_MEDIA",
        "TT_PROXY_MAX_REQUEST_MB",
        "TT_PROXY_FETCH_MAX_MB",
        "TT_PROXY_THINKING_GREEDY",
    ):
        monkeypatch.delenv(name, raising=False)
    assert lb.Shim.from_env() == lb.Shim()
    monkeypatch.setenv("TT_PROXY_FETCH_MEDIA", "1")
    monkeypatch.setenv("TT_PROXY_MAX_REQUEST_MB", "0.5")
    monkeypatch.setenv("TT_PROXY_PCM_SAMPLE_RATE", "16000")
    shim = lb.Shim.from_env(fetch_timeout_s=5)
    assert shim.fetch_media and shim.max_request_bytes == 1 << 19 and shim.pcm_sample_rate == 16000
    assert shim.fetch_timeout_s == 5 and shim.enabled
    monkeypatch.setenv("TT_PROXY_REQUEST_SHIM", "0")
    assert not lb.Shim.from_env().enabled
    assert lb.Shim.from_env().thinking_greedy is None
    monkeypatch.setenv("TT_PROXY_THINKING_GREEDY", "1.0,0.95")
    assert lb.Shim.from_env().thinking_greedy == (1.0, 0.95)


def test_proxy_fetches_media_when_enabled():
    async def run(fetch, allow_private=True):
        host = TestServer(media_host(), host="127.0.0.1")
        engine = TestServer(echo_engine(), host="127.0.0.1")
        await host.start_server()
        await engine.start_server()
        router = lb.Router([f"http://127.0.0.1:{engine.port}"])
        shim = lb.Shim(fetch_media=fetch, fetch_timeout_s=0.5, fetch_allow_private=allow_private)
        proxy = TestServer(lb.make_app(router, shim=shim), host="127.0.0.1")
        await proxy.start_server()
        base, url = f"http://127.0.0.1:{host.port}", f"http://127.0.0.1:{proxy.port}"
        try:
            async with ClientSession() as session:
                out = {}
                for name in ("cat.png", "missing.png", "slow.png"):
                    body = chat_with({"type": "image_url", "image_url": {"url": f"{base}/{name}"}})
                    out[name] = await raw_post(session, url, json.dumps(body).encode())
                return base, out, router.snapshot()
        finally:
            await proxy.close()
            await engine.close()
            await host.close()

    base, out, snap = asyncio.run(run(True))
    status, got = out["cat.png"]
    image = got["body"]["messages"][0]["content"][0]["image_url"]["url"]
    assert status == 200 and image == "data:image/png;base64," + base64.b64encode(PNG).decode()
    assert out["missing.png"][0] == 400 and "HTTP 404" in out["missing.png"][1]["error"]["message"]
    assert out["slow.png"][0] == 400 and "no answer within" in out["slow.png"][1]["error"]["message"]
    assert snap["served"] == {next(iter(snap["served"])): 1}  # the failed downloads never reached the engine
    base, out, _ = asyncio.run(run(False))
    assert out["cat.png"][1]["body"]["messages"][0]["content"][0]["image_url"]["url"] == f"{base}/cat.png"
    # The default: a loopback media host is refused with 400, and the engine is never asked.
    base, out, snap = asyncio.run(run(True, allow_private=False))
    status, got = out["cat.png"]
    assert status == 400 and "(loopback); refused" in got["error"]["message"]
    assert sum(snap["served"].values()) == 0


def closing_engine(close_first):
    """A raw HTTP/1.1 engine that reads the first ``close_first`` requests and closes their connections without a
    reply (an engine closing a pooled keep-alive connection as the proxy writes into it), then answers normally."""
    seen = {"n": 0}

    async def handle(reader, writer):
        while True:
            head = await reader.readuntil(b"\r\n\r\n")
            length = 0
            for line in head.decode().split("\r\n"):
                if line.lower().startswith("content-length:"):
                    length = int(line.split(":", 1)[1])
            await reader.readexactly(length)
            seen["n"] += 1
            if seen["n"] <= close_first:
                writer.close()
                return
            body = json.dumps({"backend": "R"}).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body) + body
            )
            await writer.drain()

    return handle, seen


@pytest.mark.parametrize("close_first, want_status", [(1, 200), (2, 500)])
def test_proxy_resends_once_after_an_unanswered_disconnect(close_first, want_status):
    async def run():
        handle, seen = closing_engine(close_first)
        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        router = lb.Router([f"http://127.0.0.1:{port}"])
        proxy = TestServer(lb.make_app(router), host="127.0.0.1")
        await proxy.start_server()
        try:
            async with ClientSession() as session:
                async with session.post(
                    f"http://127.0.0.1:{proxy.port}/v1/chat/completions",
                    data=chat("short", "q"),
                    headers={"Content-Type": "application/json"},
                ) as r:
                    status = r.status
            return status, seen["n"], router.snapshot()
        finally:
            await proxy.close()
            server.close()
            await server.wait_closed()

    status, requests, snap = asyncio.run(run())
    assert status == want_status
    assert requests == 2  # one resend, never more
    assert snap["prefix"]["stale_retried"] == 1


@pytest.mark.parametrize(
    "fmt",
    [
        {"type": "json_schema", "json_schema": {"name": "answer"}},
        {"type": "json_schema", "json_schema": {"name": "a", "strict": True, "schema": None}},
        {"type": "json_schema"},
    ],
)
def test_map_structured_outputs_schema_less_json_schema_is_json_mode(fmt):
    req = {"model": "m", "messages": [], "tools": [{"type": "function"}], "response_format": fmt}
    out = lb.map_structured_outputs(req)
    assert out["response_format"] == {"type": "json_object"} and out["tools"] == req["tools"]


@pytest.mark.parametrize(
    "req",
    [
        {"model": "m", "messages": []},
        {"model": "m", "messages": [], "response_format": {"type": "json_object"}},
        {
            "model": "m",
            "messages": [],
            "response_format": {"type": "json_schema", "json_schema": {"name": "a", "schema": {}}},
        },
        {"model": "m", "messages": [], "structured_outputs": {"choice": ["a", "b"]}},
    ],
)
def test_map_structured_outputs_leaves_valid_requests(req):
    assert lb.map_structured_outputs(req) is req


def test_map_structured_outputs_drops_empty_structured_outputs():
    assert "structured_outputs" not in lb.map_structured_outputs({"model": "m", "structured_outputs": {}})
    assert "structured_outputs" not in lb.map_structured_outputs({"model": "m", "structured_outputs": {"json": None}})
