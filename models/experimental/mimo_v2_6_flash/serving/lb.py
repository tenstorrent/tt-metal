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

Multimodal requests in OpenRouter's forms (``Shim``; ``--no-request-shim`` turns it off):

- media parts are keyed by a digest of their payload (``MEDIA_KEY_CHARS`` characters each), not by their base64 text,
  so a 20 MB video is a few index entries instead of ~80K;
- chat requests that vLLM 0.30 would reject or misread are rewritten (``rewrite_chat``): ``input_audio`` in pcm16 /
  pcm24 becomes WAV, the legacy ``input_video`` part becomes ``video_url``, OpenRouter's ``reasoning`` object becomes
  ``chat_template_kwargs.enable_thinking`` / ``include_reasoning``, and a FLAC without its length gets it. A request
  needing none of that is forwarded byte for byte;
- vLLM's 422 (a media URL that could not be fetched) is answered as 400 with vLLM's message (``map_error``): OpenRouter
  counts 400 as the caller's error, 422 in neither list;
- a body above ``--max-request-mb`` is refused with 413.

Usage: lb.py --port 8000 --backends 127.0.0.1:8001,127.0.0.1:8002,...
"""
import argparse
import asyncio
import base64
import binascii
import dataclasses
import hashlib
import io
import itertools
import json
import re
import time
import wave
from collections import OrderedDict

from aiohttp import ClientConnectorError, ClientSession, ClientTimeout, TCPConnector, web

HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "content-encoding", "upgrade", "host"}
PREFIX_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/responses")
SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?(\s.*)$")  # name, labels, value [timestamp]
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')
PROBE_TIMEOUT_S = 10.0  # an engine's /health or /metrics answer; generation requests have no timeout
OFFLOAD_BYTES = 256 << 10  # bodies above this are parsed / rewritten / keyed in a worker thread, off the event loop

# Content-part keys whose value is media: a URL or base64 string, or a dict holding one under "url" / "data".
MEDIA_KEYS = ("image_url", "video_url", "audio_url", "input_audio", "file_data")
# Characters a media part occupies in a prefix key: one item alone reaches the default --min-match-chars (1024), so a
# repeated image / video / clip is routed to the engine that holds its encoder output and KV, while a 20 MB video costs
# 4 index entries (of 256 chars) instead of ~80K.
MEDIA_KEY_CHARS = 1024


def media_key(payload):
    """A media payload (URL or base64 / data URI string) as a fixed-size prefix key: its sha256, repeated."""
    tag = "<media sha256:" + hashlib.sha256(payload.encode("utf-8", "surrogatepass")).hexdigest() + ">"
    return (tag * (MEDIA_KEY_CHARS // len(tag) + 1))[:MEDIA_KEY_CHARS]


def media_key_view(obj):
    """``obj`` (messages / Responses input) with every media payload replaced by ``media_key``. Returns ``obj`` itself
    when it holds no media, so a text-only request keys exactly as before."""
    if isinstance(obj, list):
        out = [media_key_view(x) for x in obj]
        return obj if all(a is b for a, b in zip(out, obj)) else out
    if not isinstance(obj, dict):
        return obj
    out = None
    for key, value in obj.items():
        new = value
        if key in MEDIA_KEYS:
            if isinstance(value, str):
                new = media_key(value)
            elif isinstance(value, dict):
                new = {k: media_key(v) if k in ("url", "data") and isinstance(v, str) else v for k, v in value.items()}
        elif isinstance(value, (list, dict)):
            new = media_key_view(value)
        if new is not value:
            if out is None:
                out = dict(obj)
            out[key] = new
    return obj if out is None else out


def parse_json(body):
    """The request body as a dict, or None (not JSON, or not an object)."""
    try:
        req = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    return req if isinstance(req, dict) else None


def prefix_text(path, body):
    """The text that determines a request's prompt prefix, in prompt order, or None (no prefix routing)."""
    req = parse_json(body)
    return None if req is None else prefix_text_of(path, req)


def prefix_text_of(path, req):
    """prefix_text of a parsed request. Media payloads count as ``media_key`` text (``media_key_view``)."""
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
        messages = media_key_view(messages)
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
        text = inp if isinstance(inp, str) else json.dumps(media_key_view(inp), sort_keys=True, ensure_ascii=False)
        return head + json.dumps(req.get("instructions")) + text
    return None


# ---------------------------------------------------------------- the multimodal request shim (pure functions)


class ShimError(ValueError):
    """A request the shim cannot rewrite: answered 400 with this message (vLLM would reject it too)."""


def pcm_to_wav(raw, sample_width, sample_rate, channels=1):
    """Headerless little-endian signed PCM (``sample_width`` bytes per sample, interleaved channels) as a WAV file."""
    frame = sample_width * channels
    if len(raw) == 0 or len(raw) % frame:
        raise ShimError(f"input_audio: {len(raw)} bytes of PCM is not a whole number of {frame}-byte frames")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(raw)
    return buf.getvalue()


def _b64decode(data, what):
    if data.startswith("data:"):  # some clients send a data URI in input_audio.data
        data = data.split(",", 1)[-1]
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ShimError(f"{what} is not valid base64: {exc}") from None


def flac_total_samples(data):
    """The total-samples field of a FLAC file's STREAMINFO (0 = unknown), or None when ``data`` is not FLAC."""
    if len(data) < 26 or data[:4] != b"fLaC" or data[4] & 0x7F != 0:  # STREAMINFO must be the first block
        return None
    return int.from_bytes(data[18:26], "big") & ((1 << 36) - 1)


def fix_flac_length(data):
    """A FLAC whose STREAMINFO says "length unknown" (an encoder writing to a pipe cannot seek back to fill it in) with
    the length filled in from its frames, or None when there is nothing to fix. vLLM 0.30 decodes FLAC with libsndfile,
    which reads an unknown length as 2^63 samples and fails its duration guard ("Audio exceeds maximum allowed
    duration ... 418293516410647.4s"). Counting the frames needs only PyAV's demuxer (no decode); without PyAV the file
    is left as it is."""
    if flac_total_samples(data) != 0:
        return None
    try:
        import av
    except ImportError:
        return None
    total = 0
    try:
        with av.open(io.BytesIO(data), format="flac") as container:
            stream = container.streams.audio[0]
            for packet in container.demux(stream):
                total += packet.duration or 0
    except Exception:  # noqa: BLE001 - an unreadable file stays as it is; vLLM answers it
        return None
    if not 0 < total < (1 << 36):
        return None
    fixed = bytearray(data)
    head = int.from_bytes(data[18:26], "big") & ~((1 << 36) - 1)
    fixed[18:26] = (head | total).to_bytes(8, "big")
    return bytes(fixed)


def _rewrite_part(part, shim):
    """One content part rewritten for vLLM 0.30, or the same object when it needs nothing."""
    if not isinstance(part, dict):
        return part
    kind = part.get("type")
    if kind == "input_video":
        # OpenRouter's legacy video part: the same payload as video_url (vLLM knows only video_url).
        video = part.get("video_url")
        if isinstance(video, str):
            video = {"url": video}
        new = {k: v for k, v in part.items() if k not in ("type", "video_url")}
        new.update(type="video_url", video_url=video)
        return new
    if kind == "input_audio" and isinstance(part.get("input_audio"), dict):
        audio = part["input_audio"]
        fmt = str(audio.get("format", "")).lower()
        data = audio.get("data")
        if not isinstance(data, str):
            return part
        if fmt in ("pcm16", "pcm24"):
            raw = _b64decode(data, "input_audio.data")
            wav = pcm_to_wav(raw, 2 if fmt == "pcm16" else 3, shim.pcm_sample_rate, shim.pcm_channels)
            return dict(part, input_audio=dict(audio, data=base64.b64encode(wav).decode("ascii"), format="wav"))
        if fmt == "flac" and shim.fix_flac:
            fixed = fix_flac_length(_b64decode(data, "input_audio.data"))
            if fixed is not None:
                return dict(part, input_audio=dict(audio, data=base64.b64encode(fixed).decode("ascii")))
    return part


def map_reasoning(req):
    """OpenRouter's ``reasoning`` object as the fields vLLM 0.30 and MiMo's chat template read, or ``req`` itself.

    - ``enabled: false``, ``effort: "none"`` or ``max_tokens: 0`` -> ``chat_template_kwargs.enable_thinking = false``;
      ``enabled: true`` or any other effort -> true. Never over an explicit ``chat_template_kwargs.enable_thinking``,
      and not when the top-level ``reasoning_effort`` is set (vLLM already maps that one).
    - ``exclude: true`` -> ``include_reasoning: false`` (OpenRouter: the model reasons, the reasoning is not returned;
      vLLM implements that field natively), unless the request sets ``include_reasoning`` itself.
    A top-level ``include_reasoning`` passes through: it hides the reasoning, it does not switch thinking off."""
    reasoning = req.get("reasoning")
    if not isinstance(reasoning, dict):
        return req
    out = req
    kwargs = req.get("chat_template_kwargs")
    explicit = isinstance(kwargs, dict) and "enable_thinking" in kwargs
    if not explicit and req.get("reasoning_effort") is None:
        on = None
        if reasoning.get("enabled") is not None:
            on = bool(reasoning["enabled"])
        if reasoning.get("effort") is not None:
            on = reasoning["effort"] != "none"
        if reasoning.get("max_tokens") == 0:
            on = False
        if on is not None:
            out = dict(out, chat_template_kwargs=dict(kwargs or {}, enable_thinking=on))
    if reasoning.get("exclude") is True and "include_reasoning" not in req:
        out = dict(out, include_reasoning=False)
    return out


def rewrite_chat(req, shim):
    """A chat-completions request rewritten for vLLM 0.30 (``_rewrite_part`` on every content part, ``map_reasoning``),
    or ``req`` itself when nothing applies. Raises ShimError for a request that cannot be rewritten."""
    out = map_reasoning(req)
    messages = out.get("messages")
    if isinstance(messages, list):
        new_messages = []
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list):
                parts = [_rewrite_part(part, shim) for part in content]
                if any(a is not b for a, b in zip(parts, content)):
                    message = dict(message, content=parts)
            new_messages.append(message)
        if any(a is not b for a, b in zip(new_messages, messages)):
            out = dict(out, messages=new_messages)
    return out


def map_error(status, body):
    """An engine's error answer as the proxy returns it: vLLM's 422 (a media URL it could not fetch: 404, invalid URL,
    too large) becomes 400 with vLLM's message, since OpenRouter counts 400 as the caller's error and lists 422 nowhere.
    Everything else is unchanged."""
    if status != 422:
        return status, body
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return 400, body
    err = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(err, dict):
        err["code"] = 400
        err["type"] = "BadRequestError"
    return 400, json.dumps(payload).encode()


def error_body(status, message, kind):
    return json.dumps({"error": {"message": message, "type": kind, "code": status}}).encode()


@dataclasses.dataclass
class Shim:
    enabled: bool = True
    # OpenRouter documents pcm16 / pcm24 only by name. Its one stated PCM convention (the speech-to-text endpoint's
    # "pcm": 16-bit LE mono at 16 kHz) is for another format name; pcm16 / pcm24 follow OpenAI's audio format names, where
    # pcm16 is 24 kHz mono. Assumed: 24 kHz mono, little-endian, signed (--pcm-sample-rate to change).
    pcm_sample_rate: int = 24000
    pcm_channels: int = 1
    fix_flac: bool = True
    max_request_bytes: int = 256 << 20


def prepare(path, body, shim, chunk_chars):
    """(body to forward, prefix hashes) for a POST: the shim's rewrite of a chat request (the original bytes when
    nothing changed) and its prefix keys. Raises ShimError (answered 400)."""
    req = parse_json(body) if path in PREFIX_PATHS else None
    if req is None:
        return body, []
    if shim.enabled and path == "/v1/chat/completions":
        new = rewrite_chat(req, shim)
        if new is not req:
            req = new
            body = json.dumps(req, ensure_ascii=False).encode()
    text = prefix_text_of(path, req)
    return body, ([] if text is None else chunk_hashes(text, chunk_chars))


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
        self.stats.update(shim_rejected=0, errors_mapped=0, too_large=0)

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


def make_app(router, session_factory=None, shim=None):
    shim = shim or Shim()
    state = {}

    async def proxy(request: web.Request):
        if request.content_length is not None and request.content_length > shim.max_request_bytes:
            return too_large()
        try:
            body = await request.read()
        except web.HTTPRequestEntityTooLarge:
            return too_large()
        hashes = []
        if request.method == "POST" and request.path in PREFIX_PATHS:
            try:
                if len(body) > OFFLOAD_BYTES:  # a media body: parsing and hashing it would stall every stream
                    body, hashes = await asyncio.to_thread(prepare, request.path, body, shim, router.chunk_chars)
                else:
                    body, hashes = prepare(request.path, body, shim, router.chunk_chars)
            except ShimError as exc:
                router.stats["shim_rejected"] += 1
                return web.Response(
                    status=400, body=error_body(400, str(exc), "BadRequestError"), content_type="application/json"
                )
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
                    if up.status == 422 and shim.enabled:
                        status, data = map_error(up.status, await up.read())
                        router.stats["errors_mapped"] += 1
                        return web.Response(status=status, body=data, content_type="application/json")
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

    def too_large():
        router.stats["too_large"] += 1
        msg = f"request body exceeds {shim.max_request_bytes} bytes"
        return web.Response(
            status=413, body=error_body(413, msg, "RequestEntityTooLarge"), content_type="application/json"
        )

    async def stats(_):
        return web.json_response(router.snapshot())

    async def all_metrics(_):
        # An engine that fails or hangs leaves out only its own samples (its index keeps its place).
        texts, missing = [], []
        for i, b in enumerate(router.backends):
            try:
                async with state["session"].get(b + "/metrics", timeout=ClientTimeout(total=PROBE_TIMEOUT_S)) as up:
                    texts.append(await up.text())
            except Exception:  # noqa: BLE001 - any failure means no metrics from this engine
                texts.append("")
                missing.append(f"# engine {i} ({b}) did not answer\n")
        return web.Response(text="".join(missing) + merge_metrics(texts), content_type="text/plain")

    async def health(_):
        bad = []
        for b in router.backends:
            try:
                async with state["session"].get(b + "/health", timeout=ClientTimeout(total=PROBE_TIMEOUT_S)) as up:
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

    app = web.Application(client_max_size=shim.max_request_bytes)
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
    ap.add_argument("--max-request-mb", type=float, default=256, help="larger request bodies are refused with 413")
    ap.add_argument("--pcm-sample-rate", type=int, default=24000, help="assumed rate of input_audio pcm16 / pcm24")
    ap.add_argument("--no-request-shim", action="store_true", help="forward requests and errors unchanged")
    a = ap.parse_args()
    backends = [b if b.startswith("http") else f"http://{b}" for b in a.backends.split(",")]
    router = Router(backends, a.chunk_chars, a.min_match_chars, a.slack, a.index_chunks, a.down_s)
    shim = Shim(
        enabled=not a.no_request_shim,
        pcm_sample_rate=a.pcm_sample_rate,
        max_request_bytes=int(a.max_request_mb * 2**20),
    )
    web.run_app(make_app(router, shim=shim), host=a.host, port=a.port, access_log=None)


if __name__ == "__main__":
    main()
