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
back unchanged, except a chat answer left wholly in ``reasoning`` (``AnswerFromReasoning``, below).

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
- with ``--thinking-greedy T,P`` (env ``TT_PROXY_THINKING_GREEDY``, off by default) a chat request asking for
  temperature 0 with thinking on is sampled at temperature T, top-p P instead (``map_greedy_thinking``; the model card's
  1.0,0.95), as MiMo's GPU providers on OpenRouter serve it: in thinking mode GMICloud and Xiaomi sample whatever
  temperature a request asks for (with thinking off both decode greedily), while MiMo's greedy reasoning loops until
  the token limit on hard long-video questions;
- vLLM's 422 (a media URL that could not be fetched) is answered as 400 with vLLM's message (``map_error``): OpenRouter
  counts 400 as the caller's error, 422 in neither list;
- a chat answer that stops normally with all its text in ``reasoning`` and no ``content`` and no tool call gets that
  text as its ``content`` too (``AnswerFromReasoning``; ``--no-answer-from-reasoning`` / env
  ``TT_PROXY_ANSWER_FROM_REASONING=0`` turns it off). With thinking on, the checkpoint answers a transcription-only
  request as ``<think><think>the words<|im_end|>``: it never closes the block (the fp32 reference gives ``</think>`` a
  log-probability of -17 there), so vLLM's ``mimo`` reasoning parser files the whole answer under reasoning and
  OpenRouter's input-audio-transcription check reads an empty completion;
- a body above ``--max-request-mb`` is refused with 413;
- with ``--fetch-media`` (env ``TT_PROXY_FETCH_MEDIA=1``) the proxy itself downloads a chat request's http(s) image /
  video / audio URLs (bounded time and size) and inlines them as data URIs, answering 400 when a download fails: vLLM
  0.30 turns a media host's 5xx, 408, 429, timeout or DNS failure into a 500, which OpenRouter counts against uptime.
  It fetches only from public unicast addresses (``blocked_reason``), checks every redirect hop again (at most
  ``MAX_MEDIA_REDIRECTS``) and connects only to the addresses it checked (``PinnedResolver``). Serve vLLM with
  ``--allowed-media-domains`` naming no real host so the engines never fetch a URL themselves.

Each setting also reads an env var (``TT_PROXY_*``, see ``Shim.from_env``), the names tt-inference-server's copy of this
proxy (``vllm-tt-metal/src/engine_proxy.py``) uses.

Usage: lb.py --port 8000 --backends 127.0.0.1:8001,127.0.0.1:8002,...
"""
import argparse
import asyncio
import base64
import binascii
import dataclasses
import hashlib
import io
import ipaddress
import itertools
import json
import mimetypes
import os
import re
import socket
import time
import urllib.parse
import wave
from functools import partial
from collections import OrderedDict

from aiohttp import (
    ClientConnectorError,
    ClientError,
    ClientSession,
    ClientTimeout,
    ServerDisconnectedError,
    TCPConnector,
    web,
)
from aiohttp.abc import AbstractResolver

HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "content-encoding", "upgrade", "host"}
PREFIX_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/responses")
# Idle upstream connections are closed by the proxy before the engine's HTTP server closes them (uvicorn's default
# keep-alive is 5 s): a request written into a connection the engine is closing gets "Server disconnected" and no reply.
UPSTREAM_KEEPALIVE_S = 2.0
SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?(\s.*)$")  # name, labels, value [timestamp]
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')
PROBE_TIMEOUT_S = 10.0  # an engine's /health or /metrics answer; generation requests have no timeout
OFFLOAD_BYTES = 256 << 10  # bodies above this are parsed / rewritten / keyed in a worker thread

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


def map_greedy_thinking(req, sampling):
    """``req`` with ``sampling`` = (temperature, top_p) in place of a requested temperature 0 when thinking is on (MiMo's
    chat template thinks unless ``chat_template_kwargs.enable_thinking`` is false; run after ``map_reasoning``), or
    ``req`` itself. ``sampling`` None (the default) never changes a request.

    Why: MiMo's greedy reasoning on a question it cannot settle (VGI-Bench's false-premise questions over a 130K-token
    video) re-checks the same evidence until the token limit (on one row engine, 2 to 4 of the 5 smoke questions with
    BFP4 experts, also with GPTQ experts and BFP8 in the last layer), and so does it at temperature 0.05 or 0.2; the
    model card asks for 1.0 / 0.95. MiMo's GPU providers on OpenRouter already sample in thinking mode whatever the
    request's temperature (GMICloud: four different answers to one temperature-0 request with thinking on, one answer
    with thinking off), so OpenRouter's temperature-0 runs of MiMo are sampled runs."""
    if sampling is None:
        return req
    temperature = req.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or temperature != 0:
        return req
    kwargs = req.get("chat_template_kwargs")
    if isinstance(kwargs, dict) and kwargs.get("enable_thinking") is False:
        return req
    return dict(req, temperature=sampling[0], top_p=sampling[1])


def map_structured_outputs(req):
    """``req`` with a schema-less ``json_schema`` response format as JSON mode and an empty ``structured_outputs``
    dropped, or ``req`` itself.

    vLLM 0.30 builds a structured-outputs constraint with every field empty from ``response_format: {"type":
    "json_schema", "json_schema": {...}}`` without ``schema`` (optional in the OpenAI API) and answers 400 ("You must
    use one kind of structured outputs constraint but none are specified"); OpenRouter's tool-call checks send exactly
    that. Same rule as tt-inference-server's engine_proxy (``normalize_structured_outputs``)."""
    out = req
    fmt = req.get("response_format")
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
        schema = fmt.get("json_schema")
        if not isinstance(schema, dict) or schema.get("schema") is None:
            out = dict(out, response_format={"type": "json_object"})
    so = req.get("structured_outputs")
    if isinstance(so, dict) and all(v in (None, False, "", [], {}) for v in so.values()):
        out = {k: v for k, v in out.items() if k != "structured_outputs"}
    return out


def rewrite_chat(req, shim):
    """A chat-completions request rewritten for vLLM 0.30 (``_rewrite_part`` on every content part, ``map_reasoning``,
    ``map_greedy_thinking``), or ``req`` itself when nothing applies. Raises ShimError for a request that cannot be
    rewritten."""
    out = map_greedy_thinking(map_reasoning(map_structured_outputs(req)), shim.thinking_greedy)
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


# Content-part keys whose http(s) URL the proxy may download and inline (OpenRouter's audio is base64 only; vLLM's own
# audio_url form can be a URL).
REMOTE_MEDIA_KEYS = ("image_url", "video_url", "audio_url")


def _part_url(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("url"), str):
        return value["url"]
    return None


def remote_media_urls(req):
    """The distinct http(s) URLs of a chat request's image / video / audio parts, in prompt order."""
    urls = []
    for message in req.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        for part in content if isinstance(content, list) else []:
            if not isinstance(part, dict):
                continue
            for key in REMOTE_MEDIA_KEYS:
                url = _part_url(part.get(key))
                if url and url.lower().startswith(("http://", "https://")) and url not in urls:
                    urls.append(url)
    return urls


def inline_media(req, data_uris):
    """``req`` with every media URL in ``data_uris`` ({url: data URI}) replaced by its data URI (``req`` untouched)."""
    messages = []
    for message in req.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    for key in REMOTE_MEDIA_KEYS:
                        url = _part_url(part.get(key))
                        if url in data_uris:
                            value = part[key]
                            new = data_uris[url] if isinstance(value, str) else dict(value, url=data_uris[url])
                            part = dict(part, **{key: new})
                parts.append(part)
            message = dict(message, content=parts)
        messages.append(message)
    return dict(req, messages=messages)


def media_type(content_type, url):
    """The MIME type for a downloaded file's data URI: the server's media Content-Type, else a guess from the URL's
    extension, else application/octet-stream (vLLM decodes by content, not by the declared type)."""
    declared = (content_type or "").split(";", 1)[0].strip().lower()
    if declared.startswith(("image/", "video/", "audio/")):
        return declared
    guessed = mimetypes.guess_type(url.split("?", 1)[0])[0]
    return guessed or "application/octet-stream"


# Redirects the downloader follows per URL; each hop's host is resolved and checked again.
MAX_MEDIA_REDIRECTS = 3
REDIRECT_STATUSES = (301, 302, 303, 307, 308)
# Hosts whose URLs are a page around a video, not the video: OpenRouter hands a YouTube link to the provider as it
# is, and only a provider that resolves such links itself can take one. Refused by name, before any download.
PAGE_HOSTS = ("youtube.com", "youtu.be", "youtube-nocookie.com")
# A 200 answer of these types is a web page, which no media decoder opens (the engine would answer "Could not open
# video stream" for it).
PAGE_TYPES = ("text/html", "application/xhtml+xml")


def page_host(host):
    """The PAGE_HOSTS entry ``host`` is or is a subdomain of, else None."""
    host = (host or "").lower().rstrip(".")
    return next((h for h in PAGE_HOSTS if host == h or host.endswith("." + h)), None)


def blocked_reason(ip):
    """Why the proxy refuses to download from address ``ip``, or None for a public unicast address. A public endpoint
    that fetches arbitrary URLs is a probe into the network it runs in, so only globally routable unicast addresses
    are fetched: not loopback, private (RFC 1918, fc00::/7), link-local (169.254/16 with cloud metadata, fe80::/10),
    multicast, unspecified, reserved or otherwise non-global (100.64/10, documentation ranges), and not an IPv4
    address of those kinds inside an IPv6 one (::ffff:a.b.c.d, 6to4, Teredo)."""
    try:
        addr = ipaddress.ip_address(str(ip).split("%", 1)[0])
    except ValueError:
        return "not an IP address"
    if addr.version == 6 and addr.ipv4_mapped is not None:
        # ::ffff:a.b.c.d connects to a.b.c.d: judge that address (Python versions
        # disagree on the mapped form's own flags).
        addr = addr.ipv4_mapped
    if addr.version == 6:
        embedded = addr.sixtofour or (addr.teredo[1] if addr.teredo else None)
        if embedded is not None and blocked_reason(str(embedded)):
            return f"embeds {embedded}: {blocked_reason(str(embedded))}"
    for name, label in (
        ("is_unspecified", "unspecified"),
        ("is_loopback", "loopback"),
        ("is_link_local", "link-local"),
        ("is_multicast", "multicast"),
        ("is_private", "private"),
        ("is_reserved", "reserved"),
    ):
        if getattr(addr, name):
            return label
    if not addr.is_global:
        return "not globally routable"
    return None


async def resolve_host(host, port):
    """The addresses ``host`` resolves to, in getaddrinfo order without repeats (an IP literal is itself)."""
    try:
        return [str(ipaddress.ip_address(host))]
    except ValueError:
        pass
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(info[4][0] for info in infos))


class PinnedResolver(AbstractResolver):
    """An aiohttp resolver that answers only the checked host, with only its checked addresses: the connection goes to
    an address that passed ``blocked_reason``, never to a second, later DNS answer (DNS rebinding)."""

    def __init__(self, host, addresses):
        self.host, self.addresses = host, list(addresses)

    async def resolve(self, host, port=0, family=socket.AF_INET):
        if host != self.host:
            raise OSError(f"{host}: not the host the proxy checked ({self.host})")
        out = []
        for address in self.addresses:
            fam = socket.AF_INET6 if ":" in address else socket.AF_INET
            if family in (socket.AF_INET, socket.AF_INET6) and fam != family:
                continue
            out.append({"hostname": host, "host": address, "port": port, "family": fam, "proto": 0, "flags": 0})
        if not out:
            raise OSError(f"{host}: no checked address of family {family}")
        return out

    async def close(self):
        pass


async def _download(url, timeout_s, max_bytes, max_redirects, policy):
    """(bytes, MIME type) of one URL, following up to ``max_redirects`` redirects; every hop is checked (scheme, then
    every address its host resolves to) before anything connects to it, and connects only to those addresses."""
    where = url if len(url) <= 200 else url[:200] + "..."
    fail = "Failed to fetch media from URL: "
    current = url
    for _ in range(max_redirects + 1):
        try:
            parts = urllib.parse.urlsplit(current)
            port = parts.port
        except ValueError as exc:
            raise ShimError(f"{fail}invalid URL {current[:200]!r}: {exc} ({where})") from None
        scheme = parts.scheme.lower()
        if scheme not in ("http", "https") or not parts.hostname:
            raise ShimError(f"{fail}only http(s) URLs are fetched, not {current[:80]!r} ({where})")
        host = parts.hostname
        if page_host(host):
            raise ShimError(
                f"{fail}{page_host(host)} links are not supported, they are a web page and not a media file; "
                f"send a link to the file itself or a base64 data URI ({where})"
            )
        try:
            addresses = await resolve_host(host, port or (443 if scheme == "https" else 80))
        except OSError as exc:
            raise ShimError(f"{fail}cannot resolve {host}: {exc} ({where})") from None
        for address in addresses:
            reason = policy(address)
            if reason:
                raise ShimError(f"{fail}{host} resolves to {address} ({reason}); refused ({where})")
        connector = TCPConnector(resolver=PinnedResolver(host, addresses), use_dns_cache=False, limit=0)
        async with ClientSession(connector=connector, timeout=ClientTimeout(total=timeout_s)) as session:
            async with session.get(current, allow_redirects=False) as resp:
                if resp.status in REDIRECT_STATUSES:
                    location = resp.headers.get("Location")
                    if not location:
                        raise ShimError(f"{fail}HTTP {resp.status} without a Location ({where})")
                    current = urllib.parse.urljoin(current, location)
                    continue
                if resp.status != 200:
                    raise ShimError(f"{fail}HTTP {resp.status} error ({where})")
                declared = (resp.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
                if declared in PAGE_TYPES:
                    raise ShimError(
                        f"{fail}the URL answered a web page ({declared}), not a media file; "
                        f"send a link to the file itself or a base64 data URI ({where})"
                    )
                if resp.content_length is not None and resp.content_length > max_bytes:
                    raise ShimError(f"{fail}larger than {max_bytes} bytes ({where})")
                data = bytearray()
                async for chunk in resp.content.iter_chunked(1 << 16):
                    data += chunk
                    if len(data) > max_bytes:
                        raise ShimError(f"{fail}larger than {max_bytes} bytes ({where})")
                return bytes(data), media_type(resp.headers.get("Content-Type"), current)
    raise ShimError(f"{fail}more than {max_redirects} redirects ({where})")


async def fetch_media(urls, timeout_s, max_bytes, max_redirects=MAX_MEDIA_REDIRECTS, address_policy=None):
    """{url: data URI} for every URL, downloaded concurrently, each within ``timeout_s`` (all its hops) and
    ``max_bytes``. ``address_policy(ip)`` returns why an address is refused (default ``blocked_reason``). Raises
    ShimError (answered 400) naming the first URL that failed: refused address or scheme, a page host (YouTube) or a
    web page for an answer, too many redirects, non-200, too large, timeout, connection error."""
    policy = address_policy or blocked_reason

    async def one(url):
        where = url if len(url) <= 200 else url[:200] + "..."
        try:
            data, mime = await asyncio.wait_for(_download(url, timeout_s, max_bytes, max_redirects, policy), timeout_s)
        except asyncio.TimeoutError:
            raise ShimError(f"Failed to fetch media from URL: no answer within {timeout_s:g} s ({where})") from None
        except ClientError as exc:
            raise ShimError(f"Failed to fetch media from URL: {type(exc).__name__}: {exc} ({where})") from None
        encoded = await asyncio.get_running_loop().run_in_executor(None, base64.b64encode, data)
        return url, f"data:{mime};base64," + encoded.decode("ascii")

    results = await asyncio.gather(*(one(url) for url in urls), return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return dict(results)


REASONING_KEYS = ("reasoning", "reasoning_content")  # vLLM's field and its older name


def _reasoning_of(obj):
    return next((obj[k] for k in REASONING_KEYS if isinstance(obj.get(k), str) and obj[k]), "")


def answer_from_reasoning(body):
    """(body, number of choices changed) of a non-streamed chat completion: a choice that stopped normally
    (``finish_reason`` "stop") with blank ``content``, no tool call and text in ``reasoning`` gets that text as its
    ``content``; the reasoning stays. The original bytes when nothing changes or the body is not such a JSON."""
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body, 0
    choices = payload.get("choices") if isinstance(payload, dict) else None
    changed = 0
    for choice in choices if isinstance(choices, list) else []:
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict) or choice.get("finish_reason") != "stop":
            continue
        content = message.get("content")
        if (isinstance(content, str) and content.strip()) or (content and not isinstance(content, str)):
            continue
        if message.get("tool_calls") or message.get("function_call"):
            continue
        text = _reasoning_of(message)
        if text.strip():
            message["content"] = text
            changed += 1
    return (json.dumps(payload, ensure_ascii=False).encode(), changed) if changed else (body, 0)


class AnswerFromReasoning:
    """The same for a streamed chat completion: events pass through as they arrive, and a choice whose stream ends
    with ``finish_reason`` "stop" after reasoning deltas only gets one more event before its last, carrying the
    reasoning text as a ``content`` delta. The reasoning deltas were already sent, so the text is in both fields, as
    in the non-streamed answer. An incomplete event waits for its end; anything that is not a JSON ``data:`` event is
    forwarded as it is."""

    def __init__(self):
        self.buf = b""
        self.choices = {}  # index -> [reasoning deltas, answered]
        self.changed = 0

    def feed(self, chunk):
        """The bytes to send for an upstream chunk (b"" while an event is incomplete)."""
        self.buf += chunk
        out = []
        while True:
            # An SSE event ends with a blank line; vLLM writes "\n\n", sse-starlette "\r\n\r\n".
            cut = min((i + len(sep) for sep in (b"\n\n", b"\r\n\r\n") if (i := self.buf.find(sep)) >= 0), default=0)
            if not cut:
                break
            event, self.buf = self.buf[:cut], self.buf[cut:]
            extra = self._extra(event)
            if extra:
                out.append(extra)
            out.append(event)
        return b"".join(out)

    def flush(self):
        """What is left when the upstream closes (an event without its blank line)."""
        tail, self.buf = self.buf, b""
        return tail

    def _extra(self, event):
        """The event to send before ``event``: the answer of a choice this event finishes, or None."""
        line = event.strip()
        if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
            return None
        try:
            payload = json.loads(line[5:])
        except (ValueError, UnicodeDecodeError):
            return None
        choices = payload.get("choices") if isinstance(payload, dict) else None
        answers = []
        for choice in choices if isinstance(choices, list) else []:
            if not isinstance(choice, dict):
                continue
            state = self.choices.setdefault(choice.get("index", 0), [[], False])
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
            if delta.get("content") or delta.get("tool_calls") or delta.get("function_call"):
                state[1] = True
            text = _reasoning_of(delta)
            if text and not state[1]:
                state[0].append(text)
            if choice.get("finish_reason") == "stop" and not state[1]:
                answer = "".join(state[0])
                if answer.strip():
                    state[1] = True
                    answers.append(
                        {
                            "index": choice.get("index", 0),
                            "delta": {"content": answer},
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    )
        if not answers:
            return None
        self.changed += len(answers)
        head = {k: payload[k] for k in ("id", "object", "created", "model") if k in payload}
        return b"data: " + json.dumps(dict(head, choices=answers), ensure_ascii=False).encode() + b"\n\n"


def _env_flag(name, default):
    value = os.environ.get(name)
    return default if value in (None, "") else value.strip().lower() in ("1", "true", "yes", "on")


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
    # Download http(s) media in the proxy (off by default: a text-only deployment never needs it).
    fetch_media: bool = False
    fetch_timeout_s: float = 30.0
    fetch_max_bytes: int = 100 << 20
    # Also fetch from non-public addresses (loopback, private, link-local...): only for a deployment that is itself
    # private and fetches from an internal store. Never on a public endpoint.
    fetch_allow_private: bool = False
    # (temperature, top_p) for chat requests asking for temperature 0 with thinking on (map_greedy_thinking); None: off.
    thinking_greedy: tuple = None
    # Give a chat answer left wholly in reasoning its text as content too (answer_from_reasoning).
    answer_from_reasoning: bool = True

    @staticmethod
    def parse_sampling(value):
        """The string "T,P" (e.g. "1.0,0.95") -> (T, P); "" or None -> None."""
        if value in (None, ""):
            return None
        parts = str(value).split(",")
        try:
            t, p = (float(v) for v in parts) if len(parts) == 2 else (0.0, 0.0)
        except ValueError:
            t = p = 0.0
        if not (t > 0 and 0 < p <= 1):
            raise ValueError(f"thinking-greedy sampling {value!r}: want T,P with temperature T > 0, 0 < top_p P <= 1")
        return (t, p)

    @classmethod
    def from_env(cls, **overrides):
        """Defaults from the TT_PROXY_* env vars (as tt-inference-server sets them per model spec), then overrides."""
        env = dict(
            thinking_greedy=cls.parse_sampling(os.environ.get("TT_PROXY_THINKING_GREEDY")),
            enabled=_env_flag("TT_PROXY_REQUEST_SHIM", True),
            pcm_sample_rate=int(os.environ.get("TT_PROXY_PCM_SAMPLE_RATE") or 24000),
            max_request_bytes=int(float(os.environ.get("TT_PROXY_MAX_REQUEST_MB") or 256) * 2**20),
            fetch_media=_env_flag("TT_PROXY_FETCH_MEDIA", False),
            fetch_timeout_s=float(os.environ.get("TT_PROXY_FETCH_TIMEOUT_S") or 30),
            fetch_max_bytes=int(float(os.environ.get("TT_PROXY_FETCH_MAX_MB") or 100) * 2**20),
            fetch_allow_private=_env_flag("TT_PROXY_FETCH_ALLOW_PRIVATE", False),
            answer_from_reasoning=_env_flag("TT_PROXY_ANSWER_FROM_REASONING", True),
        )
        env.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**env)


def rewrite(path, body, shim):
    """(parsed request, its rewrite): (None, None) when the body is not a JSON object on a prefix path; the rewrite is
    the parsed request itself when the shim changes nothing."""
    req = parse_json(body) if path in PREFIX_PATHS else None
    if req is None:
        return None, None
    if shim.enabled and path == "/v1/chat/completions":
        return req, rewrite_chat(req, shim)
    return req, req


def finish(path, body, req, new, chunk_chars):
    """(body to forward, prefix hashes): the original bytes unless the request was rewritten."""
    if req is None:
        return body, []
    if new is not req:
        body = json.dumps(new, ensure_ascii=False).encode()
    text = prefix_text_of(path, new)
    return body, ([] if text is None else chunk_hashes(text, chunk_chars))


def prepare(path, body, shim, chunk_chars):
    """(body to forward, prefix hashes) for a POST, without media downloads: the shim's rewrite of a chat request (the
    original bytes when nothing changed) and its prefix keys. Raises ShimError (answered 400)."""
    req, new = rewrite(path, body, shim)
    return finish(path, body, req, new, chunk_chars)


async def prepare_async(path, body, shim, chunk_chars):
    """``prepare`` in the proxy: large bodies off the event loop, and with ``shim.fetch_media`` the chat request's
    http(s) media downloaded and inlined. Raises ShimError (answered 400)."""
    loop = asyncio.get_running_loop()

    async def run(big, fn, *args):
        return await loop.run_in_executor(None, partial(fn, *args)) if big else fn(*args)

    big = len(body) > OFFLOAD_BYTES  # parsing / hashing a media body inline would stall every stream
    req, new = await run(big, rewrite, path, body, shim)
    if new is not None and shim.enabled and shim.fetch_media and path == "/v1/chat/completions":
        urls = remote_media_urls(new)
        if urls:
            policy = (lambda _ip: None) if shim.fetch_allow_private else blocked_reason
            fetched = await fetch_media(urls, shim.fetch_timeout_s, shim.fetch_max_bytes, address_policy=policy)
            new = inline_media(new, fetched)
            big = True
    return await run(big, finish, path, body, req, new, chunk_chars)


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
        self.stats.update(shim_rejected=0, errors_mapped=0, too_large=0, stale_retried=0, answers_from_reasoning=0)

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
    shim = shim or Shim.from_env()
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
                body, hashes = await prepare_async(request.path, body, shim, router.chunk_chars)
            except ShimError as exc:
                router.stats["shim_rejected"] += 1
                return web.Response(
                    status=400, body=error_body(400, str(exc), "BadRequestError"), content_type="application/json"
                )
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        tried = []
        stale_retry = True
        while True:
            b = router.pick(hashes, exclude=tried)
            if b is None:
                return web.json_response({"error": "no backend accepted the request"}, status=503)
            router.inflight[b] += 1
            if hashes:
                # Now, not when the response starts: a request sent while this one prefills (or a non-streaming one,
                # whose headers come at its end) should find the prefix where it is being computed.
                router.index.record(hashes, b)
            started = False
            try:
                async with state["session"].request(
                    request.method, b + request.path_qs, data=body or None, headers=headers
                ) as up:
                    router.served[b] += 1
                    if up.status == 422 and shim.enabled:
                        status, data = map_error(up.status, await up.read())
                        router.stats["errors_mapped"] += 1
                        return web.Response(status=status, body=data, content_type="application/json")
                    headers_out = {k: v for k, v in up.headers.items() if k.lower() not in HOP}
                    fix = shim.enabled and shim.answer_from_reasoning and up.status == 200
                    fix = fix and request.method == "POST" and request.path == "/v1/chat/completions"
                    kind = up.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                    if fix and kind == "application/json":
                        started = True  # the engine answered: no resend from here on
                        data, changed = answer_from_reasoning(await up.read())
                        router.stats["answers_from_reasoning"] += changed
                        return web.Response(status=up.status, body=data, headers=headers_out)
                    answer = AnswerFromReasoning() if fix and kind == "text/event-stream" else None
                    resp = web.StreamResponse(status=up.status, headers=headers_out)
                    await resp.prepare(request)
                    started = True
                    async for chunk in up.content.iter_any():
                        if answer is not None:
                            chunk = answer.feed(chunk)
                        if chunk:
                            await resp.write(chunk)
                    if answer is not None:
                        tail = answer.flush()
                        if tail:
                            await resp.write(tail)
                        router.stats["answers_from_reasoning"] += answer.changed
                    await resp.write_eof()
                    return resp
            except ServerDisconnectedError:
                # The engine closed the connection before answering: almost always a pooled keep-alive connection it
                # was closing as we wrote into it (the engine never read the request). Resend once on a new
                # connection; a disconnect after the reply started cannot be retried.
                if started or not stale_retry:
                    raise
                stale_retry = False
                router.stats["stale_retried"] += 1
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
            timeout=ClientTimeout(total=None), connector=TCPConnector(limit=0, keepalive_timeout=UPSTREAM_KEEPALIVE_S)
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
    ap.add_argument("--max-request-mb", type=float, help="larger bodies are refused with 413 (default 256)")
    ap.add_argument("--pcm-sample-rate", type=int, help="assumed rate of input_audio pcm16 / pcm24 (default 24000)")
    ap.add_argument("--no-request-shim", action="store_true", help="forward requests and errors unchanged")
    ap.add_argument("--fetch-media", action="store_true", help="download http(s) media in the proxy (400 on failure)")
    ap.add_argument("--fetch-timeout-s", type=float, help="per-URL download limit (default 30)")
    ap.add_argument("--fetch-max-mb", type=float, help="per-URL size limit (default 100)")
    ap.add_argument(
        "--no-answer-from-reasoning",
        action="store_true",
        help="leave a chat answer that is wholly in reasoning without content",
    )
    ap.add_argument(
        "--thinking-greedy",
        help='"T,P": sample temperature-0 chat requests with thinking on at temperature T, top-p P (e.g. "1.0,0.95")',
    )
    a = ap.parse_args()
    backends = [b if b.startswith("http") else f"http://{b}" for b in a.backends.split(",")]
    router = Router(backends, a.chunk_chars, a.min_match_chars, a.slack, a.index_chunks, a.down_s)
    mib = lambda v: None if v is None else int(v * 2**20)  # noqa: E731
    shim = Shim.from_env(
        enabled=False if a.no_request_shim else None,
        pcm_sample_rate=a.pcm_sample_rate,
        max_request_bytes=mib(a.max_request_mb),
        fetch_media=True if a.fetch_media else None,
        fetch_timeout_s=a.fetch_timeout_s,
        fetch_max_bytes=mib(a.fetch_max_mb),
        thinking_greedy=Shim.parse_sampling(a.thinking_greedy),
        answer_from_reasoning=False if a.no_answer_from_reasoning else None,
    )
    web.run_app(make_app(router, shim=shim), host=a.host, port=a.port, access_log=None)


if __name__ == "__main__":
    main()
