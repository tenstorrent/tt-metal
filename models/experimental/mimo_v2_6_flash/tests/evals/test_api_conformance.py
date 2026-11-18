# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""OpenRouter-format API conformance against a running OpenAI-compatible endpoint (the online half; the offline half
is conformance_offline.py). Every case of mm_requests.CASES is sent as raw JSON, so the bytes on the wire are exactly
what OpenRouter would send. A case passes when the endpoint answers as the case wants -- 200 with a sane answer, or a
4xx with an error message -- and the server is still healthy afterwards. A 5xx or a dropped connection always fails
(OpenRouter counts both against uptime).

  BASE_URL=http://localhost:8000/v1 MODEL=XiaomiMiMo/MiMo-V2.6-Flash-MOPD \\
    pytest -q --confcutdir=models/experimental/mimo_v2_6_flash/tests/evals \\
      models/experimental/mimo_v2_6_flash/tests/evals/test_api_conformance.py

Env: BASE_URL (required; the suite is skipped without it), MODEL, API_KEY (or OPENROUTER_API_KEY), MM_TARGET
("vllm" default, or "openrouter": reasoning is switched off with OpenRouter's field and MM_PROVIDER pins providers),
MM_TIMEOUT (s, default 600), MM_ASSETS_DIR (media cache), MM_CONFORMANCE_LOG (append one JSONL record per request:
case id, status, latency, the request with media payloads replaced by their size and sha256, and the full response). Needs httpx, Pillow and ffmpeg; no tt-metal imports
(--confcutdir keeps the repo's root conftest out).
"""

import hashlib
import json
import os
import sys
import time

import pytest

# The repo's pytest.ini uses --import-mode=importlib, which does not put this directory on sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mm_requests  # noqa: E402

BASE_URL = os.environ.get("BASE_URL", "").rstrip("/")
MODEL = os.environ.get("MODEL", "XiaomiMiMo/MiMo-V2.6-Flash-MOPD")
TARGET = os.environ.get("MM_TARGET", "openrouter" if "openrouter.ai" in BASE_URL else "vllm")
API_KEY = os.environ.get("API_KEY") or os.environ.get("OPENROUTER_API_KEY") or "none"
TIMEOUT = float(os.environ.get("MM_TIMEOUT", "600"))
LOG = os.environ.get("MM_CONFORMANCE_LOG")


def _media_ref(value):
    return f"<{len(value)} chars sha256:{hashlib.sha256(value.encode()).hexdigest()[:16]}>"


def _log(case_id, req, status, payload, latency):
    """Append the exchange to MM_CONFORMANCE_LOG (media payloads by size and hash)."""
    if not LOG:
        return
    logged = json.loads(json.dumps(req))
    for msg in logged.get("messages", []):
        for part in msg.get("content") if isinstance(msg.get("content"), list) else []:
            for key in ("image_url", "video_url", "input_video"):
                url = (part.get(key) or {}).get("url") if isinstance(part.get(key), dict) else None
                if url and url.startswith("data:"):
                    part[key]["url"] = url.split(",", 1)[0] + "," + _media_ref(url)
            audio = part.get("input_audio")
            if isinstance(audio, dict) and isinstance(audio.get("data"), str):
                audio["data"] = _media_ref(audio["data"])
    rec = {"case": case_id, "status": status, "latency_s": round(latency, 3), "request": logged, "response": payload}
    with open(LOG, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


pytestmark = pytest.mark.skipif(not BASE_URL, reason="set BASE_URL to the endpoint's /v1 URL")


@pytest.fixture(scope="session")
def http():
    httpx = pytest.importorskip("httpx")
    with httpx.Client(timeout=TIMEOUT, headers={"Authorization": f"Bearer {API_KEY}"}) as client:
        yield client


@pytest.fixture(scope="session")
def media():
    return mm_requests.Media()


def _health(http):
    """vLLM's /health (the server root, not /v1). OpenRouter has none; a text request stands in."""
    if TARGET == "openrouter":
        return True
    root = BASE_URL[: -len("/v1")] if BASE_URL.endswith("/v1") else BASE_URL
    try:
        return http.get(root + "/health", timeout=30).status_code == 200
    except Exception:
        return False


def _post(http, req):
    """POST a chat completion. Returns (status, json or None, answer text, extras) and reads SSE for stream=True."""
    url = BASE_URL + "/chat/completions"
    if not req.get("stream"):
        r = http.post(url, json=req)
        try:
            payload = r.json()
        except ValueError:
            payload = None
        return r.status_code, payload
    chunks, usage, finish, text, reasoning, status, error = [], None, None, [], [], None, None
    with http.stream("POST", url, json=req) as r:
        status = r.status_code
        if status != 200:
            r.read()
            try:
                return status, r.json()
            except ValueError:
                return status, None
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            chunks.append(chunk)
            if "error" in chunk:
                error = chunk["error"]  # a mid-stream error: counts against uptime on OpenRouter
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                text.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
                finish = choice.get("finish_reason") or finish
    message = {"content": "".join(text), "reasoning_content": "".join(reasoning)}
    payload = {"choices": [{"message": message, "finish_reason": finish}], "usage": usage, "_chunks": len(chunks)}
    if error is not None:
        payload["error"] = error
    return status, payload


def _check_ok(case, payload):
    assert payload is not None and payload.get("choices"), f"no choices in {str(payload)[:300]}"
    assert "error" not in payload, f"error in a 200 response / mid-stream: {payload['error']}"
    choice = payload["choices"][0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    usage = payload.get("usage") or {}
    assert usage.get("prompt_tokens", 0) > 0 and "completion_tokens" in usage, f"usage missing: {usage}"
    finish = choice.get("finish_reason")
    assert finish in ("stop", "length", "tool_calls"), f"finish_reason {finish!r}"
    if case.kind == "tools":
        calls = message.get("tool_calls") or []
        assert calls, f"no tool call; content={content[:200]!r}"
        call = calls[0]["function"]
        assert call["name"] == "record_color", call
        args = json.loads(call["arguments"])  # OpenRouter's InvalidJson bucket
        assert isinstance(args.get("color"), str), args  # ... and its SchemaMismatch bucket (required string)
        content = args["color"]
    if case.kind == "reasoning_on":
        assert reasoning.strip(), "reasoning requested but none returned"
    if case.kind == "reasoning_off":
        assert not reasoning.strip(), f"reasoning switched off but returned: {reasoning[:200]!r}"
    if case.expect_any and finish != "length":
        low = content.lower()
        assert any(word in low for word in case.expect_any), f"answer {content[:300]!r} has none of {case.expect_any}"


@pytest.mark.timeout(TIMEOUT + 120)
@pytest.mark.parametrize("case", mm_requests.CASES, ids=mm_requests.CASE_IDS)
def test_openrouter_form(case, http, media):
    try:
        req = mm_requests.apply_target(case.build(media), case, MODEL, TARGET)
    except RuntimeError as exc:  # ffmpeg missing or a download failed: a fixture problem, not an endpoint answer
        pytest.skip(f"fixture: {exc}")
    t0 = time.time()
    status, payload = _post(http, req)
    _log(case.id, req, status, payload, time.time() - t0)
    try:
        assert status < 500, f"{status} (5xx counts against uptime): {str(payload)[:400]}"
        if case.want == "ok":
            assert status == 200, f"want 200, got {status}: {str(payload)[:400]}"
        elif case.want == "4xx":
            assert 400 <= status < 500, f"want a 4xx, got {status}: {str(payload)[:400]}"
        if status == 200:
            _check_ok(case, payload)
        else:
            err = (payload or {}).get("error") or {}
            message = err.get("message") if isinstance(err, dict) else str(err)
            assert message, f"{status} without an error message: {payload}"
    finally:
        assert _health(http), "server unhealthy after this request"


def test_still_serves_text_after_suite(http):
    """Last: a plain text request after all the media cases (an engine killed by a prefill exception answers here)."""
    case = mm_requests.CASES[0]
    req = mm_requests.apply_target(case.build(None), case, MODEL, TARGET)
    t0 = time.time()
    status, payload = _post(http, req)
    _log("text_after_suite", req, status, payload, time.time() - t0)
    assert status == 200, payload
    _check_ok(case, payload)
