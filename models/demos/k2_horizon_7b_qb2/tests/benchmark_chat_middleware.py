"""Native IFM assistant-history normalization via vLLM's ASGI extension.

The pinned template requires a string thinking field on assistant exemplars.
The pinned chat_utils accepts ``reasoning`` and forwards both spellings to the
template. Preserve supplied text; an upstream exemplar with no thinking field
has empty reasoning. No template is rendered by this middleware.
"""

import hashlib
import json
import os
from pathlib import Path

MODEL = "IFM/K2-Horizon-7B"


def normalize_assistant_history(payload):
    """Return a copied payload only when valid assistant messages need adapting.

    Invalid supplied reasoning types remain untouched for normal API/template
    validation. In particular, a supplied null is not silently replaced.
    """
    if not isinstance(payload, dict) or payload.get("model") != MODEL:
        return payload
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return payload
    normalized = []
    changed = False
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant" and "reasoning" not in message:
            if "reasoning_content" not in message:
                message = {**message, "reasoning": ""}
                changed = True
            elif isinstance(message["reasoning_content"], str):
                message = {**message, "reasoning": message["reasoning_content"]}
                changed = True
        normalized.append(message)
    return {**payload, "messages": normalized} if changed else payload


class K2BenchmarkChatMiddleware:
    """Adapt this model's chat bodies before Pydantic and native HF rendering."""

    def __init__(self, app):
        self.app = app
        if identity_path := os.environ.get("K2_BENCHMARK_CHAT_TRANSPORT_PATH"):
            source = Path(__file__).resolve()
            Path(identity_path).write_text(
                json.dumps(
                    {
                        "class": f"{type(self).__module__}.{type(self).__qualname__}",
                        "file": str(source),
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "model": MODEL,
                        "route": "POST /v1/chat/completions",
                        "normalization": "assistant_missing_reasoning_empty_or_copy_existing_reasoning_content_v1",
                        "template_rendered_by_middleware": False,
                    },
                    indent=2,
                )
                + "\n"
            )

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST" or scope.get("path") != "/v1/chat/completions":
            return await self.app(scope, receive, send)

        events = []
        while True:
            event = await receive()
            events.append(event)
            if event["type"] != "http.request" or not event.get("more_body", False):
                break

        if all(event["type"] == "http.request" for event in events):
            body = b"".join(event.get("body", b"") for event in events)
            try:
                original = json.loads(body)
            except (ValueError, UnicodeError):
                original = None
            normalized = normalize_assistant_history(original)
            if normalized is not original:
                body = json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                scope = dict(scope)
                scope["headers"] = [
                    (key, value) for key, value in scope.get("headers", []) if key.lower() != b"content-length"
                ] + [(b"content-length", str(len(body)).encode("ascii"))]
                events = [{"type": "http.request", "body": body, "more_body": False}]

        iterator = iter(events)

        async def replay():
            event = next(iterator, None)
            return event if event is not None else await receive()

        return await self.app(scope, replay, send)
