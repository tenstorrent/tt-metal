# SPDX-License-Identifier: Apache-2.0
"""Log complete eval JSON after delivery without buffering streaming responses."""
import json
import logging

LOGGER = logging.getLogger("vllm.entrypoints.serve.middleware.log_response")


class ResponseLoggingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") not in ("/v1/chat/completions", "/v1/completions"):
            return await self.app(scope, receive, send)
        capture = False
        complete = False
        parts = []

        async def forward(message):
            nonlocal capture, complete
            # Delivery precedes accounting. In particular SSE chunks are never
            # held until generation finishes, unlike vLLM's debug HTTP logger.
            await send(message)
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                capture = message["status"] == 200 and headers.get(b"content-type", b"").startswith(b"application/json")
            elif message["type"] == "http.response.body" and capture:
                parts.append(message.get("body", b""))
                complete = not message.get("more_body", False)

        await self.app(scope, receive, forward)
        if capture and complete:
            try:
                body = b"".join(parts).decode("utf-8")
                parsed = json.loads(body)
                if parsed.get("id") and parsed.get("choices") and parsed.get("usage"):
                    LOGGER.info("response_body={%s}", body)
            except Exception:
                # Logging occurs after successful delivery and cannot change it.
                pass
