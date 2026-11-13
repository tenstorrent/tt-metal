# SPDX-License-Identifier: Apache-2.0
import asyncio
import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "response_logging", Path(__file__).resolve().parents[1] / "tt/response_logging.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_delivered_before_app_finishes(self):
        delivered = asyncio.Event()
        messages = []

        async def send(message):
            messages.append(message)
            if message.get("body") == b"first":
                delivered.set()

        async def app(scope, receive, send):
            await send(
                {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]}
            )
            await send({"type": "http.response.body", "body": b"first", "more_body": True})
            await asyncio.wait_for(delivered.wait(), 1)
            await send({"type": "http.response.body", "body": b"last", "more_body": False})

        with patch.object(MODULE.LOGGER, "info") as log:
            await MODULE.ResponseLoggingMiddleware(app)({"type": "http", "path": "/v1/chat/completions"}, None, send)
            log.assert_not_called()
        self.assertEqual([m.get("body") for m in messages], [None, b"first", b"last"])

    async def test_split_unicode_json_complete_after_delivery(self):
        body = json.dumps(
            {
                "id": "test",
                "choices": [{"message": {"content": "café"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 2},
            },
            ensure_ascii=False,
        ).encode()
        split = body.index("é".encode()) + 1
        sent = []

        async def send(message):
            sent.append(message)

        async def app(scope, receive, send):
            await send(
                {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]}
            )
            await send({"type": "http.response.body", "body": body[:split], "more_body": True})
            await send({"type": "http.response.body", "body": body[split:], "more_body": False})

        with patch.object(MODULE.LOGGER, "info") as log:
            await MODULE.ResponseLoggingMiddleware(app)({"type": "http", "path": "/v1/chat/completions"}, None, send)
            self.assertEqual(json.loads(log.call_args.args[1])["id"], "test")
        self.assertEqual(b"".join(m.get("body", b"") for m in sent), body)

    async def test_delivery_failure_propagates_without_complete_log(self):
        async def send(message):
            raise OSError("disconnected")

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})

        with patch.object(MODULE.LOGGER, "info") as log:
            with self.assertRaises(OSError):
                await MODULE.ResponseLoggingMiddleware(app)(
                    {"type": "http", "path": "/v1/chat/completions"}, None, send
                )
            log.assert_not_called()

    async def test_non_http_passthrough(self):
        seen = []

        async def send(message):
            seen.append(message)

        async def app(scope, receive, send):
            await send({"type": "websocket.accept"})

        await MODULE.ResponseLoggingMiddleware(app)({"type": "websocket"}, None, send)
        self.assertEqual(seen, [{"type": "websocket.accept"}])


if __name__ == "__main__":
    unittest.main()
