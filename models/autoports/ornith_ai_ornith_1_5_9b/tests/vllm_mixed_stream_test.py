# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU transport checks for prompt-sized first SSE events and small followups."""

import importlib.util
import json
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread
from unittest.mock import patch

import requests
from urllib3.response import HTTPResponse

PROBE = Path(__file__).with_name("vllm_mixed_trace_probe.py")
spec = importlib.util.spec_from_file_location("mixed_stream_probe", PROBE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class MixedStreamTests(unittest.TestCase):
    def test_chunked_transport_delivers_large_then_small_events_before_next_chunk(self):
        first_seen, second_seen = Event(), Event()
        failures = []
        first = (
            b"data: "
            + json.dumps({"choices": [{"prompt_token_ids": [9707] * 32769, "token_ids": [11]}]}).encode()
            + b"\n\n"
        )
        second = b'data: {"choices": [{"token_ids": [12]}]}\n\n'

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def emit(data):
                    self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                    self.wfile.flush()

                try:
                    emit(first)
                    if not first_seen.wait(5):
                        raise AssertionError("First event waited for another HTTP chunk")
                    # Exercise an SSE line split across transport chunks.
                    emit(second[:13])
                    emit(second[13:])
                    if not second_seen.wait(5):
                        raise AssertionError("Small event waited for more bytes or EOF")
                    emit(b"data: [DONE]\n\n")
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except Exception as error:
                    failures.append(str(error))
                    self.close_connection = True

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        calls = []
        original = HTTPResponse._handle_chunk

        def handle_chunk(response, amt):
            calls.append(amt)
            return original(response, amt)

        try:
            with patch.object(HTTPResponse, "_handle_chunk", handle_chunk):
                with requests.get(
                    f"http://127.0.0.1:{server.server_port}",
                    headers={"Accept-Encoding": "identity"},
                    stream=True,
                    timeout=6,
                ) as response:
                    lines = (line for line in probe.stream_lines(response) if line.startswith(b"data: "))
                    initial = json.loads(next(lines)[6:])
                    self.assertEqual(len(initial["choices"][0]["prompt_token_ids"]), 32769)
                    self.assertEqual(initial["choices"][0]["token_ids"], [11])
                    # One transport read for ~196KB: no bytewise pending-line loop.
                    self.assertEqual(calls, [None])
                    first_seen.set()
                    self.assertEqual(json.loads(next(lines)[6:])["choices"][0]["token_ids"], [12])
                    self.assertEqual(calls, [None] * 3)
                    second_seen.set()
                    self.assertEqual(next(lines), b"data: [DONE]")
                    self.assertEqual(list(lines), [])
            self.assertEqual(calls, [None] * 4)
            self.assertEqual(failures, [])
        finally:
            first_seen.set()
            second_seen.set()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_nonchunked_response_rejected_before_read(self):
        response = requests.Response()
        response.headers["Content-Type"] = "text/event-stream"
        with patch.object(response, "iter_lines") as read:
            with self.assertRaisesRegex(ValueError, "chunked HTTP"):
                probe.stream_lines(response)
            read.assert_not_called()

    def test_compressed_response_rejected_before_read(self):
        response = requests.Response()
        response.headers["Transfer-Encoding"] = "chunked"
        response.headers["Content-Encoding"] = "gzip"
        with patch.object(response, "iter_lines") as read:
            with self.assertRaisesRegex(ValueError, "uncompressed"):
                probe.stream_lines(response)
            read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
