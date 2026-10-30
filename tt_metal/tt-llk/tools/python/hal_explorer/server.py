#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Local page for the HAL pipeline.

    python3 tools/python/hal_explorer/server.py [--port 8765] [--host 127.0.0.1] [--jobs 4]

GET / serves page.html. POST /compile takes {"mode": "code" | "body", "body": str,
"variables": [{"type": str, "name": str}]} and returns the result JSON documented in
README.md. The server compiles whatever it receives: keep it on localhost.
"""

import argparse
import json
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Run as a script, the package's parent directory is not yet importable.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hal_explorer import pipeline  # noqa: E402  pylint: disable=wrong-import-position
from hal_explorer.model import (  # noqa: E402  pylint: disable=wrong-import-position
    to_json,
)
from hal_explorer.toolchain import (  # noqa: E402  pylint: disable=wrong-import-position
    build,
)

HERE = Path(__file__).resolve().parent
MAX_REQUEST_BYTES = 256 * 1024
MAX_BODY_CHARACTERS = 50_000
MAX_VARIABLES = 32
READ_TIMEOUT_SECONDS = 30


class Handler(BaseHTTPRequestHandler):
    timeout = READ_TIMEOUT_SECONDS

    def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        pass

    def send(self, status, content_type, payload):
        data = payload.encode("utf-8", errors="replace")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def send_error_json(self, status, message):
        self.send(
            status,
            "application/json",
            json.dumps({"ok": False, "diagnostics": message}),
        )

    def allowed_host(self):
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        return host in self.server.allowed_hosts

    def do_GET(self):
        if not self.allowed_host():
            self.send_error_json(403, "Unexpected Host header.")
            return
        if self.path not in ("/", "/index.html"):
            self.send(404, "text/plain", "not found")
            return
        examples = {
            path.stem: path.read_text(encoding="utf-8").split("\n\n", 1)[-1]
            for path in sorted((HERE / "examples").glob("*.cpp"))
        }
        prelude = {"code": build.INCLUDES, "body": build.INCLUDES + build.ALIASES}
        page = (
            (HERE / "ui" / "page.html")
            .read_text(encoding="utf-8")
            .replace("__PRELUDE__", json.dumps(prelude).replace("</", "<\\/"))
            .replace("__EXAMPLES__", json.dumps(examples).replace("</", "<\\/"))
        )
        self.send(200, "text/html; charset=utf-8", page)

    def do_POST(self):
        try:
            self.handle_compile()
        except Exception:  # pylint: disable=broad-except
            traceback.print_exc()
            self.send_error_json(
                500, "Internal error in the explorer; see the server log."
            )

    def handle_compile(self):
        if self.path != "/compile":
            self.send(404, "text/plain", "not found")
            return
        if not self.allowed_host():
            self.send_error_json(403, "Unexpected Host header.")
            return
        if (self.headers.get("Content-Type") or "").split(";")[
            0
        ].strip() != "application/json":
            self.send_error_json(415, "Send the request as application/json.")
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.send_error_json(411, "Content-Length is required.")
            return
        if length < 0 or length > MAX_REQUEST_BYTES:
            self.send_error_json(
                413, f"Request must be at most {MAX_REQUEST_BYTES} bytes."
            )
            return
        try:
            request = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            self.send_error_json(400, "Malformed JSON.")
            return
        problem = validate(request)
        if problem:
            self.send_error_json(400, problem)
            return
        with self.server.compile_slots:
            try:
                result = pipeline.run(
                    request.get("mode", "code"),
                    request["body"],
                    self.server.tables,
                    request.get("variables", []),
                )
                payload = json.dumps(to_json(result))
            except pipeline.InputError as error:
                payload = None
                message = str(error)
        if payload is None:
            self.send_error_json(400, message)
        else:
            self.send(200, "application/json", payload)


def validate(request):
    """A message describing what is wrong with the request, or None."""
    if not isinstance(request, dict):
        return "The request must be a JSON object."
    if request.get("mode", "code") not in pipeline.MODES:
        return f"mode must be one of {', '.join(pipeline.MODES)}."
    body = request.get("body")
    if not isinstance(body, str):
        return "body must be a string."
    if len(body) > MAX_BODY_CHARACTERS:
        return f"body must be at most {MAX_BODY_CHARACTERS} characters."
    variables = request.get("variables", [])
    if not isinstance(variables, list) or len(variables) > MAX_VARIABLES:
        return f"variables must be a list of at most {MAX_VARIABLES} entries."
    for variable in variables:
        if not isinstance(variable, dict) or not all(
            isinstance(variable.get(k, ""), str) for k in ("type", "name")
        ):
            return 'each variable must be {"type": string, "name": string}.'
    return None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--jobs", type=int, default=4, help="concurrent compilations (at least 1)"
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.tables = pipeline.load_tables()
    server.compile_slots = threading.BoundedSemaphore(args.jobs)
    server.allowed_hosts = {"localhost", "127.0.0.1", "::1", args.host}
    print(f"HAL explorer on http://{args.host}:{args.port}/ (tt-llk: {build.LLK_ROOT})")
    server.serve_forever()


if __name__ == "__main__":
    main()
