# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Synchronous HTTP service for the Gemma 4 long-context prefill demo.

The service owns eight logical device-resident KV cache slots by default. They share
one paged-KV block pool; requests run serially and replace slots round-robin.

Example:
    python models/demos/gemma4/demo/prefill_service.py \
        --mesh 8x4 --chunk-size 8192 --max-context-len 262144
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
import uuid
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Protocol

from loguru import logger

DEFAULT_MODEL = "google/gemma-4-31B-it"
DEFAULT_MAX_REQUEST_BYTES = 32 * 1024 * 1024


class PrefillRuntime(Protocol):
    def prefill(self, prompt: str, request_id: str) -> dict:
        ...

    def info(self) -> dict:
        ...


class PrefillHTTPServer(HTTPServer):
    allow_reuse_address = True

    def __init__(self, server_address, runtime: PrefillRuntime, max_request_bytes: int):
        super().__init__(server_address, PrefillRequestHandler)
        self.runtime = runtime
        self.max_request_bytes = max_request_bytes


class PrefillRequestHandler(BaseHTTPRequestHandler):
    server: PrefillHTTPServer
    protocol_version = "HTTP/1.1"

    def _write_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str, request_id: str | None = None) -> None:
        payload = {"status": "error", "error": message}
        if request_id is not None:
            payload["request_id"] = request_id
        self._write_json(status, payload)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._error(HTTPStatus.NOT_FOUND, "unknown endpoint")
            return
        self._write_json(HTTPStatus.OK, self.server.runtime.info())

    def do_POST(self) -> None:
        if self.path != "/prefill":
            self._error(HTTPStatus.NOT_FOUND, "unknown endpoint")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
            return
        if content_length <= 0:
            self._error(HTTPStatus.BAD_REQUEST, "request body is required")
            return
        if content_length > self.server.max_request_bytes:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body is too large")
            return

        try:
            payload = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._error(HTTPStatus.BAD_REQUEST, "request body must be valid UTF-8 JSON")
            return
        if not isinstance(payload, dict):
            self._error(HTTPStatus.BAD_REQUEST, "request body must be a JSON object")
            return

        request_id = payload.get("request_id") or str(uuid.uuid4())
        prompt = payload.get("prompt")
        if not isinstance(request_id, str) or not request_id:
            self._error(HTTPStatus.BAD_REQUEST, "request_id must be a non-empty string")
            return
        if not isinstance(prompt, str) or not prompt.strip():
            self._error(HTTPStatus.BAD_REQUEST, "prompt must be a non-empty string", request_id)
            return

        try:
            result = self.server.runtime.prefill(prompt, request_id)
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc), request_id)
            return
        except Exception as exc:
            logger.error("Prefill request {} failed:\n{}", request_id, traceback.format_exc())
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc), request_id)
            return
        self._write_json(HTTPStatus.OK, result)

    def log_message(self, format_string: str, *args) -> None:
        logger.info("{} - {}", self.address_string(), format_string % args)


def create_server(
    runtime: PrefillRuntime,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
) -> PrefillHTTPServer:
    return PrefillHTTPServer((host, port), runtime, max_request_bytes)


def _parse_mesh_shape(value: str) -> tuple[int, int]:
    try:
        rows, cols = (int(part) for part in value.lower().split("x", maxsplit=1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("mesh must be formatted as ROWSxCOLS, for example 8x4") from exc
    if rows <= 0 or cols <= 0:
        raise argparse.ArgumentTypeError("mesh dimensions must be positive")
    return rows, cols


@contextmanager
def _open_mesh_device(mesh_shape: tuple[int, int], trace_region_size: int):
    import ttnn

    rows, cols = mesh_shape
    fabric_config = None
    if rows * cols > 1:
        fabric_config = ttnn.FabricConfig.FABRIC_2D if rows > 1 and cols > 1 else ttnn.FabricConfig.FABRIC_1D
        router_config = ttnn.FabricRouterConfig()
        router_config.max_packet_payload_size_bytes = 8192
        ttnn.set_fabric_config(
            fabric_config,
            ttnn.FabricReliabilityMode.RELAXED_INIT,
            None,
            ttnn.FabricTensixConfig.DISABLED,
            ttnn.FabricUDMMode.DISABLED,
            ttnn.FabricManagerMode.DEFAULT,
            router_config,
        )

    try:
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(rows, cols),
            trace_region_size=trace_region_size,
        )
    except Exception:
        if fabric_config is not None:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        raise

    try:
        yield mesh_device
    finally:
        for submesh in mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh_device)
        if fabric_config is not None:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="address to bind (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8080, help="port to bind (default: %(default)s)")
    parser.add_argument("--mesh", type=_parse_mesh_shape, default=(8, 4), help="device mesh (default: 8x4)")
    parser.add_argument(
        "--model",
        default=None,
        help=f"Hugging Face model path (default: HF_MODEL/GEMMA4_MODEL_PATH/{DEFAULT_MODEL})",
    )
    parser.add_argument("--chunk-size", type=int, default=8192, help="prefill trace chunk size")
    parser.add_argument("--max-context-len", type=int, default=262144, help="maximum prompt token count")
    parser.add_argument("--cache-slots", type=int, default=8, help="number of round-robin KV cache slots")
    parser.add_argument(
        "--trace-region-size",
        type=int,
        default=int(os.environ.get("GEMMA4_PREFILL_TRACE_REGION_SIZE", "256000000")),
        help="device trace-region size in bytes",
    )
    parser.add_argument(
        "--max-request-bytes",
        type=int,
        default=DEFAULT_MAX_REQUEST_BYTES,
        help="maximum JSON request-body size",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _create_parser().parse_args(argv)
    from models.demos.gemma4.demo.prefill_runtime import TracedPrefillRuntime

    model_path = args.model or os.getenv("HF_MODEL") or os.getenv("GEMMA4_MODEL_PATH", DEFAULT_MODEL)
    with _open_mesh_device(args.mesh, args.trace_region_size) as mesh_device:
        runtime = TracedPrefillRuntime(
            mesh_device,
            model_path=model_path,
            chunk_size=args.chunk_size,
            max_context_len=args.max_context_len,
            cache_slots=args.cache_slots,
        )
        server = create_server(
            runtime,
            host=args.host,
            port=args.port,
            max_request_bytes=args.max_request_bytes,
        )
        logger.info("Gemma 4 prefill service ready at http://{}:{}", args.host, args.port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Stopping Gemma 4 prefill service")
        finally:
            server.server_close()
            runtime.close()


if __name__ == "__main__":
    main()
