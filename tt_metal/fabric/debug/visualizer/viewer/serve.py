# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Serve the static viewer on loopback with caching disabled."""

from __future__ import annotations

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


class ViewerHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Serve the offline TT Fabric Debug Viewer.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"listen address (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"listen port (default: {DEFAULT_PORT})")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    viewer_root = Path(__file__).resolve().parent
    handler = functools.partial(ViewerHandler, directory=str(viewer_root))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"TT Fabric Debug Viewer: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
