# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pipeline stage monitor: per-process HTTP server and standalone dashboard viewer.

Each pipeline process runs a StageMonitor that serves its StageInfo as JSON
on ``base_port + mesh_id``.  The standalone ``main()`` polls all stage endpoints
and renders a live terminal table.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from loguru import logger

from models.demos.deepseek_v3_b1.demo.stage import StageInfo

DEFAULT_BASE_PORT = 8400


class StageMonitor:
    """Lightweight HTTP server that exposes a single stage's StageInfo as JSON."""

    def __init__(self, stage_info: StageInfo, *, base_port: int = DEFAULT_BASE_PORT) -> None:
        self._info = stage_info
        self._port = base_port + stage_info.mesh_id
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._port

    def start(self) -> None:
        info = self._info

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                payload = json.dumps(info.to_dict()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

        self._server = HTTPServer(("0.0.0.0", self._port), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="stage-monitor")
        self._thread.start()
        logger.info("Stage monitor listening on port {}", self._port)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None
            self._thread = None


# ---------------------------------------------------------------------------
# Standalone dashboard
# ---------------------------------------------------------------------------

_PHASE_COLORS = {
    "init": "\033[90m",
    "configuring_block": "\033[33m",
    "block_configured": "\033[33m",
    "setting_up": "\033[33m",
    "ready": "\033[36m",
    "pipeline_starting": "\033[34m",
    "pipeline_running": "\033[34m",
    "computing": "\033[32m",
    "prefilling": "\033[35m",
    "decoding": "\033[32m",
    "error": "\033[31m",
    "terminated": "\033[90m",
}
_RESET = "\033[0m"


def _fetch_stage(host: str, port: int, timeout: float = 1.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/", timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _render_dashboard(rows: list[dict[str, Any]], num_stages: int) -> str:
    online = sum(1 for r in rows if r.get("phase") != "unreachable")
    lines = [
        f"\033[2J\033[H",
        f"Pipeline Monitor  ({online}/{num_stages} stages online)  [Ctrl+C to quit]",
        "",
        f"{'ID':>4}  {'Type':<24} {'Phase':<20} {'Iter':>8}  {'In Phase':>9}  {'Uptime':>8}  {'Error'}",
        "-" * 95,
    ]
    for r in rows:
        phase = r.get("phase", "unreachable")
        color = _PHASE_COLORS.get(phase, "")
        err = r.get("error") or ""
        iter_str = f"{r.get('iteration', '-')}" if phase != "unreachable" else "-"
        in_phase = f"{r.get('seconds_in_phase', 0):.1f}s" if phase != "unreachable" else "-"
        uptime = f"{r.get('uptime_s', 0):.0f}s" if phase != "unreachable" else "-"
        stage_type = r.get("stage_type", "?")
        lines.append(
            f"{r['mesh_id']:>4}  {stage_type:<24} {color}{phase:<20}{_RESET} {iter_str:>8}  {in_phase:>9}  {uptime:>8}  {err}"
        )
    return "\n".join(lines)


def run_dashboard(
    num_stages: int,
    *,
    host: str = "localhost",
    base_port: int = DEFAULT_BASE_PORT,
    refresh_interval: float = 1.0,
) -> None:
    """Poll all stage endpoints and render a live table until interrupted."""
    try:
        while True:
            rows: list[dict[str, Any]] = []
            for i in range(num_stages):
                data = _fetch_stage(host, base_port + i)
                if data is not None:
                    rows.append(data)
                else:
                    rows.append({"mesh_id": i, "phase": "unreachable", "stage_type": "?"})
            print(_render_dashboard(rows, num_stages), flush=True)
            time.sleep(refresh_interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Connect to a running pipeline and display stage status")
    parser.add_argument("num_stages", type=int, help="Number of pipeline stages (4, 16, or 64)")
    parser.add_argument("--host", default="localhost", help="Hostname where pipeline processes run")
    parser.add_argument(
        "--base-port", type=int, default=DEFAULT_BASE_PORT, help="Base port (stage N listens on base+N)"
    )
    parser.add_argument("--refresh", type=float, default=1.0, help="Refresh interval in seconds")
    args = parser.parse_args(argv)
    run_dashboard(args.num_stages, host=args.host, base_port=args.base_port, refresh_interval=args.refresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
