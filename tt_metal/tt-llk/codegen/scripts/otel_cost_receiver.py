# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tiny local server that records Claude Code's real cost.

With telemetry enabled, Claude Code sends its usage metrics to a local endpoint.
One of them, `claude_code.cost.usage`, is the cost in USD that Claude Code itself
computed — the same number `/usage` shows — and it includes the subagent and
background spend that the transcript files leave out. This server catches those
sends and appends each cost datapoint to a sink file; session_cost.py then reads
the sink and records that real cost instead of estimating it from token counts.

Plain Python stdlib, no OpenTelemetry package. It reads the OTLP/JSON body, so the
producer must set OTEL_EXPORTER_OTLP_PROTOCOL=http/json. Fail-soft: any problem
here is swallowed so it can never disrupt a codegen run.

Each sink line is {"ts", "session_id", "query_source", "model", "cost_usd"}. Cost
is a delta metric, so a session's total is the sum of its lines.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

COST_METRIC = "claude_code.cost.usage"

# One writer lock: ThreadingHTTPServer handles requests concurrently, so appends
# and the occasional trim must not interleave.
_WRITE_LOCK = threading.Lock()


def _attr_map(attrs: list) -> dict:
    """Flatten an OTLP attribute list [{key,value:{stringValue|intValue|...}}] to a dict."""
    out: dict = {}
    for a in attrs or []:
        try:
            k = a.get("key")
            v = a.get("value") or {}
            out[k] = (
                v.get("stringValue")
                if "stringValue" in v
                else (
                    v.get("intValue")
                    if "intValue" in v
                    else (
                        v.get("doubleValue")
                        if "doubleValue" in v
                        else v.get("boolValue")
                    )
                )
            )
        except Exception:
            continue
    return out


def _cost_rows(body: dict) -> list[dict]:
    """Pull every claude_code.cost.usage datapoint out of an OTLP/JSON metrics export."""
    rows: list[dict] = []
    for rm in body.get("resourceMetrics", []) or []:
        for sm in rm.get("scopeMetrics", []) or []:
            for m in sm.get("metrics", []) or []:
                if m.get("name") != COST_METRIC:
                    continue
                # cost is a Sum; be tolerant of gauge shape too.
                dps = (m.get("sum") or m.get("gauge") or {}).get("dataPoints", [])
                for dp in dps or []:
                    val = dp.get("asDouble")
                    if val is None:
                        val = dp.get("asInt")
                    if val is None:
                        continue
                    attrs = _attr_map(dp.get("attributes"))
                    rows.append(
                        {
                            "ts": dp.get("timeUnixNano"),
                            "session_id": attrs.get("session.id"),
                            "query_source": attrs.get("query_source"),
                            "model": attrs.get("model"),
                            "cost_usd": float(val),
                        }
                    )
    return rows


class _Handler(BaseHTTPRequestHandler):
    sink_path = ""  # set on the class before serving
    max_lines = 0  # >0 caps the sink; oldest lines are trimmed
    _line_count = 0  # running total, seeded from the existing file at startup

    def _append(self, rows: list) -> None:
        """Add new cost lines to the sink, and keep the file from growing forever.

        Writes under a lock, because the server is multi-threaded and two requests
        must not interleave their lines. Once the file passes max_lines it is trimmed
        down to the newest max_lines. That only drops old, long-finished runs (whose
        cost was already saved to their run.json); the cap is large enough (default
        200k) that a run still in progress is never trimmed before session_cost reads
        its lines.
        """
        with _WRITE_LOCK:
            with open(self.sink_path, "a") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            _Handler._line_count += len(rows)
            # Hysteresis: trim only after exceeding the cap by 10%, so we rewrite
            # rarely (bulk) rather than on every append.
            if self.max_lines and _Handler._line_count > int(self.max_lines * 1.1):
                self._trim_locked()

    def _trim_locked(self) -> None:
        try:
            with open(self.sink_path) as fh:
                lines = fh.readlines()
            keep = lines[-self.max_lines :]
            tmp = self.sink_path + ".tmp"
            with open(tmp, "w") as fh:
                fh.writelines(keep)
            os.replace(tmp, self.sink_path)
            _Handler._line_count = len(keep)
        except Exception:
            pass  # trimming is best-effort; a failure just leaves the file larger

    def _ok(self) -> None:
        # OTLP success is an empty ExportServiceResponse.
        payload = b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            if (self.headers.get("Content-Encoding") or "").lower() == "gzip":
                raw = gzip.decompress(raw)
            # Only metrics carry cost; traces/logs endpoints are acked and ignored.
            if self.path.endswith("/v1/metrics") and raw:
                rows = _cost_rows(json.loads(raw))
                if rows:
                    self._append(rows)
        except Exception:
            pass  # never fail the exporter; the next export retries the delta
        self._ok()

    def do_GET(self) -> None:  # noqa: N802 — a health probe
        self._ok()

    def log_message(self, *args) -> None:  # silence per-request stderr logging
        return


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sink",
        default=os.environ.get("CODEGEN_OTEL_SINK"),
        help="JSONL file to append cost datapoints to (or $CODEGEN_OTEL_SINK).",
    )
    ap.add_argument(
        "--port", type=int, default=4318, help="OTLP/HTTP port (default 4318)."
    )
    ap.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)."
    )
    ap.add_argument(
        "--max-lines",
        type=int,
        default=200_000,
        help="Cap the sink at this many lines (0 = unlimited). Oldest lines are "
        "trimmed once exceeded; the default holds many runs, so no in-flight run is "
        "ever trimmed before session_cost reads it.",
    )
    args = ap.parse_args(argv)
    if not args.sink:
        print(
            "otel_cost_receiver: --sink (or $CODEGEN_OTEL_SINK) is required",
            file=sys.stderr,
        )
        return 2
    os.makedirs(os.path.dirname(os.path.abspath(args.sink)) or ".", exist_ok=True)

    _Handler.sink_path = args.sink
    _Handler.max_lines = max(0, args.max_lines)
    try:
        with open(args.sink) as fh:
            _Handler._line_count = sum(1 for _ in fh)
    except OSError:
        _Handler._line_count = 0
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(
        f"otel_cost_receiver: listening on http://{args.host}:{args.port} -> {args.sink}",
        file=sys.stderr,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
