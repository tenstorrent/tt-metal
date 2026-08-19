# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Minimal OTLP/HTTP receiver that captures Claude Code cost telemetry.

Claude Code, when launched with CLAUDE_CODE_ENABLE_TELEMETRY=1 and an OTLP HTTP
exporter, POSTs its metrics to <endpoint>/v1/metrics. Among them is
`claude_code.cost.usage` (USD), which the CLI derives from actual API usage — the
same source /usage reads, and it includes the subagent + auxiliary (background)
costs that the transcript JSONLs do not. This server listens for those exports and
appends each cost datapoint to a sink JSONL so session_cost.py can use the
authoritative figure instead of the token estimate.

It is a plain stdlib http.server — no OpenTelemetry libraries needed — parsing the
OTLP/JSON encoding (set OTEL_EXPORTER_OTLP_PROTOCOL=http/json on the producer).
Fail-soft everywhere: a receiver hiccup must never disturb the codegen run.

Usage:
    python codegen/scripts/otel_cost_receiver.py --sink <path> [--port 4318]
    # then launch claude with:
    #   CLAUDE_CODE_ENABLE_TELEMETRY=1
    #   OTEL_METRICS_EXPORTER=otlp
    #   OTEL_EXPORTER_OTLP_PROTOCOL=http/json
    #   OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:<port>

Each sink line: {"ts","session_id","query_source","model","cost_usd"}.
Cost temporality is delta, so a session's total is the SUM of its datapoints.
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
        """Append rows under the writer lock; trim to max_lines when it grows past it.

        Trimming keeps the NEWEST lines. A cap large enough to hold many runs (default
        200k) means a single run's datapoints are never trimmed before session_cost
        reads them — only long-past runs' lines are dropped, and those were already
        recorded into their run.json.
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
