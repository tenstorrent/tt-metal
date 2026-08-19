#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Launch claude with cost telemetry captured locally, so the codegen dashboard can
# show the CLI's own claude_code.cost.usage (authoritative — includes subagent and
# auxiliary/background spend the transcripts omit) instead of the token estimate.
# Works for INTERACTIVE runs, which have no cli_output.json.
#
# Usage:
#   codegen/scripts/with_telemetry.sh            # then: > Generate gelu for Quasar
#   codegen/scripts/with_telemetry.sh -p "…"     # headless also fine
#
# It starts otel_cost_receiver.py on a local port, points claude's OTLP exporter at
# it, and exports CODEGEN_OTEL_SINK so session_cost.py (run by refresh_cost.sh inside
# the session) reads the sink with no further wiring. The receiver is stopped on exit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${CODEGEN_OTEL_PORT:-4318}"
export CODEGEN_OTEL_SINK="${CODEGEN_OTEL_SINK:-/proj_sw/user_dev/llk_code_gen/otel_cost.jsonl}"
# Test seam: override the launched binary (default: claude).
CLAUDE_BIN="${CODEGEN_CLAUDE_BIN:-claude}"

python "$SCRIPT_DIR/otel_cost_receiver.py" --sink "$CODEGEN_OTEL_SINK" --port "$PORT" &
RECV_PID=$!
trap 'kill "$RECV_PID" 2>/dev/null || true' EXIT

export CLAUDE_CODE_ENABLE_TELEMETRY=1
export OTEL_METRICS_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_PROTOCOL=http/json
export OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:${PORT}"
# Delta temporality: each cost datapoint is an increment, so a session's total is
# the sum of its datapoints (what _otel_cost in session_cost.py computes).
export OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE="${OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE:-delta}"
# Short export interval (default ~60s) so a run's cost reaches the sink within
# seconds — the finalize snapshot then captures the whole run, not all-but-the-last-minute.
export OTEL_METRIC_EXPORT_INTERVAL="${OTEL_METRIC_EXPORT_INTERVAL:-5000}"

"$CLAUDE_BIN" "$@"
