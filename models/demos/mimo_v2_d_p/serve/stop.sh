#!/bin/bash
# Stop the server started by start.sh: SIGTERM -> it finishes the current request, closes the mesh; run_safe_pytest
# then releases the device lock (and resets the device).
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
PID_FILE=${MIMO_SERVE_PID:-$ROOT/generated/mimo_serve/server.pid}
[ -f "$PID_FILE" ] || { echo "no pid file $PID_FILE (not running?)"; exit 1; }
kill -TERM "$(cat "$PID_FILE")" && echo "sent SIGTERM to $(cat "$PID_FILE")"
