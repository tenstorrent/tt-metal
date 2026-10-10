#!/bin/bash
# Stop the server started by start.sh: SIGTERM -> it finishes the current request and closes the mesh; run_safe_pytest
# then releases the device lock.
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
PID_FILE=${GLM_SERVE_PID:-$ROOT/generated/glm_serve/server.pid}
[ -f "$PID_FILE" ] || { echo "no pid file $PID_FILE (not running?)"; exit 1; }
kill -TERM "$(cat "$PID_FILE")" && echo "sent SIGTERM to $(cat "$PID_FILE")"
