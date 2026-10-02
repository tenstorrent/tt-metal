#!/bin/bash
# Start the MiMo OpenAI-compatible server on this box (detached; holds the device lock until stopped).
#   models/demos/mimo_v2_d_p/serve/start.sh [ENV=val ...]      e.g. MIMO_SERVE_MAX_CTX=32768 MIMO_SERVE_CHUNK=1024
# Log: generated/mimo_serve/server.log ("MIMO_SERVE: listening" once ready, after ~5 min of model load). Stop: stop.sh
set -e
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
mkdir -p "$ROOT/generated/mimo_serve"
LOG=${MIMO_SERVE_LOG:-$ROOT/generated/mimo_serve/server.log}
cd "$ROOT"
setsid nohup env -u PYTHONPATH TT_METAL_HOME="$ROOT" PYTHONPATH="$ROOT/ttnn:$ROOT/tools:$ROOT" \
    MIMO_MESH=${MIMO_MESH:-2x4} "$@" \
    bash -c 'source python_env/bin/activate && scripts/run_safe_pytest.sh models/demos/mimo_v2_d_p/serve/test_serve.py::test_serve -s; echo "RUN_EXIT=$?"' \
    < /dev/null > "$LOG" 2>&1 &
echo "started (log $LOG); wait for 'MIMO_SERVE: listening' in it"
