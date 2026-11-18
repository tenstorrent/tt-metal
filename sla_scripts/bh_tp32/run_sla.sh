#!/bin/bash
# Full SLA run: start the server, run bench.sh, print the table, stop the server with SIGINT and wait for it to exit.
# Env as for run_server.sh (source env.sh first). Usage: bash run_sla.sh [extra vLLM args]
: "${TT_METAL_DIR:?}" "${HF_MODEL:?}" "${TT_CACHE_PATH:?}" "${PLUGIN_DIR:?}" "${OUT:?}"
D=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
bash "$D/run_server.sh" "$@" &
export SERVER_PID=$!
bash "$D/bench.sh"
(cd "$TT_METAL_DIR" && source python_env/bin/activate && python "$D/summary.py" "$OUT" | tee "$OUT/summary.txt")
kill -INT "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID"
