#!/bin/bash
# Start the Xing serve stack (tt-d-gen engine -> prefill runner -> LM head; serve/README.md), detached. It holds the
# device lock until stopped.
#   models/demos/xing40_a4b_d_p/serve/start.sh [ENV=val ...]      e.g. XING_SERVE_SLOTS=2 XING_SERVE_PORT=8001
# Log: generated/xing_serve/server.log ("XING_SERVE: listening" once ready). Stop: stop.sh
set -e
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
DIR=${XING_SERVE_DIR:-$ROOT/generated/xing_serve}
mkdir -p "$DIR"
LOG=$DIR/server.log
cd "$ROOT"
setsid nohup env -u PYTHONPATH TT_METAL_HOME="$ROOT" PYTHONPATH="$ROOT" XING_SERVE_DIR="$DIR" \
    BRINGUP_SPEC="${BRINGUP_SPEC:-$ROOT/models/demos/xing40_a4b_d_p/bringup/spec.yaml}" \
    BRINGUP_SERVER_REPO="${BRINGUP_SERVER_REPO:-/localdev/$USER/tt-d-gen}" \
    ${BRINGUP_HF:+BRINGUP_HF="$BRINGUP_HF"} "$@" \
    bash -c 'source python_env/bin/activate && scripts/run_safe_pytest.sh --run-all -s models/demos/xing40_a4b_d_p/serve/test_serve.py::test_serve; echo "RUN_EXIT=$?"' \
    < /dev/null > "$LOG" 2>&1 &
echo "started (log $LOG); wait for 'XING_SERVE: listening' in it (model load + compile: ~5-10 min)"
