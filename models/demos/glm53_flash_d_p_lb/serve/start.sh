#!/bin/bash
# Start the GLM-5.3-Flash chat server on this box (detached; holds the device lock until stopped).
#   models/demos/glm53_flash_d_p_lb/serve/start.sh [ENV=val ...]     e.g. GLM_SERVE_MAX_CTX=32768
# Log: generated/glm_serve/server.log ("GLM_SERVE: listening" once ready). Stop: stop.sh
set -e
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
mkdir -p "$ROOT/generated/glm_serve"
LOG=${GLM_SERVE_LOG:-$ROOT/generated/glm_serve/server.log}
cd "$ROOT"
setsid nohup env BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 \
    PYTHONPATH="$ROOT" "$@" \
    bash -c 'source python_env/bin/activate && scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/glm53_flash_d_p_lb/serve/test_serve.py -s; echo "RUN_EXIT=$?"' \
    < /dev/null > "$LOG" 2>&1 &
echo "started (log $LOG); wait for 'GLM_SERVE: listening' in it"
