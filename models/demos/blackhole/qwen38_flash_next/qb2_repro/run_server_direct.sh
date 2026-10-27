#!/usr/bin/env bash
# Lanes-capable wrapper. Reproduces tools/run_qwen38_chat_server.sh --profile qb2 --allocated-context 32768 (no --sampling:
# greedy-only server), which has no --lanes flag. Extra args go to the server: e.g. --host 127.0.0.1 --port 8000 --mtp 4 --lanes 4
# Env: HF_MODEL, TT_CACHE_PATH (required); QWEN38_PY; TT_METAL_HOME.
set -euo pipefail
SELF="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="${TT_METAL_HOME:-$(cd -- "$SELF/../../../../.." && pwd -P)}"
HERE="$REPO_ROOT/models/demos/blackhole/qwen38_flash_next/tools"
PY="${QWEN38_PY:-$REPO_ROOT/python_env_q38fn/bin/python}"
CKPT="${HF_MODEL:?set HF_MODEL to the flat checkpoint directory}"
cache_root="${TT_CACHE_PATH:?set TT_CACHE_PATH}"
hp=tt-quietbox-2; ctx=32768
caches="$cache_root/caches/c$ctx-$hp"
run_dir="$cache_root/runs/q38-chat-server-$hp-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$caches/components" "$cache_root/caches/bf4-experts" "$caches/model-io" "$cache_root/jit-cache" "$run_dir/tmp"
cd "$REPO_ROOT"
export TT_METAL_HOME="$REPO_ROOT" TT_VISIBLE_DEVICES=0,1,2,3
export TT_MESH_GRAPH_DESC_PATH="$HERE/qb2_p300_1x4_line_mesh_graph_descriptor.textproto"
export QWEN38_HARDWARE_MODE=diagnostic_non_promoting TT_METAL_TRACE_ALLOC_TRACKING=1
export TT_METAL_CACHE="$cache_root/jit-cache" TT_METAL_LOGS_PATH="$run_dir/metal-logs" TMPDIR="$run_dir/tmp"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
args=(--hardware-profile $hp --checkpoint "$CKPT" --component-cache-root "$caches/components"
  --routed-bf4-scratch-root "$cache_root/caches/bf4-experts" --model-io-cache-root "$caches/model-io"
  --phase-log "$run_dir/phase-markers.jsonl" --evidence "$run_dir" --allocated-context $ctx --prefill-mode chunked
  "$@" --stall-seconds 300)
echo "RUN_DIR=$run_dir"
echo "CMD: $PY $HERE/qwen38_chat_server.py ${args[*]}"
exec "$PY" "$HERE/qwen38_chat_server.py" "${args[@]}"
