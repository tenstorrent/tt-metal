#!/bin/bash
# Serve Laguna-XS-2.1 on vLLM: stock vLLM 0.24.0 + public vllm-tt-plugin + this model's vllm_ext.
# Builds the env on first use (setup_vllm.sh), then backgrounds the server (setsid) and streams
# the FULL raw server log to /home/ttuser/laguna_serve.log.
#   Launch:  ./serve_vllm.sh
#   Watch:   tail -f /home/ttuser/laguna_serve.log   (ready at "Application startup complete", ~10 min)
#   Stop:    ./serve_vllm.sh stop                    (TERM/KILL + tt-smi -r all)
set +e

MODEL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$MODEL_DIR/../../.." && pwd)
VLLM_ENV="${VLLM_ENV:-$MODEL_DIR/.venv}"
VLLM_ENV_BIN="${VLLM_ENV_BIN:-$VLLM_ENV/bin}"
LOG=/home/ttuser/laguna_serve.log
PIDF=/tmp/laguna_vllm_srv.pid

stop() {
  g=$(cat "$PIDF" 2>/dev/null)
  [ -n "$g" ] && kill -TERM -"$g" 2>/dev/null
  sleep 10; [ -n "$g" ] && kill -KILL -"$g" 2>/dev/null
  pkill -9 -f "vllm serve poolside" 2>/dev/null; sleep 3
  # Hard-killing a FABRIC_1D_RING server dirties eth cores — reset before reopening the mesh.
  TT_SMI=$(command -v tt-smi 2>/dev/null || echo /home/ttuser/.tenstorrent-venv/bin/tt-smi)
  if [ -x "$TT_SMI" ]; then
    "$TT_SMI" -r all >/dev/null 2>&1
    echo "stopped + mesh reset"
  else
    echo "stopped — WARNING: no tt-smi found, mesh NOT reset (run 'tt-smi -r all' before rebooting)"
  fi
}
[ "$1" = "stop" ] && { stop; exit 0; }

# One command from a fresh clone: build the env if it isn't there yet.
if [ ! -x "$VLLM_ENV_BIN/vllm" ]; then
  echo "[serve_vllm] no env at $VLLM_ENV — running setup_vllm.sh first (~30-45 min)"
  VLLM_ENV="$VLLM_ENV" "$MODEL_DIR/setup_vllm.sh" || { echo "[serve_vllm] setup failed"; exit 1; }
fi

# ttnn is a wheel here: it self-locates its runtime root, so TT_METAL_HOME must stay UNSET —
# pointing it at some other tt-metal tree would mix in a different version's kernels.
export PYTHONPATH="$REPO_ROOT"          # so EXTRA_MODELS_DIR's main_class (generator_vllm) resolves
export EXTRA_MODELS_DIR="$MODEL_DIR/vllm_ext/extra_models"
export MESH_DEVICE=P150x4 HF_MODEL=poolside/Laguna-XS-2.1
export TT_LAGUNA_PIPE_CHUNK=2048 TT_LAGUNA_PREFIX_CACHE=1 TT_LAGUNA_PREFILL_FAST=1 TT_LAGUNA_HYBRID_KV=0

: > "$LOG"
echo "[serve_vllm] vllm $("$VLLM_ENV_BIN/python" -c 'import vllm;print(vllm.__version__)' 2>/dev/null) | env: $VLLM_ENV | log: $LOG" | tee -a "$LOG"
cd /tmp
setsid "$VLLM_ENV_BIN/vllm" serve poolside/Laguna-XS-2.1 \
  --trust-remote-code --max-model-len 131072 --max-num-seqs 8 --block-size 64 \
  --additional-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 1500000000, "fabric_config": "FABRIC_1D_RING"}}' \
  --enable-prefix-caching --enable-auto-tool-choice \
  --tool-call-parser poolside_v1 --reasoning-parser poolside_v1 --port 8000 >> "$LOG" 2>&1 &
echo $! > "$PIDF"
echo "[serve_vllm] booting (pid $(cat "$PIDF")). Ready at 'Application startup complete' (~10 min)."
echo "  tail -f $LOG"
