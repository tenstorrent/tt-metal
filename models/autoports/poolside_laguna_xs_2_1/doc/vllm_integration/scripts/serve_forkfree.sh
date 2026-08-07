#!/bin/bash
# Serve Laguna-XS-2.1 on the FORK-FREE stack (stock vLLM 0.24.0 + public vllm-tt-plugin + tt-metal vllm_ext).
# Backgrounds the server (setsid) and streams the FULL raw server log to /home/ttuser/laguna_serve.log.
#   Launch:  serve_forkfree.sh
#   Watch:   tail -f /home/ttuser/laguna_serve.log      (ready at "Application startup complete", ~10 min)
#   Stop:    serve_forkfree.sh stop                     (TERM/KILL + tt-smi -r all)
set +e

# Point FF at your fork-free env (see serve_forkfree.md Section 1). Override by exporting FF before running.
FF="${FF:-/home/ttuser/.venv_laguna_forkfree/bin}"
LOG=/home/ttuser/laguna_serve.log
PIDF=/tmp/laguna_forkfree_srv.pid

stop() {
  g=$(cat "$PIDF" 2>/dev/null)
  [ -n "$g" ] && kill -TERM -"$g" 2>/dev/null
  sleep 10; [ -n "$g" ] && kill -KILL -"$g" 2>/dev/null
  pkill -9 -f "vllm serve poolside" 2>/dev/null; sleep 3
  /home/ttuser/.tenstorrent-venv/bin/tt-smi -r all >/dev/null 2>&1
  echo "stopped + mesh reset"
}
[ "$1" = "stop" ] && { stop; exit 0; }

if [ ! -x "$FF/vllm" ]; then echo "ERROR: no vllm at $FF (build the fork-free env first; see serve_forkfree.md §1)"; exit 1; fi

export TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal
export PYTHONPATH=/home/ttuser/dev/tt-metal
export EXTRA_MODELS_DIR=/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1/vllm_ext/extra_models
export MESH_DEVICE=P150x4 HF_MODEL=poolside/Laguna-XS-2.1
export TT_LAGUNA_PIPE_CHUNK=2048 TT_LAGUNA_PREFIX_CACHE=1 TT_LAGUNA_PREFILL_FAST=1 TT_LAGUNA_HYBRID_KV=0

: > "$LOG"
echo "[serve_forkfree] vllm $($FF/python -c 'import vllm;print(vllm.__version__)' 2>/dev/null) | log: $LOG" | tee -a "$LOG"
cd /tmp
setsid "$FF/vllm" serve poolside/Laguna-XS-2.1 \
  --trust-remote-code --max-model-len 131072 --max-num-seqs 8 --block-size 64 \
  --additional-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 1500000000, "fabric_config": "FABRIC_1D_RING"}}' \
  --enable-prefix-caching --enable-auto-tool-choice \
  --tool-call-parser poolside_v1 --reasoning-parser poolside_v1 --port 8000 >> "$LOG" 2>&1 &
echo $! > "$PIDF"
echo "[serve_forkfree] booting (pid $(cat "$PIDF")). Ready at 'Application startup complete' (~10 min)."
echo "  tail -f $LOG"
