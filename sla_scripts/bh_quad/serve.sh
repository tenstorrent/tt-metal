#!/bin/bash
# One Galaxy's Llama-3.3-70B vLLM server in the shipped SLA configuration (DP4 x TP8 prefill, 8 API servers,
# Gio plugin, decode after every prefill step), listening on 0.0.0.0:8000 so the balancer on another host can reach it.
# Runs in the foreground until SIGINT/SIGTERM (it forwards SIGINT to vLLM and waits: a hard kill hangs the chips).
# Env: J (work dir holding tt-metal/, weights/, cache/, vllm-tt-plugin-gio/, tmp/; default /mnt/nvme0/jerrywang),
#      OUT (log dir, default $J/runs/serve_<host>), MBT (step token budget, default 16384).
J=${J:-/mnt/nvme0/jerrywang}
export TT_METAL_DIR=$J/tt-metal HF_MODEL=${HF_MODEL:-$J/weights/Llama-3.3-70B-Instruct} TT_CACHE_PATH=${TT_CACHE_PATH:-$J/cache/llama70b}
export PLUGIN_DIR=$J/vllm-tt-plugin-gio PLUGIN_SRC=$J/vllm-tt-plugin-gio/src
export TMPDIR=$J/tmp TT_METAL_CACHE=${TT_METAL_CACHE:-$J/jit-cache} XDG_CACHE_HOME=$J/xdg-cache
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-2400} TT_PLUGIN_IDLE_GATHER_MS=${TT_PLUGIN_IDLE_GATHER_MS:-150}
export SERVE_CLI=1 TRACE_REGION=700000000 MAX_BATCHED_TOKENS=${MBT:-16384} LLAMA_BH_PREFILL_DP4=${DP4:-1}
export OUT=${OUT:-$J/runs/serve_$(hostname -s)}
cd "$TT_METAL_DIR" && source sla_scripts/bh_tp32/env.sh
export TT_CFG_EXTRA="$TT_CFG_EXTRA, \"flush_prefill_output\": true"
mkdir -p "$OUT" "$TMPDIR"
bash sla_scripts/bh_tp32/run_server.sh --api-server-count=8 --host 0.0.0.0 "$@" &
SRV=$!
trap 'pkill -INT -P $SRV; kill -INT $SRV' INT TERM
wait $SRV; wait $SRV
