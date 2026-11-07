#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# MiMo-V2.6-Flash multimodal row engines for evals on a shared Galaxy: one vLLM engine per requested row, each started
# under its own per-row device lock (DEVLOCK_ROW <row> <timeout>, which exports the row's TT_VISIBLE_DEVICES), behind
# lb.py with its request shim on a fixed port. Engine settings are those of the last validated multimodal load run
# (the MTP release config with the multimodal class): TT_MODEL_CLASS_OVERRIDES, MTP drafting (8 slots, adaptive
# groups), 1.2 GB trace region, device sampling for greedy decode, 1M context, 8 sequences per engine.
#
# Usage: serve_mm_eval.sh <log dir> [rows, default "2 3"]
#   Stop with SIGTERM / Ctrl-C: each engine's process group gets SIGTERM and STOP_TIMEOUT (120) seconds before
#   SIGKILL, then its row lock is released when the lock wrapper sees it exit. When an engine or the proxy exits on its
#   own, the others are stopped and the script exits with that status. Readiness: "ready" in <log dir>/status once every
#   engine logged "Application startup complete".
# env (defaults are bhg4's):
#   R                 root (/data/ttuser-agentx); DEVLOCK_ROW ($R/mm/devlock_row.sh); ROW_TIMEOUT seconds (14400)
#   MODEL_SRC         tt-metal checkout under test (default: this file's repository); TT_METAL_HOME ($R/tt-metal)
#   PLUGIN_SRC        vllm-tt-plugin src/ ($R/mm/vllm-tt-plugin-mm/src); EXTRA_PYTHONPATH (torchaudio: $R/mm/llm/pylib)
#   MIMO_TT_CACHE     the LLM's TT cache (the MTP entry's tt-inference-server volume); MIMO_VISION_CACHE
#   BASE_PORT         proxy port (8730); engine of row r on BASE_PORT + 1 + r
#   EXTRA_VLLM_ARGS_LINES  extra engine arguments, one per line; GEN_CONFIG  generation-config override JSON
#   TT_PROXY_*        lb.py's shim (TT_PROXY_FETCH_MEDIA defaults to 1 here: the proxy downloads http(s) media and
#                     the engines refuse media URLs)
set -u
LOGS=$(realpath -m "${1:?log dir}"); mkdir -p "$LOGS"
ROWS=${2:-2 3}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=${R:-/data/ttuser-agentx}
DEVLOCK_ROW=${DEVLOCK_ROW:-$R/mm/devlock_row.sh}
ROW_TIMEOUT=${ROW_TIMEOUT:-14400}
export TT_METAL_HOME=${TT_METAL_HOME:-$R/tt-metal}
MODEL_SRC=${MODEL_SRC:-$(cd "$HERE/../../../.." && pwd)}
PLUGIN_SRC=${PLUGIN_SRC:-$R/mm/vllm-tt-plugin-mm/src}
MODEL=${MODEL:-$R/hf/MiMo-V2.6-Flash-MOPD}
VLLM=${VLLM:-$R/tt-metal/python_env/bin/vllm}
PYTHON=${PYTHON:-$(dirname "$VLLM")/python}
BASE_PORT=${BASE_PORT:-8730}

export PYTHONPATH=$TT_METAL_HOME/ttnn:$MODEL_SRC:$PLUGIN_SRC:${EXTRA_PYTHONPATH:-$R/mm/llm/pylib}
export VLLM_TT_PLUGIN_ROOT=$(dirname "$PLUGIN_SRC") MESH_DEVICE="(1,8)" VLLM_PLUGINS=tt,tt_model_registry
export HF_HUB_OFFLINE=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ENGINE_READY_TIMEOUT_S=3600 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MIMO_MODEL_PATH=$MODEL MIMO_VISION_CACHE=${MIMO_VISION_CACHE:-$R/mm/llm/vision-cache} MIMO_VLLM_TIMING=1
export MIMO_MM_DRAM_LOG_S=${MIMO_MM_DRAM_LOG_S:-120}
export MIMO_TT_CACHE=${MIMO_TT_CACHE:-$R/tt-inference-server/persistent_volume/volume_id_mimo_v2_6_flash_mtp-MiMo-V2.6-Flash-MOPD-vNone/tt_metal_cache/cache_MiMo-V2.6-Flash-MOPD/P150x8}
export TT_MODEL_CLASS_OVERRIDES=TTMiMoV2OmniForCausalLM=models.experimental.mimo_v2_6_flash.tt.generator_vllm:MiMoV2OmniForCausalLM
export MIMO_SPEC_DECODE=mtp MIMO_SPEC_SLOTS=8 MIMO_SPEC_GROUPS=adaptive MIMO_SDPA_FULL_GRID=1
export TT_PROXY_FETCH_MEDIA=${TT_PROXY_FETCH_MEDIA:-1}
TT='{"tt": {"trace_mode": "decode_only", "trace_region_size": 1200000000, "l1_small_size": 65536, "dispatch_core_axis": "col", "sample_on_device_mode": "decode_only", "prefill_chunk_alignment": 256}}'
SPEC='{"method": "custom_class", "model": "vllm_tt_plugin.model_owned_drafter", "num_speculative_tokens": 3}'
MEDIA_ARGS=()
# Extra engine arguments, one per line in EXTRA_VLLM_ARGS_LINES (e.g. --media-io-kwargs and its JSON, or
# --allowed-local-media-path and a directory); GEN_CONFIG replaces the generation-config override (JSON).
EXTRA_VLLM_ARGS=()
GEN_CONFIG=${GEN_CONFIG:-'{"max_new_tokens": null}'}
[ -n "${EXTRA_VLLM_ARGS_LINES:-}" ] && mapfile -t EXTRA_VLLM_ARGS <<< "$EXTRA_VLLM_ARGS_LINES"
case "$TT_PROXY_FETCH_MEDIA" in 1|true|yes|on) MEDIA_ARGS=(--allowed-media-domains fetched-by-the-proxy.invalid) ;; esac

cd "$MODEL_SRC"  # the model's kernel sources resolve against the CWD first
LOCKS=()  # each row's lock wrapper (it starts the engine in its own session)
BACKENDS=()
LB=
echo "starting" > "$LOGS/status"

engine_group() {  # the process group of the engine a lock wrapper started (its only child, a session leader)
  pgrep -P "$1" 2>/dev/null | head -n 1
}

stop() {
  trap - INT TERM EXIT
  echo "stopping" > "$LOGS/status"
  local w g t live groups=()
  for w in "${LOCKS[@]}"; do
    g=$(engine_group "$w")
    [ -n "$g" ] && groups+=("$g") && kill -TERM -- "-$g" 2>/dev/null
  done
  [ -n "$LB" ] && kill -TERM "$LB" 2>/dev/null
  for ((t = 0; t < ${STOP_TIMEOUT:-120}; t++)); do
    live=0
    for g in ${groups[@]+"${groups[@]}"}; do pgrep -g "$g" > /dev/null 2>&1 && live=1; done
    [ "$live" = 0 ] && break
    sleep 1
  done
  for g in ${groups[@]+"${groups[@]}"}; do kill -KILL -- "-$g" 2>/dev/null; done
  [ -n "$LB" ] && kill -KILL "$LB" 2>/dev/null
  wait 2>/dev/null  # the lock wrappers exit (and release their rows) once their engines are gone
  echo "stopped" > "$LOGS/status"
}
trap 'stop; exit 130' INT
trap 'stop; exit 143' TERM
trap stop EXIT

for r in $ROWS; do
  port=$((BASE_PORT + 1 + r))
  # The lock wrapper exports the row's TT_VISIBLE_DEVICES; one wrapper per engine (each holds its own fd 9).
  "$DEVLOCK_ROW" "$r" "$ROW_TIMEOUT" "$VLLM" serve "$MODEL" \
    --served-model-name XiaomiMiMo/MiMo-V2.6-Flash-MOPD --trust-remote-code \
    --data-parallel-size 1 --max-num-seqs "${SEQS:-8}" --max-model-len "${MAX_LEN:-1048576}" --block-size 64 \
    --enable-chunked-prefill --max-num-batched-tokens "${MNBT:-4096}" --enable-prefix-caching \
    --prefix-cache-retention-interval "${APC_RETENTION:-1024}" \
    --reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice \
    --port "$port" --additional-config "$TT" --speculative-config "$SPEC" \
    --override-generation-config "$GEN_CONFIG" ${MEDIA_ARGS[@]+"${MEDIA_ARGS[@]}"} \
    ${EXTRA_VLLM_ARGS[@]+"${EXTRA_VLLM_ARGS[@]}"} \
    > "$LOGS/row$r.log" 2>&1 &
  LOCKS+=($!)
  BACKENDS+=("127.0.0.1:$port")
done
LB_BACKENDS=$(IFS=,; echo "${BACKENDS[*]}")
"$PYTHON" "$HERE/lb.py" --port "$BASE_PORT" --backends "$LB_BACKENDS" > "$LOGS/lb.log" 2>&1 &
LB=$!
echo "rows [$ROWS] on $LB_BACKENDS, proxy on 127.0.0.1:$BASE_PORT; logs in $LOGS"

t0=$(date +%s)
ready() {
  local r
  for r in $ROWS; do grep -q "Application startup complete" "$LOGS/row$r.log" 2>/dev/null || return 1; done
}
until ready; do
  for w in "${LOCKS[@]}" "$LB"; do
    kill -0 "$w" 2>/dev/null || { echo "process $w exited before the engines were ready"; exit 1; }
  done
  sleep 10
done
echo "ready after $(( $(date +%s) - t0 )) s" | tee "$LOGS/status"
wait -n "${LOCKS[@]}" "$LB"
status=$?
echo "a process exited with status $status; stopping the others"
exit "$status"
