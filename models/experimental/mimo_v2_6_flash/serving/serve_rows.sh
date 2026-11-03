#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# MiMo-V2.6-Flash on a Blackhole Galaxy as one vLLM engine per row, behind a least-connections proxy (lb.py).
# Each engine runs --data-parallel-size 1 on its row's 8 chips (TT_VISIBLE_DEVICES, MESH_DEVICE=(1,8)), so a row
# prefills while the others decode; --data-parallel-size 4 instead runs four lanes in one engine, which prefill or
# decode together. The proxy listens on BASE_PORT, the engines on BASE_PORT + 1 + row.
#
# Needs a vllm-tt-plugin that runs a model declaring folds_dp_into_lanes as one lane at --data-parallel-size 1
# (branch mimo-v2.6-flash/single-lane): the lane path chains speculative verifies and accepts sampled ones on device.
# Without it an engine takes the plugin's single-engine path, which decodes as fast but verifies a sampled request on
# the host (about a tenth of the speed with MTP).
#
# Usage: serve_rows.sh <log dir> [max_model_len] [extra vllm serve args, the same for every engine]
# Stop it with Ctrl-C or SIGTERM (started in the background by a script, it ignores SIGINT, as bash does). Each engine
# runs in its own process group; stopping sends SIGTERM to every group (the API server and its EngineCore) and SIGKILL
# to what is left after STOP_TIMEOUT seconds. SIGTERM, not SIGINT: an engine started in the background inherits SIGINT
# ignored, and its EngineCore installs a handler only once the model is loaded, so SIGINT would not stop one that is
# loading. When an engine or the proxy exits, the others are stopped and the script exits with that process's status.
# env:
#   TT_METAL_HOME    the tt-metal build (required); MODEL_SRC the model checkout (default: this file's repository)
#   PLUGIN_SRC       a plugin checkout's src/ to use instead of the installed plugin
#   MODEL            checkpoint path (default XiaomiMiMo/MiMo-V2.6-Flash-MOPD); VLLM: the vllm executable (vllm);
#                    PYTHON: the interpreter for lb.py (default: the one next to VLLM; it needs aiohttp, as vLLM does)
#   ROWS             rows to serve (default "0 1 2 3")
#   ROW_DEVICES      each row's chips, space-separated groups. By default discover_rows.py finds them before any engine
#                    starts (chip ids depend on the machine); it opens the whole mesh, so nothing else may use it then
#   BASE_PORT        default 8000; SEQS: --max-num-seqs per engine (default 4; 8 for MIMO_SPEC_SLOTS=8)
#   MNBT             --max-num-batched-tokens, the prefill chunk (default 4096, as tt-inference-server's MiMo spec)
#   APC_RETENTION    --prefix-cache-retention-interval (default 1024; vLLM's 0 keeps a SWA layer's cached KV only at the
#                    end of each prompt, see doc/vllm_integration/README.md)
#   MIMO_KV_GB_PER_CHIP  the model's KV pool per chip; unset, the model's default (6.7 with the global KV pair split,
#                    vLLM's default: 4 x 1M per row), which is what tt-inference-server's MiMo spec sets
#   STOP_TIMEOUT     seconds the engines get to stop on SIGINT before SIGKILL (default 120)
#   SAMPLE_ON_DEVICE decode_only|all (plugin sample_on_device_mode; default host sampling); TRACE_REGION bytes
#   OMP_NUM_THREADS  torch's intra-op threads per engine (default 8). Its default, one per core, oversubscribes the
#                    host with four engines: every small torch op of a step then waits on spinning pools, which made
#                    an MTP verify step 60 ms instead of 23.
set -u
LOGS=$(realpath -m "${1:?log dir}"); mkdir -p "$LOGS"
MAX_LEN=${2:-32768}
shift 2 || shift $#
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${TT_METAL_HOME:?set TT_METAL_HOME to the tt-metal build}"
MODEL_SRC=${MODEL_SRC:-$(cd "$HERE/../../../.." && pwd)}
MODEL=${MODEL:-XiaomiMiMo/MiMo-V2.6-Flash-MOPD}
VLLM=${VLLM:-vllm}
PYTHON=${PYTHON:-$(dirname "$(command -v "$VLLM")")/python}
ROWS=${ROWS:-0 1 2 3}
BASE_PORT=${BASE_PORT:-8000}
export PYTHONPATH=$TT_METAL_HOME/ttnn:$MODEL_SRC${PLUGIN_SRC:+:$PLUGIN_SRC}${PYTHONPATH:+:$PYTHONPATH}
if [ -z "${ROW_DEVICES:-}" ]; then
  "$PYTHON" "$HERE/discover_rows.py" > "$LOGS/discover_rows.log" 2>&1 \
    || { echo "discover_rows.py failed; see $LOGS/discover_rows.log" >&2; exit 1; }
  ROW_DEVICES=$(sed -n 's/^ROW_DEVICES=//p' "$LOGS/discover_rows.log" | tail -n 1)
fi
read -r -a DEVICES <<< "$ROW_DEVICES"
for r in $ROWS; do
  [ -n "${DEVICES[$r]:-}" ] || { echo "no chips for row $r in ROW_DEVICES=\"$ROW_DEVICES\"" >&2; exit 1; }
done
[ -n "${PLUGIN_SRC:-}" ] && export VLLM_TT_PLUGIN_ROOT=$(dirname "$PLUGIN_SRC")
export MESH_DEVICE="(1,8)" VLLM_PLUGINS=tt,tt_model_registry OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
TT_CONFIG="{\"trace_mode\": \"decode_only\", \"trace_region_size\": ${TRACE_REGION:-1000000000}, \"l1_small_size\": 65536, \"dispatch_core_axis\": \"col\"${SAMPLE_ON_DEVICE:+, \"sample_on_device_mode\": \"$SAMPLE_ON_DEVICE\"}}"
cd "$MODEL_SRC"  # kernel sources resolve against the CWD first
ENGINES=()  # each engine's PID, which leads its process group (setsid)
BACKENDS=()
LB=
stop() {
  # Whole process groups: an EngineCore can outlive its API server (one stopped while it loads, for instance).
  trap - INT TERM EXIT
  local p t live idle=0
  for p in "${ENGINES[@]}"; do kill -TERM -- "-$p" 2>/dev/null; done
  [ -n "$LB" ] && kill -TERM "$LB" 2>/dev/null
  for ((t = 0; t < ${STOP_TIMEOUT:-120}; t++)); do
    # Each engine leads its own session (setsid), so -s lists its group; zombies (exited, not yet reaped) don't count.
    live=$(for p in "${ENGINES[@]}"; do ps -o stat=,args= -s "$p" 2>/dev/null; done | awk '$1 !~ /^Z/')
    [ -z "$live" ] && break
    # An API server still waiting for its EngineCore to start cannot run its SIGTERM handler, and multiprocessing's
    # resource tracker ignores SIGTERM: once no EngineCore (the process holding the chips) is left, give them 10 s.
    if grep -q "VLLM::EngineCore" <<< "$live"; then idle=0; else idle=$((idle + 1)); [ "$idle" -ge 10 ] && break; fi
    sleep 1
  done
  for p in "${ENGINES[@]}"; do kill -KILL -- "-$p" 2>/dev/null; done
  [ -n "$LB" ] && kill -KILL "$LB" 2>/dev/null
  wait 2>/dev/null
}
trap 'stop; exit 130' INT
trap 'stop; exit 143' TERM
trap stop EXIT
for r in $ROWS; do
  TT_VISIBLE_DEVICES=${DEVICES[$r]} setsid "$VLLM" serve "$MODEL" \
    --served-model-name XiaomiMiMo/MiMo-V2.6-Flash-MOPD --trust-remote-code \
    --data-parallel-size 1 --max-num-seqs "${SEQS:-4}" --max-model-len "$MAX_LEN" --block-size 64 \
    --enable-chunked-prefill --max-num-batched-tokens "${MNBT:-4096}" --enable-prefix-caching \
    --prefix-cache-retention-interval "${APC_RETENTION:-1024}" \
    --reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice \
    --port $((BASE_PORT + 1 + r)) --additional-config "{\"tt\": $TT_CONFIG}" "$@" > "$LOGS/row$r.log" 2>&1 &
  ENGINES+=($!)
  BACKENDS+=("127.0.0.1:$((BASE_PORT + 1 + r))")
done
LB_BACKENDS=$(IFS=,; echo "${BACKENDS[*]}")
"$PYTHON" "$HERE/lb.py" --port "$BASE_PORT" --backends "$LB_BACKENDS" > "$LOGS/lb.log" 2>&1 &
LB=$!
echo "rows [$ROWS] on ports $LB_BACKENDS, proxy on 127.0.0.1:$BASE_PORT; logs in $LOGS"
wait -n "${ENGINES[@]}" "$LB"
status=$?
echo "a process exited with status $status; stopping the others"
exit "$status"
