#!/bin/bash
# E7: 2-stage intragalaxy pipeline (2 x (4,4), layers 0-7 | 8-15, W = 4096) via the common prefill runner.
# Session 1 (no per-chunk sync): cold and hot (139264-deep) streams x in-flight K in {1, 2, 4, open}.
# Session 2 (PREFILL_SYNC_PER_CHUNK=1): per-chunk compute and hop time; overlap disabled.
# One runner per session; producers run one after another against it, the last one sends SHUTDOWN.
cd "$(dirname "$0")"; HERE=$PWD; RES=$HERE/results/e7; mkdir -p "$RES"

ulimit -Su "$(ulimit -Hu)" 2>/dev/null
unset $(env | sed -n 's/^\(SLURM[^=]*\)=.*/\1/p')
export PRTE_MCA_ras="^slurm" PRTE_MCA_plm="^slurm"
cd .. && source python_env/bin/activate && export PYTHONPATH=$PWD
export TT_CACHE_PATH=/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill
export PREFILL_MANIFEST=models/demos/minimax_m3/tt/runners/manifests/minimax_m3.json

producer () {  # $1 = log name, rest = extra env
  local name=$1; shift
  echo "=== producer $name $(date -Is)"
  env LOGURU_LEVEL=INFO PREFILL_MODEL=minimax_m3 PREFILL_H2D_SERVICE_ID=ds_prefill \
    PREFILL_TRACE_DIR=$TT_CACHE_PATH/golden/longbook_56320 \
    PREFILL_SP=4 PREFILL_TP=4 PREFILL_NUM_LAYERS=16 PREFILL_CHUNK_SIZE=4096 PREFILL_MAX_SEQ_LEN=147456 \
    PREFILL_NUM_USERS=2 PREFILL_PRODUCER_INTERLEAVE=round_robin "$@" \
    timeout 900 python3 -m models.demos.common.prefill.runners.prefill_producer > "$RES/$name.log" 2>&1
  echo "exit=$? $(grep -h 'DONE wall\|drained' "$RES/$name.log" | tail -2 | tr '\n' ' ')"
}

session () {  # $1 = sync flag, then producer specs "name|ENV=.. ENV=.."
  local sync=$1; shift
  tt-smi -glx_reset > "$RES/reset_sync$sync.log" 2>&1
  ./models/demos/common/prefill/runners/run_pipeline_prefill.sh "$HERE/e7_binding_sync$sync.yaml" \
    "$(hostname -s):2" > "$RES/runner_sync$sync.log" 2>&1 &
  local rpid=$! t0=$(date +%s)
  until grep -q "\[h2d\] descriptor" "$RES/runner_sync$sync.log"; do
    sleep 10
    if ! kill -0 $rpid 2>/dev/null || [ $(( $(date +%s) - t0 )) -gt 1800 ]; then
      echo "runner sync=$sync did not come up"; kill -TERM $rpid 2>/dev/null; return 1
    fi
  done
  echo "runner sync=$sync up after $(( $(date +%s) - t0 ))s"
  local n=$# i=0
  for spec in "$@"; do
    i=$((i + 1)); local name=${spec%%|*} envs=${spec#*|}
    [ $i -eq $n ] && envs="$envs PREFILL_SEND_SHUTDOWN=1"
    producer "$name" $envs
  done
  local t1=$(date +%s)
  while kill -0 $rpid 2>/dev/null && [ $(( $(date +%s) - t1 )) -lt 300 ]; do sleep 5; done
  kill -0 $rpid 2>/dev/null && { echo "runner sync=$sync still up; killing"; pkill -TERM -f prefill_runner; sleep 15; pkill -KILL -f prefill_runner; }
  tt-smi -glx_reset >> "$RES/reset_sync$sync.log" 2>&1
}

COLD="PREFILL_PRODUCER_CHUNKS=4 PREFILL_PRODUCER_MAX_REQUESTS=12"
HOT="PREFILL_PRODUCER_CHUNKS=1 PREFILL_PRODUCER_MAX_REQUESTS=48 PREFILL_PRODUCER_PREFIX_TOKENS=139264"
session 0 \
  "warm|$COLD PREFILL_PRODUCER_MAX_REQUESTS=4 PREFILL_PRODUCER_WARMUP_CHUNKS=2 PREFILL_PRODUCER_MAX_IN_FLIGHT=10000" \
  "cold_k1|$COLD PREFILL_PRODUCER_MAX_IN_FLIGHT=1" "cold_k2|$COLD PREFILL_PRODUCER_MAX_IN_FLIGHT=2" \
  "cold_k4|$COLD PREFILL_PRODUCER_MAX_IN_FLIGHT=4" "cold_open|$COLD PREFILL_PRODUCER_MAX_IN_FLIGHT=10000" \
  "hot_warm|$HOT PREFILL_PRODUCER_MAX_REQUESTS=4 PREFILL_PRODUCER_MAX_IN_FLIGHT=10000" \
  "hot_k1|$HOT PREFILL_PRODUCER_MAX_IN_FLIGHT=1" "hot_k2|$HOT PREFILL_PRODUCER_MAX_IN_FLIGHT=2" \
  "hot_k4|$HOT PREFILL_PRODUCER_MAX_IN_FLIGHT=4" "hot_open|$HOT PREFILL_PRODUCER_MAX_IN_FLIGHT=10000"
session 1 \
  "sync_warm|$COLD PREFILL_PRODUCER_MAX_REQUESTS=2 PREFILL_PRODUCER_WARMUP_CHUNKS=2 PREFILL_PRODUCER_MAX_IN_FLIGHT=10000" \
  "sync_cold|$COLD PREFILL_PRODUCER_MAX_IN_FLIGHT=10000" "sync_hot|$HOT PREFILL_PRODUCER_MAX_REQUESTS=24 PREFILL_PRODUCER_MAX_IN_FLIGHT=10000"
echo "E7 done $(date -Is)"
