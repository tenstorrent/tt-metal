#!/bin/bash
# SP2 follow-up, Part B: full 60-layer model on one galaxy with the common prefill runner,
# 4 x (2,4) SP=2 vs 2 x (4,4) SP=4. Streams of 48 one-chunk requests: cold, hot-139k, hot-549k (synthetic
# prefix: PREFILL_PRODUCER_PREFIX_TOKENS, the history KV is never written) x in-flight K = stages / open.
# Sync sessions (PREFILL_SYNC_PER_CHUNK=1) give per-stage compute and the hop.
cd "$(dirname "$0")"; HERE=$PWD; source sp2_env.sh; E2E=$BUDGET_RESULTS/e2e
until ! pgrep -f "batch_sp2_a[.]sh" >/dev/null; do sleep 30; done
ulimit -Su "$(ulimit -Hu)" 2>/dev/null
unset $(env | sed -n 's/^\(SLURM[^=]*\)=.*/\1/p')
export PRTE_MCA_ras="^slurm" PRTE_MCA_plm="^slurm"
cd .. && source python_env/bin/activate && export PYTHONPATH=$PWD
export TT_CACHE_PATH=/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill
export PREFILL_MANIFEST=models/demos/minimax_m3/tt/runners/manifests/minimax_m3.json

producer () {  # $1 = out dir, $2 = name, $3 = ranks, $4 = W, rest = env
  local d=$1 name=$2 ranks=$3 W=$4; shift 4
  echo "=== $(basename $d) producer $name $(date -Is)"
  env LOGURU_LEVEL=INFO PREFILL_MODEL=minimax_m3 PREFILL_H2D_SERVICE_ID=ds_prefill \
    PREFILL_TRACE_DIR=$TT_CACHE_PATH/golden/longbook_56320 \
    PREFILL_SP=$((8 / ranks)) PREFILL_TP=4 PREFILL_NUM_LAYERS=60 PREFILL_CHUNK_SIZE=$W PREFILL_MAX_SEQ_LEN=557056 \
    PREFILL_NUM_USERS=2 PREFILL_PRODUCER_INTERLEAVE=round_robin PREFILL_PRODUCER_CHUNKS=1 "$@" \
    timeout 1200 python3 -m models.demos.common.prefill.runners.prefill_producer > "$d/$name.log" 2>&1
  echo "exit=$? $(grep -h 'DONE wall\|drained' "$d/$name.log" | tail -2 | tr '\n' ' ' | cut -c1-300)"
}

session () {  # $1 = binding name, $2 = ranks, $3 = W, rest = "name|ENV ..." producer specs
  local b=$1 ranks=$2 W=$3; shift 3
  grep -q "owner=$BUDGET_LOCK_OWNER" "$BUDGET_LOCK" || { echo "lock not ours; stopping"; exit 3; }
  local d=$E2E/$b; mkdir -p "$d"
  tt-smi -glx_reset > "$d/reset.log" 2>&1
  ./models/demos/common/prefill/runners/run_pipeline_prefill.sh "$HERE/e2e_bindings/$b.yaml" \
    "$(hostname -s):$ranks" > "$d/runner.log" 2>&1 &
  local rpid=$! t0=$(date +%s)
  until grep -q "\[h2d\] descriptor" "$d/runner.log"; do
    sleep 10
    if ! kill -0 $rpid 2>/dev/null || [ $(( $(date +%s) - t0 )) -gt 2400 ]; then
      echo "runner $b did not come up after $(( $(date +%s) - t0 ))s"; kill -TERM $rpid 2>/dev/null; sleep 10
      pkill -KILL -f "models.demos.common.prefill.runners.prefill_runner"; tt-smi -glx_reset >> "$d/reset.log" 2>&1
      return 1
    fi
  done
  echo "runner $b up after $(( $(date +%s) - t0 ))s"
  local n=$# i=0
  for spec in "$@"; do
    i=$((i + 1)); local name=${spec%%|*} envs=${spec#*|}
    [ $i -eq $n ] && envs="$envs PREFILL_SEND_SHUTDOWN=1"
    producer "$d" "$name" $ranks $W $envs
  done
  local t1=$(date +%s)
  while kill -0 $rpid 2>/dev/null && [ $(( $(date +%s) - t1 )) -lt 300 ]; do sleep 5; done
  kill -0 $rpid 2>/dev/null && { echo "runner $b still up; killing"; pkill -TERM -f "models.demos.common.prefill.runners.prefill_runner"; sleep 15; pkill -KILL -f "models.demos.common.prefill.runners.prefill_runner"; }
  tt-smi -glx_reset >> "$d/reset.log" 2>&1
}

G=PREFILL_PRODUCER_MAX_IN_FLIGHT
N48=PREFILL_PRODUCER_MAX_REQUESTS=48
H141=PREFILL_PRODUCER_PREFIX_TOKENS=139264
H549=PREFILL_PRODUCER_PREFIX_TOKENS=548864
streams () {  # $1 = K (stages)
  echo "warm|PREFILL_PRODUCER_MAX_REQUESTS=8 PREFILL_PRODUCER_WARMUP_CHUNKS=2 $G=10000"
  echo "cold_k$1|$N48 $G=$1"; echo "cold_open|$N48 $G=10000"
  echo "h141_warm|PREFILL_PRODUCER_MAX_REQUESTS=4 $H141 $G=10000"
  echo "h141_k$1|$N48 $H141 $G=$1"; echo "h141_open|$N48 $H141 $G=10000"
  echo "h549_warm|PREFILL_PRODUCER_MAX_REQUESTS=4 $H549 $G=10000"
  echo "h549_k$1|$N48 $H549 $G=$1"; echo "h549_open|$N48 $H549 $G=10000"
}
run_session () { local b=$1 r=$2 W=$3 k=$4; mapfile -t S < <(streams $k); session $b $r $W "${S[@]}"; }
sync_specs=("warm|PREFILL_PRODUCER_MAX_REQUESTS=4 PREFILL_PRODUCER_WARMUP_CHUNKS=2 $G=10000"
  "cold_open|PREFILL_PRODUCER_MAX_REQUESTS=24 $G=10000" "h549_open|PREFILL_PRODUCER_MAX_REQUESTS=24 $H549 $G=10000")
split_specs () { echo "warm|PREFILL_PRODUCER_MAX_REQUESTS=8 PREFILL_PRODUCER_WARMUP_CHUNKS=2 $G=10000"
  echo "cold_k$1|$N48 $G=$1"; echo "cold_open|$N48 $G=10000"
  echo "h549_warm|PREFILL_PRODUCER_MAX_REQUESTS=4 $H549 $G=10000"
  echo "h549_k$1|$N48 $H549 $G=$1"; echo "h549_open|$N48 $H549 $G=10000"; }

run_session r4_w4096_sync0 4 4096 4
session r4_w4096_sync1 4 4096 "${sync_specs[@]}"
run_session r4_w8192_sync0 4 8192 4
run_session r2_w4096_sync0 2 4096 2
session r2_w4096_sync1 2 4096 "${sync_specs[@]}"
run_session r2_w8192_sync0 2 8192 2
mapfile -t S < <(split_specs 4); session r4_w4096_sync0_split12-16-16-16 4 4096 "${S[@]}"
mapfile -t S < <(split_specs 2); session r2_w4096_sync0_split24-36 2 4096 "${S[@]}"
echo "Part B done $(date -Is)"
