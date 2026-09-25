#!/bin/bash
# One ROW of the MiniMax-M3 prefill matrix (one cached-token count, all new-token counts), run from a LOGIN node:
#   JOB=<slurm job> HOSTS=<A,B,C,D> STAGES=16 CACHED=<C> [USERS=4] [ITERS=3] [RESET=0|1] [NEW=...] [WORK=...] matrix_row.sh
#  1. if a runner from a previous row is live (WORK/last_runner_log without EXIT=), send SHUTDOWN and wait for it
#  2. RESET=1: tt-smi -glx_reset every host in parallel (do this for the first row of a session)
#  3. launch matrix_runner.sh on rank 0's host with this row's KV capacity, wait for STAGES x "setup complete"
#  4. run matrix_producer.py: idle-pipeline TTFT per cell, then (USERS>0) the LOADED aggregate-throughput pass
# Results append to $OUT (JSONL, one line per idle iteration and one per loaded cell). See README.md.
set -uo pipefail
PKG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TT_METAL_HOME=${TT_METAL_HOME:-$(cd "$PKG_DIR/../../../../.." && pwd)}
JOB=${JOB:?}; HOSTS=${HOSTS:?comma list, rank-0 host first}; STAGES=${STAGES:-16}; CACHED=${CACHED:?}
ITERS=${ITERS:-3}; RESET=${RESET:-0}; USERS=${USERS:-0}; NEW=${NEW:-640,1600,3072,5120,32768,51200}
WORK=${WORK:-$TT_METAL_HOME/generated/m3_prefill_matrix}; mkdir -p "$WORK"
OUT=${OUT:-$WORK/results_${STAGES}stage.jsonl}
IFS=, read -r -a HOST_ARR <<< "$HOSTS"; R0=${HOST_ARR[0]}
RANKS_PER_HOST=$(( STAGES / ${#HOST_ARR[@]} )); LAST=$((STAGES - 1))
HOSTLIST=$(printf "%s:$RANKS_PER_HOST," "${HOST_ARR[@]}"); HOSTLIST=${HOSTLIST%,}
[ $((RANKS_PER_HOST * ${#HOST_ARR[@]})) -eq "$STAGES" ] || { echo "STAGES=$STAGES not divisible over ${#HOST_ARR[@]} hosts"; exit 2; }
ULIM="ulimit -u 2318132; ulimit -n 131072; ulimit -t unlimited;"   # a step on a host running ranks cannot even fork otherwise
PENV="PREFILL_MODEL=minimax_m3 PREFILL_H2D_SERVICE_ID=ds_prefill PREFILL_SP=2 PREFILL_TP=4 PREFILL_NUM_LAYERS=60 PREFILL_CHUNK_SIZE=5120 PREFILL_NUM_USERS=1 TT_METAL_HOME=$TT_METAL_HOME PYTHONPATH=$TT_METAL_HOME"
[ -n "${HF_MODEL:-}" ] && PENV="$PENV HF_MODEL=$HF_MODEL"; [ -n "${TT_CACHE_PATH:-}" ] && PENV="$PENV TT_CACHE_PATH=$TT_CACHE_PATH"
[ -n "${PREFILL_TRACE_DIR:-}" ] && PENV="$PENV PREFILL_TRACE_DIR=$PREFILL_TRACE_DIR"
log() { echo "[row C=$CACHED] $(date +%T) $*"; }

# 1. shutdown a live runner (previous row)
if [ -f "$WORK/last_runner_log" ] && ! grep -q '^EXIT=' "$(cat "$WORK/last_runner_log")"; then
  PREV=$(cat "$WORK/last_runner_log"); PREVCAP=$(grep -o 'capacity=[0-9]*' "$PREV" | head -1 | cut -d= -f2)
  log "shutting down live runner ($PREV)"
  srun --jobid="$JOB" --overlap -N1 -n1 -w "$R0" bash -c "$ULIM cd $TT_METAL_HOME && source python_env/bin/activate && env $PENV PREFILL_MAX_SEQ_LEN=${PREVCAP:-56320} python3 $PKG_DIR/matrix_shutdown.py" 2>&1 | grep -v '^srun: '
  for i in $(seq 1 120); do grep -q '^EXIT=' "$PREV" && break; sleep 2; done
  if grep -q '^EXIT=' "$PREV"; then log "runner exited: $(grep '^EXIT=' "$PREV")"; else
    log "runner did not exit in 240 s; killing on every host"
    for h in "${HOST_ARR[@]}"; do srun --jobid="$JOB" --overlap -N1 -n1 -w "$h" bash -c "$ULIM pkill -9 -u \$USER -f '[p]refill_runner'; pkill -9 -u \$USER -f '[t]trun.py'; pkill -9 -u \$USER -f '[p]rterun'" 2>/dev/null; done
    RESET=1; sleep 5
  fi
fi
# 2. reset
if [ "$RESET" = "1" ]; then
  log "resetting ${#HOST_ARR[@]} galaxies"
  for h in "${HOST_ARR[@]}"; do srun --jobid="$JOB" --overlap -N1 -n1 -w "$h" bash -c 'tt-smi -glx_reset 2>&1 | tail -1; echo "[reset] $(hostname -s) done"' 2>&1 | grep -v '^srun: ' & done; wait
fi
# 3. runner
LOG=$WORK/runner${STAGES}_c${CACHED}_$(date +%H%M%S).log; echo "$LOG" > "$WORK/last_runner_log"
log "launching runner -> $LOG"
( srun --jobid="$JOB" --overlap -N1 -n1 -w "$R0" env STAGES="$STAGES" CACHED="$CACHED" USERS="${USERS:-1}" WORK="$WORK" HOSTS="$HOSTLIST" \
    TT_METAL_HOME="$TT_METAL_HOME" ${HF_MODEL:+HF_MODEL=$HF_MODEL} ${TT_CACHE_PATH:+TT_CACHE_PATH=$TT_CACHE_PATH} "$PKG_DIR/matrix_runner.sh" > "$LOG" 2>&1; echo "EXIT=$?" >> "$LOG" ) &
n=0
for i in $(seq 1 600); do
  n=$(grep -c 'setup complete' "$LOG" 2>/dev/null); [ "${n:-0}" -ge "$STAGES" ] && break
  grep -qE 'Timed out while waiting for active ethernet|could not fit|^EXIT=' "$LOG" 2>/dev/null && { log "runner FAILED during bring-up:"; grep -E 'Timed out while|could not fit|^EXIT=' "$LOG" | head -3; exit 2; }
  sleep 2
done
[ "${n:-0}" -ge "$STAGES" ] || { log "runner not ready after 20 min"; exit 2; }
log "runner ready ($n/$STAGES)"
# 4. producer
PLOG=$WORK/producer${STAGES}_c${CACHED}_$(date +%H%M%S).log; echo "$PLOG" > "$WORK/last_producer_log"
TIMING_DIR=$(cat "$WORK/last_timing_dir")
log "producer -> $PLOG (out $OUT)"
srun --jobid="$JOB" --overlap -N1 -n1 -w "$R0" bash -c "$ULIM cd $TT_METAL_HOME && source python_env/bin/activate && env $PENV PREFILL_MAX_SEQ_LEN=$((CACHED + 51200)) \
  python3 $PKG_DIR/matrix_producer.py --cached $CACHED --new '$NEW' --iters $ITERS --timing-dir '$TIMING_DIR' --out '$OUT' --last-rank $LAST \
  --label ${STAGES}stage ${USERS:+--users $USERS} ${REQS:+--reqs $REQS} ${SKIP_IDLE:+--skip-idle}" > "$PLOG" 2>&1; rc=$?; echo "EXIT=$rc" >> "$PLOG"
log "producer exit=$rc"; grep -E '\[matrix\] (CELL|LOADED cached)' "$PLOG" | sed 's/.*\[matrix\]/[matrix]/'
exit $rc
