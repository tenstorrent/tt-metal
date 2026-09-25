#!/bin/bash
# One ROW of the MiniMax-M3 prefill matrix (one cached-token count, all new-token counts), run from a LOGIN node:
#   JOB=<slurm job> HOSTS=<A,B,C,D> WORK=<shared dir> CACHED=<C> [STAGES=16] [USERS=0] [ITERS=3] [RESET=0|1] [NEW=...] matrix_row.sh
#  1. if a runner from a previous row is live (WORK/last_runner_log without EXIT=), send SHUTDOWN and wait for it
#  2. RESET=1: tt-smi -glx_reset every host in parallel (do this for the first row of a session)
#  3. launch matrix_runner.sh on rank 0's host with this row's KV capacity, wait for STAGES x "setup complete"
#  4. run matrix_producer.py: idle-pipeline TTFT per cell, then (USERS>0) the LOADED aggregate-throughput pass
# WORK holds this session's mutable state (last_runner_log, last_timing_dir): one WORK per session, never shared.
# Optional: OUT (results JSONL), MAX_NEW (capacity = CACHED + MAX_NEW, default 51200), REQS (requests per user),
# TARGET_CHUNKS (chunks per loaded stream when REQS is unset, default 240), SKIP_IDLE=1 (loaded pass only).
# Results append to $OUT (JSONL, one line per idle iteration and one per loaded cell). See README.md.
set -uo pipefail
PKG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TT_METAL_HOME=${TT_METAL_HOME:-$(cd "$PKG_DIR/../../../../.." && pwd)}
export TT_METAL_HOME
source "$PKG_DIR/matrix_common.sh"
JOB=${JOB:?}; HOSTS=${HOSTS:?comma list, rank-0 host first}; STAGES=${STAGES:-16}; CACHED=${CACHED:?}
ITERS=${ITERS:-3}; RESET=${RESET:-0}; USERS=${USERS:-0}; NEW=${NEW:-640,1600,3072,5120,6900,32768,51200}
MAX_NEW=${MAX_NEW:-$MATRIX_MAX_NEW}
WORK=${WORK:?set WORK=<shared per-session dir> (run_matrix.sh sets it)}; mkdir -p "$WORK"
OUT=${OUT:-$WORK/results_${STAGES}stage.jsonl}
IFS=, read -r -a HOST_ARR <<< "$HOSTS"; R0=${HOST_ARR[0]}; LAST=$((STAGES - 1))
# The bindings put 4 trays (= 4 ranks) on every host and wire host h's last tray to host h+1's first one.
RANKS_PER_HOST=4
[ $((STAGES % RANKS_PER_HOST)) -eq 0 ] && [ "${#HOST_ARR[@]}" -eq $((STAGES / RANKS_PER_HOST)) ] \
  || { echo "STAGES=$STAGES needs exactly $((STAGES / RANKS_PER_HOST)) hosts (bindings are $RANKS_PER_HOST trays/host), got ${#HOST_ARR[@]}: $HOSTS"; exit 2; }
for h in "${HOST_ARR[@]}"; do [ -n "$h" ] || { echo "empty host in HOSTS=$HOSTS"; exit 2; }; done
HOSTLIST=$(printf "%s:$RANKS_PER_HOST," "${HOST_ARR[@]}"); HOSTLIST=${HOSTLIST%,}
[ $((CACHED % MATRIX_CHUNK)) -eq 0 ] || { echo "CACHED=$CACHED is not a multiple of the chunk ($MATRIX_CHUNK)"; exit 2; }
for n in ${NEW//,/ }; do [ "$n" -le "$MAX_NEW" ] || { echo "NEW=$n exceeds MAX_NEW=$MAX_NEW (the runner's capacity is CACHED + MAX_NEW)"; exit 2; }; done
CAP=$((CACHED + MAX_NEW))
log() { echo "[row C=$CACHED] $(date +%T) $*"; }

# 1. shutdown a live runner (previous row); a runner that had to be killed leaves the devices in an unknown state
matrix_shutdown_runner "$JOB" "$WORK" "${HOST_ARR[@]}" || RESET=1
# 2. reset
if [ "$RESET" = "1" ]; then
  log "resetting ${#HOST_ARR[@]} galaxies"
  pids=()
  for h in "${HOST_ARR[@]}"; do
    matrix_srun "$JOB" "$h" "timeout 900 tt-smi -glx_reset > $(matrix_q "$WORK/reset_$h.log") 2>&1; rc=\$?; echo \"[reset] \$(hostname -s) rc=\$rc\"; exit \$rc" & pids+=($!)
  done
  reset_failed=0; for p in "${pids[@]}"; do wait "$p" || reset_failed=1; done
  [ "$reset_failed" = 0 ] || { log "galaxy reset FAILED (see $WORK/reset_*.log)"; exit 2; }
fi
# 3. runner: its own session (setsid) so a Ctrl-C on the login shell never reaches the srun step; the runner is
#    always torn down through the deterministic sentinel -> wait -> scoped-kill path (matrix_shutdown_runner).
LOG=$WORK/runner${STAGES}_c${CACHED}_$(date +%Y%m%d_%H%M%S).log; echo "$LOG" > "$WORK/last_runner_log"
log "launching runner -> $LOG"
LOG=$LOG setsid -f bash -c 'srun "$@" > "$LOG" 2>&1; echo "EXIT=$?" >> "$LOG"' _ \
    --jobid="$JOB" --overlap -N1 -n1 -w "$R0" env STAGES="$STAGES" CACHED="$CACHED" USERS="$USERS" MAX_NEW="$MAX_NEW" WORK="$WORK" HOSTS="$HOSTLIST" \
    TT_METAL_HOME="$TT_METAL_HOME" ${HF_MODEL:+"HF_MODEL=$HF_MODEL"} ${TT_CACHE_PATH:+"TT_CACHE_PATH=$TT_CACHE_PATH"} "$PKG_DIR/matrix_runner.sh"
n=0
for _ in $(seq 1 600); do
  n=$(grep -c 'setup complete' "$LOG" 2>/dev/null); [ "${n:-0}" -ge "$STAGES" ] && break
  grep -qE 'Timed out while waiting for active ethernet|could not fit|^EXIT=' "$LOG" 2>/dev/null && { log "runner FAILED during bring-up:"; grep -E 'Timed out while|could not fit|^EXIT=' "$LOG" | head -3; exit 2; }
  sleep 2
done
[ "${n:-0}" -ge "$STAGES" ] || { log "runner not ready after 20 min"; exit 2; }
log "runner ready ($n/$STAGES)"
# 4. producer (on rank 0's host, same host as rank 0 so the chunk-index alignment check in the producer holds)
PLOG=$WORK/producer${STAGES}_c${CACHED}_$(date +%Y%m%d_%H%M%S).log; echo "$PLOG" > "$WORK/last_producer_log"
TIMING_DIR=$(grep -o 'timing_dir=[^ ]*' "$LOG" | head -1 | cut -d= -f2-)
[ -n "$TIMING_DIR" ] || { log "runner log has no timing_dir= token"; exit 2; }
log "producer -> $PLOG (out $OUT)"
matrix_srun "$JOB" "$R0" "cd $(matrix_q "$TT_METAL_HOME") && source python_env/bin/activate && env $(matrix_producer_env "$CAP") \
  python3 $(matrix_q "$PKG_DIR/matrix_producer.py") --cached $CACHED --new $(matrix_q "$NEW") --iters $ITERS --timing-dir $(matrix_q "$TIMING_DIR") \
  --out $(matrix_q "$OUT") --last-rank $LAST --label ${STAGES}stage --users $USERS ${REQS:+--reqs $REQS} ${TARGET_CHUNKS:+--target-chunks $TARGET_CHUNKS} ${SKIP_IDLE:+--skip-idle}" > "$PLOG"; rc=$?; echo "EXIT=$rc" >> "$PLOG"
log "producer exit=$rc"; grep -E '\[matrix\] (CELL|LOADED cached)' "$PLOG" | sed 's/.*\[matrix\]/[matrix]/'
exit "$rc"
