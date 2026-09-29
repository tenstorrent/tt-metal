#!/bin/bash
# MiniMax-M3 pipeline-prefill PERF MATRIX: (new tokens) x (cached tokens) on a 16-stage quad-galaxy pipeline.
# Matrix values are Agent X dataset percentiles: new 640 (p25) / 1600 (p50) / 3072 (p75) / 6900 (p90) / 51200 (p99)
# (+5120 = one chunk, +32768); cached 61440 (p25) / 143360 (p50) / 312320 (p75) / 552960 (p90) / 860160 (p99) (+0).
# (4 hosts x 4 trays, [2,4] per stage). For each cell: idle-pipeline TTFT (single request) and, with USERS>0,
# fully-loaded aggregate throughput (USERS users streaming back-to-back, fill/drain excluded) + TTFT under load.
# Run from a LOGIN node inside a Slurm allocation that holds the 4 galaxies. See README.md.
#   JOB=<slurm job id> HOSTS=<rank0-host,host2,host3,host4> ./run_matrix.sh
# Env (all optional): STAGES=16|12  USERS=4  ITERS=3  CACHED=0,61440,143360,312320,552960,860160  NEW=640,1600,3072,5120,6900,32768,51200
#   WORK=<shared dir for logs/results>  OUT=<results jsonl>  HF_MODEL / TT_CACHE_PATH / PREFILL_TRACE_DIR  REPS=1 (full passes)
#   REQS / TARGET_CHUNKS / SKIP_IDLE / MAX_NEW are passed through to matrix_row.sh (see its header).
# Ctrl-C / TERM shuts the live runner down (sentinel, then scoped kill) before exiting.
set -uo pipefail
PKG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TT_METAL_HOME=${TT_METAL_HOME:-$(cd "$PKG_DIR/../../../../.." && pwd)}
export TT_METAL_HOME
source "$PKG_DIR/matrix_common.sh"
JOB=${JOB:?}; HOSTS=${HOSTS:?}; STAGES=${STAGES:-16}; USERS=${USERS:-4}; ITERS=${ITERS:-3}; REPS=${REPS:-1}
CACHED=${CACHED:-0,61440,143360,312320,552960,860160}
WORK=${WORK:-$TT_METAL_HOME/generated/m3_prefill_matrix/$(date +%Y%m%d_%H%M%S)}; mkdir -p "$WORK"
OUT=${OUT:-$WORK/results_${STAGES}stage.jsonl}
IFS=, read -r -a HOST_ARR <<< "$HOSTS"
echo "[matrix] $(date) job=$JOB hosts=$HOSTS stages=$STAGES users=$USERS iters=$ITERS reps=$REPS cached=$CACHED new=${NEW:-default} commit=$(git -C "$TT_METAL_HOME" rev-parse --short HEAD)"
echo "[matrix] work=$WORK out=$OUT"

shutdown_last_runner() { matrix_shutdown_runner "$JOB" "$WORK" "${HOST_ARR[@]}"; }
trap 'echo "[matrix] interrupted; shutting the runner down"; trap - INT TERM; shutdown_last_runner; exit 130' INT TERM

first=1; failed=0
for rep in $(seq 1 "$REPS"); do
  for C in ${CACHED//,/ }; do
    R=0; [ "$first" = 1 ] && R=1; first=0     # reset the galaxies before the first launch (and after a failed row)
    JOB=$JOB HOSTS=$HOSTS STAGES=$STAGES CACHED=$C ITERS=$ITERS USERS=$USERS RESET=$R WORK=$WORK OUT=$OUT "$PKG_DIR/matrix_row.sh"; rc=$?
    echo "[matrix] rep $rep row cached=$C exit=$rc"
    [ "$rc" -ne 0 ] && { failed=$((failed + 1)); first=1; }
  done
done
shutdown_last_runner
echo; echo "[matrix] RESULTS ($OUT):"; python3 "$PKG_DIR/matrix_table.py" "$OUT"
if [ "$failed" -ne 0 ]; then echo "[matrix] $failed row(s) FAILED -- the matrix above is partial"; exit 1; fi
