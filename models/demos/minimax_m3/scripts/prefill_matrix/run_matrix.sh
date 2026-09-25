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
set -uo pipefail
PKG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TT_METAL_HOME=${TT_METAL_HOME:-$(cd "$PKG_DIR/../../../../.." && pwd)}
export TT_METAL_HOME
source "$PKG_DIR/matrix_common.sh"
JOB=${JOB:?}; HOSTS=${HOSTS:?}; STAGES=${STAGES:-16}; USERS=${USERS:-4}; ITERS=${ITERS:-3}; REPS=${REPS:-1}
CACHED=${CACHED:-0,61440,143360,312320,552960,860160}
WORK=${WORK:-$TT_METAL_HOME/generated/m3_prefill_matrix/$(date +%Y%m%d_%H%M%S)}; mkdir -p "$WORK"
OUT=${OUT:-$WORK/results_${STAGES}stage.jsonl}
echo "[matrix] $(date) job=$JOB hosts=$HOSTS stages=$STAGES users=$USERS iters=$ITERS reps=$REPS cached=$CACHED commit=$(git -C "$TT_METAL_HOME" rev-parse --short HEAD)"
echo "[matrix] work=$WORK out=$OUT"
first=1; failed=0
for rep in $(seq 1 "$REPS"); do
  for C in ${CACHED//,/ }; do
    R=0; [ $first = 1 ] && R=1; first=0     # reset the galaxies once, before the first runner launch
    JOB=$JOB HOSTS=$HOSTS STAGES=$STAGES CACHED=$C ITERS=$ITERS USERS=$USERS RESET=$R WORK=$WORK OUT=$OUT "$PKG_DIR/matrix_row.sh"; rc=$?
    echo "[matrix] rep $rep row cached=$C exit=$rc"; [ $rc -ne 0 ] && failed=$((failed + 1))
  done
done
# shut the last runner down
PREV=$(cat "$WORK/last_runner_log" 2>/dev/null)
if [ -n "$PREV" ] && ! grep -q '^EXIT=' "$PREV"; then
  IFS=, read -r R0 _ <<< "$HOSTS"; CAP=$(grep -o 'capacity=[0-9]*' "$PREV" | head -1 | cut -d= -f2)
  srun --jobid="$JOB" --overlap -N1 -n1 -w "$R0" bash -c "$MATRIX_ULIMITS; cd $TT_METAL_HOME && source python_env/bin/activate && env $(matrix_producer_env ${CAP:-56320}) timeout 300 python3 $PKG_DIR/matrix_shutdown.py" 2>&1 | grep -v '^srun: '
fi
echo; echo "[matrix] RESULTS ($OUT):"; python3 "$PKG_DIR/matrix_table.py" "$OUT"
if [ $failed -ne 0 ]; then echo "[matrix] $failed row(s) FAILED -- the matrix above is partial"; exit 1; fi
