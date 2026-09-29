#!/bin/bash
# M3 op microbenchmarks on a (2,4) sub-mesh of the 8x4 galaxy, one process per script.
# Usage: run_bench.sh {experts|moe_reduce|msa|all} [script args...]
#   BENCH_TIMEOUT=<s> (default 1200)  BENCH_STALL=<s> without log output (default 300)  BENCH_TAG=<run-id suffix>  BENCH_TIMING=device|wall  BENCH_LOCK_OWNER (default vmelnykov-ops-agent)
# Per run: lock check -> tt-smi -glx_reset -> timeout python3 bench_<name>.py -> reset again on failure.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$(dirname "$HERE")"                      # m3_budget_study/results_ops
export TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$HERE/../../.." && pwd)}"
LOCK="$RES/.lock"; OWNER="${BENCH_LOCK_OWNER:-vmelnykov-ops-agent}"
LOGS="$RES/logs"; mkdir -p "$LOGS"
TIMEOUT="${BENCH_TIMEOUT:-1200}" STALL="${BENCH_STALL:-300}"

check_lock() {
  grep -q "owner=$OWNER" "$LOCK" 2>/dev/null || { echo "lock $LOCK is not owned by $OWNER; refusing to run"; exit 3; }
}

[ $# -ge 1 ] || { echo "usage: $0 {experts|moe_reduce|msa|all} [args...]"; exit 2; }
WHAT="$1"; shift
case "$WHAT" in
  all) NAMES="experts moe_reduce msa" ;;
  experts|moe_reduce|msa) NAMES="$WHAT" ;;
  *) echo "unknown bench '$WHAT'"; exit 2 ;;
esac
check_lock

cd "$TT_METAL_HOME"
source python_env/bin/activate
export PYTHONPATH="$TT_METAL_HOME" LOGURU_LEVEL=INFO
export TT_MESH_GRAPH_DESC_PATH="${TT_MESH_GRAPH_DESC_PATH:-$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto}"
if [ "${BENCH_TIMING:-device}" = device ]; then
  export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1
  export TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1
fi
ulimit -Su "$(ulimit -Hu)" 2>/dev/null

rc_all=0
for name in $NAMES; do
  check_lock
  RUN_ID="bench_${name}_$(date +%Y%m%d_%H%M%S)${BENCH_TAG:+_$BENCH_TAG}"
  LOG="$LOGS/$RUN_ID.log"
  { echo "run_id=$RUN_ID"; echo "git_sha=$(git rev-parse HEAD)"; echo "dirty=$(git status --porcelain -uno | wc -l)"
    echo "date=$(date -Is)"; echo "args=$*"; env | grep -E '^(BENCH_|M3_|TT_)' | sort; } > "$LOGS/$RUN_ID.env"
  if ! env -u TT_VISIBLE_DEVICES tt-smi -glx_reset > "$LOGS/$RUN_ID.reset" 2>&1; then
    echo "STATUS=ERROR reset failed" | tee -a "$LOG"; rc_all=1; continue
  fi
  T0=$(date +%s); START_UTC=$(date -u +%FT%TZ)
  env -u TT_VISIBLE_DEVICES timeout -k 30 "$TIMEOUT" python3 -u "$HERE/bench_${name}.py" "$@" > "$LOG" 2>&1 &
  PID=$!; STALLED=0
  while kill -0 $PID 2>/dev/null; do  # stall watchdog: no log output for STALL s -> kill
    sleep 10
    if [ $(( $(date +%s) - $(stat -c %Y "$LOG") )) -gt "$STALL" ]; then
      STALLED=1; pkill -TERM -P $PID; sleep 15; pkill -KILL -P $PID; kill -KILL $PID 2>/dev/null; break
    fi
  done
  wait $PID; rc=$?
  if [ $STALLED -eq 1 ]; then STATUS=HANG
  elif [ $rc -eq 0 ] && grep -q "DONE ->" "$LOG"; then STATUS=OK
  elif [ $rc -eq 124 ] || [ $rc -eq 137 ]; then STATUS=TIMEOUT
  else STATUS="ERROR rc=$rc"; fi
  echo "STATUS=$STATUS elapsed=$(( $(date +%s) - T0 ))s" | tee -a "$LOG"
  python3 - "$RES/runs.csv" "$RUN_ID" "$START_UTC" "$(date -u +%FT%TZ)" "$STATUS" "${LOG#$TT_METAL_HOME/}" "run_bench.sh $name $*" <<'PY'
import csv, os, subprocess, sys
out, run_id, start, end, status, log, cmd = sys.argv[1:8]
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout
dirty = [l[3:] for l in git("status", "--porcelain", "-uno").splitlines()]
sha = git("rev-parse", "HEAD").strip() + (f" dirty({len(dirty)}: {' '.join(dirty)})" if dirty else " clean")
env = " ".join(f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith(("M3_", "BUDGET_", "PROFILE_", "BENCH_")))
new = not os.path.exists(out)
with open(out, "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow("run_id,block,start_utc,end_utc,sha,env,cmd,status,log,notes".split(","))
    w.writerow([run_id, "B_bench", start, end, sha, env, cmd, status, log, os.getenv("BENCH_NOTE", "")])
PY
  if [ "$STATUS" != OK ]; then
    rc_all=1
    env -u TT_VISIBLE_DEVICES tt-smi -glx_reset >> "$LOGS/$RUN_ID.reset" 2>&1
  fi
  if [ "$name" = experts ] && [ -s "$HERE/experts.csv" ]; then
    python3 "$HERE/fit_experts.py" "$HERE/experts.csv" --out "$HERE/experts_fit.txt" >> "$LOG" 2>&1 || true
  fi
done
exit $rc_all
