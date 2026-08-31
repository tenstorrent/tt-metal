#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Does pinning the tunnel fix REMOTE aggregator launch reliability?
#
# The launch writes ~25 KB to the target chip. On a remote chip that crosses the NON_MMIO
# tunnel, which UMD defaults to every active eth channel on the MMIO chip -- six on a T3K,
# four of which reach other boards entirely (5h). Remote launches fail intermittently as a
# result.
#
# A SINGLE RUN CANNOT ANSWER THIS. Observed unpinned remote successes across four earlier
# runs were 4/4, 3/4, 2/4 and 1/4 -- the variance is larger than any plausible effect. So
# this repeats the launch step many times inside ONE workload session and interleaves the
# two arms, which also removes workload startup and thermal drift as confounds.
#
# Each rep must stop the aggregators before the next: launching onto a core that already
# holds a live kernel does NOT replace it, it corrupts it (agg_core_select.hpp).
#
# Usage: bash launch_ab.sh --artifact <dir> [--reps 6]

set -uo pipefail
TT_METAL_HOME="${TT_METAL_HOME:-$(pwd)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTOR="${COLLECTOR:-${TT_METAL_HOME}/build_Release/tools/ttnvtop-collector}"
WORKLOAD_PY="${WORKLOAD_PY:-${SCRIPT_DIR}/ab_workload.py}"

ARTIFACT=""; REPS=6; MESH_ROWS=2; MESH_COLS=4; SIZE=2048
while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifact) ARTIFACT="$2"; shift 2 ;;
    --reps)     REPS="$2"; shift 2 ;;
    --rows)     MESH_ROWS="$2"; shift 2 ;;
    --cols)     MESH_COLS="$2"; shift 2 ;;
    -h|--help)  sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -f "$ARTIFACT/aggregator.desc" ]] || { echo "FATAL: no artifact at $ARTIFACT" >&2; exit 2; }

TS="$(date +%Y%m%d-%H%M%S)"
OUT="/tmp/launch-ab-${TS}"   # peer-LOCAL, so an mpi-shfs blip cannot take the results
mkdir -p "$OUT"
echo "=== remote launch reliability: pinned vs unpinned ==="
echo "out  : ${OUT}   (local disk, not the share)"
echo "reps : ${REPS} per arm, interleaved"
echo

WL_PID=""
cleanup() {
  "$COLLECTOR" --stop-aggregator >/dev/null 2>&1
  # Do NOT signal the workload. Let it finish its own --seconds budget and exit through
  # ttnn's `finally`.
  #
  # SIGINT is not reliably prompt here: Python cannot deliver KeyboardInterrupt while it
  # is inside a ttnn C++ call, so the handler can land mid-teardown. Every time this
  # script signalled the workload, the NEXT device open died with a frozen fabric
  # heartbeat ("Stuck at 0xaabb...."), costing a board reset -- 0xAABB is
  # FABRIC_HEARTBEAT_SIGNATURE, i.e. fabric firmware stopped mid-loop on an ACTIVE eth
  # core, and UMD throws on a valid-but-frozen signature. Waiting costs a couple of
  # minutes; signalling costs a reset and invalidates the run.
  if [[ -n "$WL_PID" ]] && kill -0 "$WL_PID" 2>/dev/null; then
    echo "  waiting for the workload to finish its budget (never signalled -- see comment)"
    wait "$WL_PID" 2>/dev/null
  fi
  return 0
}
trap cleanup EXIT INT TERM

# Workload holds the device for the whole session: an aggregator can only be STARTED while
# a tt-metal process has the device (5r), and it dies when that process closes.
WL_SECS=$(( REPS * 2 * 40 + 150 ))
python3 "$WORKLOAD_PY" --rows "$MESH_ROWS" --cols "$MESH_COLS" --size "$SIZE" \
    --seconds "$WL_SECS" --warmup 200 --label launch-ab > "$OUT/workload.log" 2>&1 &
WL_PID=$!
for i in $(seq 1 240); do
  grep -q "Config{" "$OUT/workload.log" 2>/dev/null && break
  kill -0 "$WL_PID" 2>/dev/null || { echo "FATAL: workload died in startup" >&2; exit 1; }
  sleep 1
done
sleep 20
kill -0 "$WL_PID" 2>/dev/null || { echo "FATAL: workload died before reps" >&2; exit 1; }
echo "workload up (pid $WL_PID), ${WL_SECS}s budget"
echo

run_arm() {  # $1 = arm (plain|pinned), $2 = rep
  local arm="$1" rep="$2" pin=""
  [[ "$arm" == "pinned" ]] && pin="--pin-tunnel"
  local log="$OUT/${arm}-rep${rep}.log"
  "$COLLECTOR" --launch-aggregator "$ARTIFACT" $pin > "$log" 2>&1
  # Per-chip verdicts. The launcher prints "— RUNNING (sweeps a -> b)" only on success;
  # "NOT RUNNING" contains "RUNNING" so the em-dash prefix is load-bearing.
  local rem_ok rem_bad mmio_ok
  # Count DISTINCT CHIPS, not lines: the launcher now falls back through candidate eth
  # cores, so one chip can log several NOT RUNNING attempts before a RUNNING one.
  rem_ok=$(grep -- "(remote).*RUNNING (sweeps" "$log" | grep -v "NOT RUNNING" \
             | grep -oE "chip [0-9]+" | sort -u | wc -l)
  mmio_ok=$(grep -- "(mmio).*RUNNING (sweeps" "$log" | grep -v "NOT RUNNING" \
             | grep -oE "chip [0-9]+" | sort -u | wc -l)
  rem_tot=$(grep -- "(remote)" "$log" | grep -oE "chip [0-9]+" | sort -u | wc -l)
  rem_bad=$(( rem_tot - rem_ok ))
  echo "${arm} ${rep} ${rem_ok} ${rem_bad} ${mmio_ok}" >> "$OUT/results.txt"
  printf "  %-7s rep%-2s remote %s/%s ok   mmio %s/4 ok\n" \
      "$arm" "$rep" "$rem_ok" "$((rem_ok+rem_bad))" "$mmio_ok"
  "$COLLECTOR" --stop-aggregator > "$OUT/${arm}-rep${rep}.stop.log" 2>&1
  sleep 2
}

for ((r=1; r<=REPS; r++)); do
  # Alternate which arm goes first so a monotonic drift cannot load onto one arm.
  if (( r % 2 == 1 )); then
    run_arm plain "$r";  run_arm pinned "$r"
  else
    run_arm pinned "$r"; run_arm plain "$r"
  fi
done

echo
echo "=== summary ==="
awk '{ok[$1]+=$3; tot[$1]+=$3+$4; mm[$1]+=$5; n[$1]++}
     END{ printf "%-8s %-18s %-14s %s\n","arm","remote launched","remote rate","mmio launched";
          for (a in ok) printf "%-8s %-18s %-14s %s\n", a, ok[a]"/"tot[a],
               (tot[a]?sprintf("%.0f%%",100*ok[a]/tot[a]):"-"), mm[a]"/"(4*n[a]) }' "$OUT/results.txt"
echo
echo "artifacts: ${OUT}"
