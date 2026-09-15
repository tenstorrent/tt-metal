#!/usr/bin/env bash
# The 2026-09-15 corrected PP=4 campaign: 2 topologies x 4 ISLs x {latency, throughput} = 16 cells.
#
# Differences from the 09-14 campaign (run_campaign.sh), all of them required by
# ~/debug-docs/pp4/HANDOFF_NEW_CAMPAIGN.md:
#
#   1. LATENCY CELLS ARE PROCESS-WARM. MODES=lat, i.e. PREFILL_PRODUCER_MAX_REQUESTS=2, and the
#      analyzer reports request 2. `ttft` (req=1) cannot be fixed by re-running: every run is a
#      fresh process that re-captures the trace, so the capture lands inside request 1 either way.
#      The fix is a second REQUEST, not a second run.
#   2. EVERY CELL RUNS TWICE AND THE SECOND IS REPORTED. This is a different effect from (1): the
#      JIT kernel cache is on disk and survives the process, so pass 1 populates it and pass 2
#      measures with it hot. Cold is WRONG, not slow -- a cold PP=4 25,600 latency cell once read
#      6.5 s against 1.17 s warm. Pass 1 is kept as `<tag>.pass1` so the delta is auditable.
#
#      *** THIS IS ALSO WHAT MAKES GATE 2 SATISFIABLE, and the two requirements are not
#      *** interchangeable. `JIT cache stats: N/N hits (100.0%)` is emitted once at process
#      *** teardown (build_cache_telemetry.cpp:268) and is cumulative for the PROCESS -- it reports
#      *** the state TT_METAL_CACHE was in when the process STARTED. Two requests in one process
#      *** (requirement 1) therefore cannot make it pass: request 1 warms the cache from inside the
#      *** same process that already reported 0%. Only a second RUN can, because pass 2 is a new
#      *** process starting against the cache pass 1 filled. So:
#      ***     requirement 1 (2 requests) puts trace capture outside the measured WINDOW
#      ***     requirement 2 (2 runs)     puts kernel compile outside the measured PROCESS
#      *** Session 1 hit this with a single-pass cell: 1rank 5,120 measured a clean 129.0 ms with
#      *** exact closure and still failed gate 2 at 0/1161 hits, because it was the first run of a
#      *** new code state. A throwaway pre-warm cell would also fix it, but it is unnecessary here:
#      *** pass 1 of each cell IS the pre-warm, per shape. Do not weaken gate 2 to avoid this --
#      *** it is the only check that catches a cell whose kernels were compiled inside the window,
#      *** and under this two-pass protocol it is satisfiable as written.
#   3. ISL 40,960 (8 chunks) replaces 102,400, to bracket the predicted break-even. The 09-14 grid
#      (1/5/20/51 chunks) jumps straight over it.
#   4. BOARD HEALTH IS STAMPED BEFORE AND AFTER EVERY CELL, into <tag>.board.{before,after}, so
#      gate 4 can be asserted afterwards instead of assumed. A degraded galaxy is silently 2-5x
#      slow rather than erroring (measured: 7.970 s vs 0.820 s on the same cell).
#
# Ordering is PAIR-MAJOR by ISL (1rank and pp4 adjacent, ISLs in prediction-value order), so that
# if the clock runs out what exists is COMPLETE PAIRS -- a speedup needs both halves.
#
# Never run this in a foreground tool call with a timeout: a killed wrapper is a SIGKILL mid-fabric
# and the next mesh open dies on an ethernet-core timeout.  setsid nohup ./run_campaign2.sh & and poll.
set -u
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../pipeline_prefill_harness/wrappers
HARNESS="$(cd "$S/.." && pwd)"
cd "$(cd "$HARNESS/../../../../../.." && pwd)" || exit 1    # repo root
source "$HARNESS/env.sh"
H=$(hostname)
RES="${RES:-$PWD/mistral4_pp4_campaign2_$H}"
mkdir -p "$RES"
ISLS_ORDER=${ISLS_ORDER:-"25600 40960 5120 261120"}
MODES_RUN=${MODES_RUN:-"lat thru"}
CFGS=${CFGS:-"1rank pp4"}
HARD_STOP=${HARD_STOP:-$(( $(date +%s) + 86400 ))}
left(){ echo $(( HARD_STOP - $(date +%s) )); }

stamp(){ # $1=tag $2=before|after
  if "$HARNESS/check_board.sh" >/dev/null 2>&1; then echo BOARD_OK > "$RES/$1.board.$2"
  else echo BOARD_BAD > "$RES/$1.board.$2"; fi
  echo "[c2] board $2 $1: $(cat "$RES/$1.board.$2")"
}

echo "[c2] START $(date -Is)  results=$RES"
echo "[c2] order: ISLs [$ISLS_ORDER] x cfgs [$CFGS] x modes [$MODES_RUN], two passes each"

for isl in $ISLS_ORDER; do
 for mode in $MODES_RUN; do
  for cfg in $CFGS; do
    tag="${cfg}_${isl}_${mode}"
    if [ -s "$RES/$tag/runner.log" ] && [ -z "${FORCE:-}" ]; then
      echo "[c2] skip $tag (pass-2 log exists; FORCE=1 to redo)"; continue
    fi
    # 1rank cells are the slow ones (~3.5 min each pass); two passes plus two board checks.
    need=900; [ "$cfg" = pp4 ] && need=600; [ "$isl" -ge 261120 ] && need=$(( need + 900 ))
    if [ "$(left)" -lt "$need" ]; then
      echo "[c2] STOP before $tag: $(( $(left)/60 ))m left, needs ~$(( need/60 ))m"; break 3
    fi

    stamp "$tag" before
    if [ "$(cat "$RES/$tag.board.before")" != BOARD_OK ]; then
      echo "[c2] ABORT: board unhealthy BEFORE $tag -- a number measured now would be silently 2-5x slow."
      echo "[c2] Run: tt-smi -glx_reset   (budget two) then re-run; completed cells are skipped."
      exit 2
    fi

    for pass in 1 2; do
      echo "[c2] === $tag pass $pass/2 ($(date -Is), $(( $(left)/60 ))m left)"
      OUT="$RES" CONFIGS="$cfg" ISLS="$isl" MODES="$mode" FORCE=1 "$HARNESS/run_matrix.sh" \
        >> "$RES/$tag.campaign.log" 2>&1
      rc=$?
      echo "[c2]     pass $pass rc=$rc"
      # Gate on the RUNNER log, not the producer's rc: run_pp4_model.sh exits with the producer's,
      # so a dead runner reports success and the cell would be recorded complete.
      nerr=$(grep -cE "AssertionError|Traceback \(most recent call last\)" "$RES/$tag/runner.log" 2>/dev/null | head -1)
      nerr=${nerr:-0}
      if [ "$rc" != "0" ] || [ "${nerr}" != "0" ]; then
        echo "[c2]     FAILED (rc=$rc runner-error-lines=$nerr); keeping log for PROVENANCE, not re-passing"
        mv "$RES/$tag" "$RES/$tag.FAILED.pass$pass" 2>/dev/null
        break
      fi
      # Keep pass 1 so the cold-vs-warm delta stays auditable; pass 2 keeps the plain tag.
      [ "$pass" = 1 ] && { rm -rf "$RES/$tag.pass1"; mv "$RES/$tag" "$RES/$tag.pass1"; }
    done
    stamp "$tag" after
  done
 done
done
echo "[c2] END $(date -Is)"
echo "[c2] now assert the gates:  python3 ~/debug-docs/pp4/scripts/gate_matrix.py $RES"
