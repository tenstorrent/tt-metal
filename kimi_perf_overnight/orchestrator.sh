#!/usr/bin/env bash
# Kimi prefill runner-vs-test perf overnight orchestrator.
#
# Runs a QUEUE of experiment definitions (queue/*.exp) one at a time on the shared 8x4 mesh,
# appends results to RESULTS.md, and moves each finished .exp into done/. When the queue drains
# it IDLES (rescans every 30s) so a live session can drop in more experiments. `touch STOP` ends it.
#
# Self-contained: no Claude in the loop. If the controlling session dies, this keeps running in tmux
# and every result lands durably in RESULTS.md + logs/. See RECOVERY.md.
set -u

INVDIR=/home/ppopovic/kimi_perf_overnight
source "$INVDIR/lib.sh"

source /home/ppopovic/tt-metal/python_env/bin/activate

trap 'log "orchestrator trap: cleaning up child procs"; cleanup_procs' EXIT INT TERM

log "######## orchestrator START (pid=$$) ########"
[ -f "$RESULTS" ] || cat > "$RESULTS" <<EOF
# Kimi prefill runner-vs-test perf — overnight results

Baseline (durably known): runner ~3.3 s/chunk @ MAX_SEQ_LEN=61440; no-PCC test ~1.94 s/chunk.
Gap ~1.4 s constant/additive per chunk. See RECOVERY.md for the question and method.

Results appended below as each experiment completes.
EOF

idle_announced=0
while [ ! -f "$INVDIR/STOP" ]; do
  # Re-scan and re-sort EVERY iteration, then process only the FIRST file. This way a lower-numbered
  # .exp dropped in by a live session jumps ahead of higher-numbered pending ones.
  # NOTE: do NOT use nullglob here — with it, an empty queue makes `ls "$QUEUE"/*.exp` collapse to a
  # bare `ls` (listing cwd) and pick a garbage file. find avoids the glob entirely.
  f=$(find "$QUEUE" -maxdepth 1 -name '*.exp' 2>/dev/null | sort | head -1)
  if [ -z "$f" ]; then
    [ "$idle_announced" -eq 0 ] && { log "queue empty; idling (drop *.exp into $QUEUE, or 'touch $INVDIR/STOP')"; idle_announced=1; }
    sleep 30; continue
  fi
  idle_announced=0
  unset EXP_NAME EXP_TYPE EXP_DESC RUNNER_ENV PRODUCER_ENV TEST_ENV
  RUNNER_ENV=""; PRODUCER_ENV=""; TEST_ENV=""
  # shellcheck disable=SC1090
  source "$f"
  log ">>> processing $(basename "$f"): $EXP_NAME [$EXP_TYPE]"
  { echo; echo "---"; echo "## $EXP_NAME — $EXP_DESC"; } >> "$RESULTS"
  case "$EXP_TYPE" in
    runner)     run_runner "$EXP_NAME" "$RUNNER_ENV" "$PRODUCER_ENV" ;;
    test)       run_test "$EXP_NAME" "$TEST_ENV" ;;
    standalone) run_standalone "$EXP_NAME" "$RUNNER_ENV" ;;
    *)          log "!!! unknown EXP_TYPE='$EXP_TYPE' in $f; skipping"; record_fail "$EXP_NAME" "unknown EXP_TYPE" ;;
  esac
  mv "$f" "$DONE/"
  log "<<< done $EXP_NAME; moved to done/"
done
log "######## STOP sentinel found; orchestrator EXIT ########"
