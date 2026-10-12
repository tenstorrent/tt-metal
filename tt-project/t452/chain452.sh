#!/bin/bash
# t452 (detached on g15blx02): submits each spec given (one Turbo arm per blx01 broker job, run452.sh) in order
# through tt-project/t288/blx01-submit.sh (health check + device-lint, refuses while another smarton job is there)
# and waits for each to end. A job that ends without PASS and without a pytest result (a drop) is resubmitted once.
# Logs "<spec> job <id> <status> pass|fail" per spec and "T452_DONE <rc> <failed specs>" last to $T452_LOG.
set -eo pipefail
P=/home/smarton/fasth3/tt-metal; L=${T452_LOG:?log}; FAILED=""
trap 'rc=$?; echo "$(date -u +%FT%TZ) T452_DONE $rc ${FAILED:-none}" >> $L' EXIT
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
B1="ssh -o BatchMode=yes -o ConnectTimeout=20 g15blx01"
wait_clear() {
  local ok=0 i s
  for i in $(seq 120); do
    s=$($B1 "tt-device-mcp status 2>&1 | sed -n '/^RUNNING/,/^RECENT/p'" || true)
    if [ -n "$s" ] && ! echo "$s" | grep -qE 'smarton|HELD|hold|fabric-check'; then ok=$((ok+1)); else ok=0; fi
    [ $ok -ge 2 ] && return 0; sleep 60
  done
  return 1
}
run_spec() {  # <spec file>; returns 0 on PASS, 6 on a test failure, else non-zero
  local S=$1 out id ST rc i tag arm O
  tag=$(sed -n 's/^CMD=.*run452.sh \([^ ]*\) \([^ ]*\)$/\1-\2/p' $S); O=/var/tmp/fasth3/t452/out_$tag
  $B1 test -e $O/PASS && { log "$S already passed"; return 0; }
  wait_clear || { log "$S: blx01 not clear"; return 4; }
  for i in $(seq 60); do
    rc=0; out=$($P/tt-project/t288/blx01-submit.sh $S 2>&1) || rc=$?
    echo "$out" >> $L; [ $rc = 75 ] || break; sleep 120
  done
  [ $rc = 0 ] || { log "$S: submit rc $rc"; return 5; }
  id=$(echo "$out" | sed -n 's/^job=//p'); log "$S: job $id submitted"
  for i in $(seq 120); do
    ST=$($B1 "tt-device-mcp status -j $id" 2>&1 | sed -n 's/^Status: *//p' | head -1 | tr A-Z a-z || true)
    case "$ST" in running|queued|pending|"") sleep 30 ;; *) break ;; esac
  done
  $B1 "tt-device-mcp status -j $id" >> $L 2>&1 || true
  $B1 "tail -25 $O/run.log" >> $L 2>&1 || true
  if $B1 test -e $O/PASS; then log "$S job $id $ST pass"; return 0; fi
  log "$S job $id $ST fail"
  $B1 "grep -qE 'T409_EXIT=1$' $O/run.log" && return 6
  return 7
}
for S in "$@"; do
  rc=0; run_spec $S || rc=$?
  if [ $rc = 7 ] || [ $rc = 5 ]; then log "$S: retrying once (drop or submit refusal)"; rc=0; run_spec $S || rc=$?; fi
  [ $rc = 0 ] || FAILED="$FAILED $(basename $S)"
done
[ -z "$FAILED" ]
