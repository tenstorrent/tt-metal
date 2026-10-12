#!/bin/bash
# t416 (detached on g15blx02): waits for the blx01 VSA build (build409), then submits the attention PCC job
# (spec attn.txt, run409a.sh) through tt-project/t288/blx01-submit.sh, which refuses while another smarton job
# runs there (the t408 chain), and waits for it to end. A job that does not complete is retried once after the
# broker is clear. Writes "T416A_DONE <rc> <final status>" last to $L.
set -eo pipefail
P=/home/smarton/fasth3/tt-metal; D=$P/tt-project/worktrees/t416/tt-project/t409
L=${T416_LOG:?log}; ST=none
trap 'rc=$?; echo "$(date -u +%FT%TZ) T416A_DONE $rc $ST" >> $L' EXIT
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
B1="ssh -o BatchMode=yes -o ConnectTimeout=20 g15blx01"
for i in $(seq 120); do $B1 test -e /var/tmp/fasth3/t409/build409.rc && break; sleep 60; done
m=$($B1 cat /var/tmp/fasth3/t409/build409.marker 2>&1 || true); log "build: $m"
case "$m" in "rc=0 step=done"*) ;; *) ST=build-failed; exit 3;; esac
scp -q $D/run409a.sh g15blx01:/var/tmp/fasth3/t409/run409a.sh.tmp
$B1 mv /var/tmp/fasth3/t409/run409a.sh.tmp /var/tmp/fasth3/t409/run409a.sh
wait_clear() {
  local ok=0 i s
  for i in $(seq 120); do
    s=$($B1 "tt-device-mcp status 2>&1 | sed -n '/^RUNNING/,/^QUEUED/p'" || true)
    if [ -n "$s" ] && ! echo "$s" | grep -qE 'smarton|broker|HELD|hold|fabric-check'; then ok=$((ok+1)); else ok=0; fi
    [ $ok -ge 2 ] && return 0; sleep 60
  done
  return 1
}
for try in 1 2; do
  wait_clear || { ST=not-clear; exit 4; }
  rc=0
  for i in $(seq 60); do
    rc=0; out=$($P/tt-project/t288/blx01-submit.sh $D/attn.txt 2>&1) || rc=$?
    echo "$out" >> $L; [ $rc = 75 ] || break; sleep 120
  done
  [ $rc = 0 ] || { ST=submit-rc$rc; exit 5; }
  id=$(echo "$out" | sed -n 's/^job=//p'); log "attn try $try: job $id submitted"
  for i in $(seq 120); do
    ST=$($B1 "tt-device-mcp status -j $id" 2>&1 | sed -n 's/^Status: *//p' | head -1 | tr A-Z a-z)
    case "$ST" in running|queued|pending|"") sleep 30 ;; *) break ;; esac
  done
  $B1 "tt-device-mcp status -j $id" >> $L 2>&1 || true
  $B1 "tail -30 /var/tmp/fasth3/t409/out_attn/run.log" >> $L 2>&1 || true
  log "attn try $try: job $id status $ST"
  [ "$ST" = completed ] && exit 0
  $B1 test -e /var/tmp/fasth3/t409/out_attn/PASS && exit 0
  # A test failure (exit from pytest) is a result, not a drop: do not retry it.
  $B1 "grep -qE 'T409_EXIT=1$' /var/tmp/fasth3/t409/out_attn/run.log" && exit 6
done
exit 7
