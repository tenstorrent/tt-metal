#!/bin/bash
# t449 (detached on g15blx02): waits until blx01's broker has no smarton job or fabric check running or queued,
# submits the attention PCC spec ($1, run409a.sh) via tt-project/t288/blx01-submit.sh (health check + device-lint),
# then waits for the job to end. Writes "T449A_DONE <rc> <job> <status>" last to $T449_LOG.
set -eo pipefail
P=/home/smarton/fasth3/tt-metal; SPEC=${1:?spec}; L=${T449_LOG:?log}; ST=none; id=none
trap 'rc=$?; echo "$(date -u +%FT%TZ) T449A_DONE $rc $id $ST" >> $L' EXIT
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
B1="ssh -o BatchMode=yes -o ConnectTimeout=20 g15blx01"
wait_clear() {
  local ok=0 i s
  for i in $(seq 240); do
    s=$($B1 "tt-device-mcp status 2>&1 | sed -n '/^RUNNING/,/^RECENT/p'" || true)
    if [ -n "$s" ] && ! echo "$s" | grep -qE 'smarton|HELD|hold|fabric-check'; then ok=$((ok+1)); else ok=0; fi
    [ $ok -ge 2 ] && return 0
    [ $((i % 15)) = 1 ] && log "waiting for blx01 to clear"
    sleep 60
  done
  return 1
}
rc=75
for i in $(seq 6); do
  wait_clear || { ST=not-clear; exit 4; }
  rc=0; out=$($P/tt-project/t288/blx01-submit.sh $SPEC 2>&1) || rc=$?
  echo "$out" >> $L; [ $rc = 75 ] || break; sleep 60
done
[ $rc = 0 ] || { ST=submit-rc$rc; exit 5; }
id=$(echo "$out" | sed -n 's/^job=//p'); log "job $id submitted"
for i in $(seq 180); do
  ST=$($B1 "tt-device-mcp status -j $id" 2>&1 | sed -n 's/^Status: *//p' | head -1 | tr A-Z a-z || true)
  case "$ST" in running|queued|pending|"") sleep 30 ;; *) break ;; esac
done
$B1 "tt-device-mcp status -j $id" >> $L 2>&1 || true
$B1 "tail -30 /var/tmp/fasth3/t409/out_attn/run.log" >> $L 2>&1 || true
[ "$ST" = completed ] || exit 6
