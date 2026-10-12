#!/bin/bash
# t421 chain (detached on g15blx02): submits the blx01 specs in order, one job at a time through
# tt-project/t288/blx01-submit.sh (refuses while another smarton job runs there). A job that does not
# complete is retried once after the broker is clear; a second failure skips that spec. A failed
# on-cache skips every on arm; three skipped specs in a row stop the chain. Then runs analyze421.py.
# Writes T421_DONE <rc> last to tt-project/state/t421/chain.log (untracked).
set -eo pipefail
cd /home/smarton/fasth3/tt-metal
D=tt-project/state/t421; L=$D/chain.log
trap 'rc=$?; echo "$(date -u +%FT%TZ) T421_DONE $rc" >> $L' EXIT
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
B1="ssh -o BatchMode=yes -o ConnectTimeout=20 g15blx01"
wait_clear() {
  local ok=0 i s
  for i in $(seq 120); do
    s=$($B1 "tt-device-mcp status 2>&1 | sed -n '/^RUNNING/,/^QUEUED/p'" || true)
    if [ -n "$s" ] && ! echo "$s" | grep -qE 'broker|HELD|hold|fabric-check'; then ok=$((ok+1)); else ok=0; fi
    [ $ok -ge 2 ] && return 0; sleep 60
  done
  return 1
}
# One attempt: prints the final broker status of the job (or submit-fail / not-clear).
attempt() {
  local spec=$1 out rc id st i
  wait_clear || { echo not-clear; return; }
  for i in $(seq 45); do
    rc=0; out=$(tt-project/t288/blx01-submit.sh $D/specs/$spec.txt 2>&1) || rc=$?
    echo "$out" >> $L; [ $rc = 75 ] || break; sleep 120
  done
  [ $rc = 0 ] || { echo "submit-rc$rc"; return; }
  id=$(echo "$out" | sed -n 's/^job=//p'); log "$spec: job $id submitted"
  for i in $(seq 240); do
    st=$($B1 "tt-device-mcp status -j $id" 2>&1 | sed -n 's/^Status: *//p' | head -1 | tr A-Z a-z)
    case "$st" in running|queued|pending|"") sleep 30 ;; *) break ;; esac
  done
  $B1 "tt-device-mcp status -j $id" >> $L 2>&1 || true
  log "$spec: job $id status: $st"
  echo "$st"
}
skip_on=0; streak=0
for spec in "$@"; do
  if [ $skip_on = 1 ] && [ "${spec#on-s}" != "$spec" ]; then log "$spec: SKIP (on-cache failed)"; continue; fi
  st=""
  for try in 1 2; do
    st=$(attempt $spec | tail -1)
    if [ "$st" = completed ]; then break; fi
    log "$spec: try $try ended '$st'"
    $B1 "tail -20 /var/tmp/fasth3/t421/out_$spec/run.log" >> $L 2>&1 || true
  done
  if [ "$st" = completed ]; then streak=0; else
    log "$spec: SKIP after 2 tries"; streak=$((streak+1))
    if [ $spec = on-cache ]; then skip_on=1; fi
    if [ $streak -ge 3 ]; then log "3 specs skipped in a row: stop"; exit 8; fi
  fi
done
log "analysis"
timeout 1200 python_env/bin/python $D/analyze421.py >> $L 2>&1
