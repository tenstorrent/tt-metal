#!/bin/bash
# t422 chain (detached on g15blx02): wait for the blx01 build (setup422-blx01.sh), then the off arm and the on arm
# as adjacent blx01 jobs (same box, same clock), one at a time through tt-project/t288/blx01-submit.sh.
# A pair does not fit one job: one 5 s arm is ~250-330 s, so a pair at +50% is ~750-1000 s against the 600 s cap.
# Each job gets 2 tries. Writes T422_DONE <rc> last to $D/chain.log.
set -eo pipefail
cd /home/smarton/fasth3/tt-metal
S=${T422_SPECS:?dir of spec files}; D=${T422_STATE:?state dir}; L=$D/chain.log
mkdir -p $D/specs
trap 'rc=$?; echo "$(date -u +%FT%TZ) T422_DONE $rc" >> $L' EXIT
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
attempt() {
  local spec=$1 out rc id st i
  wait_clear || { echo not-clear; return; }
  for i in $(seq 180); do
    rc=0; out=$(tt-project/t288/blx01-submit.sh $spec 2>&1) || rc=$?
    echo "$out" >> $L; [ $rc = 75 ] || break; sleep 120
  done
  [ $rc = 0 ] || { echo "submit-rc$rc"; return; }
  id=$(echo "$out" | sed -n 's/^job=//p'); log "$(basename $spec): job $id submitted"
  for i in $(seq 240); do
    st=$($B1 "tt-device-mcp status -j $id" 2>&1 | sed -n 's/^Status: *//p' | head -1 | tr A-Z a-z)
    case "$st" in running|queued|pending|"") sleep 30 ;; *) break ;; esac
  done
  $B1 "tt-device-mcp status -j $id" >> $L 2>&1 || true
  log "$(basename $spec): job $id status: $st"
  echo "$st"
}
run_spec() {
  local spec=$1 tag=$2 st try
  for try in 1 2; do
    st=$(attempt $spec | tail -1)
    [ "$st" = completed ] && { log "$tag: completed"; return 0; }
    log "$tag: try $try ended '$st'"
    $B1 "tail -20 /var/tmp/fasth3/t422/out_$tag/run.log" >> $L 2>&1 || true
  done
  log "$tag: FAILED after 2 tries"; return 1
}
for i in $(seq 240); do
  m=$($B1 "cat /var/tmp/fasth3/t422/setup.marker 2>/dev/null" || true)
  [ -n "$m" ] && break; sleep 60
done
log "setup: '$m'"
case "$m" in "rc=0 step=done"*) ;; *) exit 3 ;; esac
rc=0
run_spec $S/off.txt off || rc=1
run_spec $S/on.txt on || rc=1
$B1 "grep -h 'Denoising (per step)\|Total Pipeline\|process wall\|mp4' /var/tmp/fasth3/t422/out_off/run.log /var/tmp/fasth3/t422/out_on/run.log" >> $L 2>&1 || true
exit $rc
