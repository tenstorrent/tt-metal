#!/bin/bash
# t460 chain (detached on g15blx02): the base/int arms in specs/ order as adjacent blx01 broker jobs (same box,
# same clock), one at a time through tt-project/t288/blx01-submit.sh (refuses with 75 while another smarton job
# runs). One arm per job: a 6 s arm is ~260-330 s and a 10 s arm over 400 s, so a pair at +50% exceeds 600 s.
# A tag whose out dir has PASS is skipped (safe to rerun). Each job gets 2 tries; a non-completed try logs the
# broker status and the run.log tail (drop record). Writes T460_DONE <rc> last to $D/chain.log.
set -eo pipefail
cd /home/smarton/fasth3/tt-metal
S=${T460_SPECS:?dir of spec files}; D=${T460_STATE:?state dir}; L=$D/chain.log
mkdir -p $D
trap 'rc=$?; echo "$(date -u +%FT%TZ) T460_DONE $rc" >> $L' EXIT
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
B1="ssh -o BatchMode=yes -o ConnectTimeout=20 g15blx01"
attempt() {
  local spec=$1 out rc id st i
  for i in $(seq 240); do
    rc=0; out=$(tt-project/t288/blx01-submit.sh $spec 2>&1) || rc=$?
    echo "$out" >> $L
    # 75: another smarton job holds the queue. A broker hold, failed health check or unreachable box also waits.
    [ $rc = 75 ] || echo "$out" | grep -qiE 'broker holds the device|broker inactive|unreachable|not healthy|health' || break
    sleep 120
  done
  [ $rc = 0 ] || { echo "submit-rc$rc"; return; }
  id=$(echo "$out" | sed -n 's/^job=//p'); log "$(basename $spec): job $id submitted"
  for i in $(seq 480); do
    st=$($B1 "tt-device-mcp status -j $id" 2>&1 | sed -n 's/^Status: *//p' | head -1 | tr A-Z a-z) || true
    case "$st" in running|queued|pending|"") sleep 30 ;; *) break ;; esac
  done
  $B1 "tt-device-mcp status -j $id" >> $L 2>&1 || true
  log "$(basename $spec): job $id status: $st"
  echo "$st"
}
rc=0
for spec in $S/*.txt; do
  tag=$(sed -n 's/^ID=t460-//p' $spec)
  $B1 "test -e /var/tmp/fasth3/t460/out_$tag/PASS" && { log "$tag: already passed"; continue; }
  ok=0
  for try in 1 2; do
    st=$(attempt $spec | tail -1)
    if [ "$st" = completed ] && $B1 "test -e /var/tmp/fasth3/t460/out_$tag/PASS"; then ok=1; log "$tag: completed"; break; fi
    log "$tag: try $try ended '$st' (drop/failure record follows)"
    $B1 "tail -20 /var/tmp/fasth3/t460/out_$tag/run.log; tt-device-mcp status 2>&1 | head -15" >> $L 2>&1 || true
  done
  [ $ok = 1 ] || { log "$tag: FAILED after 2 tries"; rc=1; }
done
$B1 "grep -H 'Denoising (per step)\|Total Pipeline\|VAE\|process wall\|mp4\|T408_EXIT' /var/tmp/fasth3/t460/out_*/run.log" >> $L 2>&1 || true
exit $rc
