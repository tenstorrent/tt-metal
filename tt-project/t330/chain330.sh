#!/bin/bash
# t330 chain (g15blx02): wait for blx01 job $1 (t2va), take stills and frame 0 as the fl2va keyframe,
# submit the fl2va spec, wait for it, take its stills. Writes chain.log; last line T330_CHAIN_DONE <rc>.
set -eo pipefail
J1=${1:?t2va job id}
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
D=tt-project/t330; L=$D/chain.log; mkdir -p $D/stills
trap 'rc=$?; echo "T330_CHAIN_DONE $rc $(date -u +%T)" >> $L' EXIT
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 g15blx01"
O=/var/tmp/fasth3/t330
log() { echo "[chain $(date -u +%T)] $*" >> $L; }

wait_job() {  # up to 3 h, queued behind other tenants
  for _ in $(seq 360); do
    st=$($SSH "tt-device-mcp status -j $1" 2>/dev/null | sed -n 's/^Status: *//p' | head -1) || st=
    case "$st" in ""|running*|queued*|pending*|started*) ;; *) log "job $1: $st"; return 0;; esac
    sleep 30
  done
  log "job $1 still not ended after 3 h"; return 3
}

stills() {  # $1 tag
  mp4=$($SSH "ls $O/out_$1/*.mp4" | head -1)
  [ -n "$mp4" ] || { log "no mp4 for $1"; return 4; }
  $SSH "cd $O/out_$1 && for t in 0 2.5 4.5; do ffmpeg -loglevel error -y -ss \$t -i $mp4 -frames:v 1 -q:v 3 still_\$t.jpg; done"
  scp -q "g15blx01:$O/out_$1/still_*.jpg" $D/stills/ 2>>$L
  for f in $D/stills/still_*.jpg; do case $f in *_t2va_*|*_fl2va_*) ;; *) mv $f ${f/still_/still_$1_};; esac; done
  scp -q "g15blx01:$O/out_$1/run.log" $D/run_$1.log
  log "stills for $1 from $mp4"
}

log "waiting for t2va job $J1"
wait_job $J1
$SSH "test -e $O/out_t2va/PASS" || { log "t2va did not pass"; scp -q g15blx01:$O/out_t2va/run.log $D/run_t2va.log || true; exit 5; }
stills t2va
mp4=$($SSH "ls $O/out_t2va/*.mp4" | head -1)
$SSH "ffmpeg -loglevel error -y -i $mp4 -vf 'select=eq(n\,0)' -frames:v 1 $O/kf_t2va_f0.png && ls -l $O/kf_t2va_f0.png" >> $L
for _ in $(seq 120); do  # another smarton job (#288) may hold blx01: retry, never run alongside
  set +e; out=$(tt-project/t288/blx01-submit.sh $D/specs/blx01/fl2va.txt 2>&1); rc=$?; set -e
  echo "$out" >> $L
  [ $rc = 75 ] || break
  sleep 60
done
[ $rc = 0 ] || { log "fl2va submit rc $rc"; exit $rc; }
J2=$(echo "$out" | sed -n 's/^job=//p')
wait_job $J2
$SSH "test -e $O/out_fl2va/PASS" || { log "fl2va did not pass"; scp -q g15blx01:$O/out_fl2va/run.log $D/run_fl2va.log || true; exit 6; }
stills fl2va
