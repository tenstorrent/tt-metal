#!/bin/bash
# drv363b: A/B-only rerun after job 468 failed in the test's latent loader (seed0.pt is a tensor, not a dict).
# t363 driver on blx01 (no device work itself): (1) build t363/b = reflink of t48 at ba3636df5ee + t363 test files;
# (2) one broker sweep job per layer (s2_res s3_res s3_chg s1_res), T=2/4 blockings that fit L1, -t 600 each
# (unmeasured), rerun once on a drop; (3) pick winners (>= 3% faster); (4) if any, one A/B full-decode job;
# (5) delete jit and b. Marker t363/drv363.done.
set -o pipefail
F=/var/tmp/fasth3; D=$F/t363; B=$D/b; M=$D/drv363b.done; L=$D/drv363b.log; R=$D/res
REV=ba3636df5ee
trap 'rc=$?; echo "exit=$rc $(date -u +%T)" >> $L; [ -e $M ] || echo "DRIVER_EXIT rc=$rc" > $M' EXIT
log() { echo "$(date -u '+%F %T') $*" >> $L; }
fail() { log "FAIL $*"; rm -rf $D/jit $B; echo "FAIL $*" > $M; exit 1; }
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp
mkdir -p $TMPDIR $R
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); log "start df / $use%"; [ "$use" -le 70 ] || fail "df / $use%"
gb=$(timeout 120 du -sxBG $F | cut -f1 | tr -dc 0-9); log "footprint ${gb}G"; [ "${gb:-999}" -le 138 ] || fail "footprint ${gb}G"

# (1) build
if [ ! -r $B/ttnn/ttnn/_ttnn.so ]; then
  rm -rf $B
  cp -a --reflink=always $F/t48 $B || fail "copy rc=$?"
  rm -rf $B/build $B/build_Release $B/python_env $B/tmp
  ( cd $B && git checkout -q -f $REV && git submodule update --init --recursive -q && git clean -fdq ) >> $L 2>&1 || fail "checkout rc=$?"
  ( cd $B && timeout 5400 ./build_metal.sh --build-type Release ) > $D/build.log 2>&1; rc=$?
  log "build rc=$rc"; [ $rc = 0 ] || { tail -40 $D/build.log > $R/build_tail.log; fail "build rc=$rc"; }
fi
cp $D/bruteforce_conv3d_sweep_ltx.py $D/test_vae_ltx_blk_ab_4x8.py $B/models/tt_dit/tests/models/ltx/ || fail "test copy"
log "checkout $(git -C $B log -1 --format='%h %s' | cut -c1-80) dirty=$(git -C $B status --porcelain | tr '\n' ' ')"
[ -r $B/ttnn/ttnn/_ttnn.so ] || fail "missing _ttnn.so"

# (2) device jobs
st() { tt-device-mcp status -j $1 2>&1 | awk '/^Status:/{print $2}'; }
waitjob() { for i in $(seq 1 240); do s=$(st $1); case $s in running|queued|pending|"") sleep 30;; *) echo $s; return;; esac; done; echo stillrunning; }
submit() {  # args: the run363.sh arguments
  for i in $(seq 1 720); do
    out=$(tt-device-mcp status 2>&1)
    run=$(echo "$out" | sed -n '/^RUNNING/,/^QUEUED/p'); q=$(echo "$out" | sed -n '/^QUEUED/,/^RECENT/p')
    if echo "$out" | grep -qi upgrade || echo "$run$q" | grep -qiE 'hold|health|reset|fabric-check|smarton'; then sleep 30; continue; fi
    bash $D/lint.sh --device --env $D/env363.yaml --timeout 600 t363 bash $D/run363.sh >> $L 2>&1 || { log "lint refused"; return; }
    sub=$(tt-device-mcp run-bg "bash $D/run363.sh $(printf '%q ' "$@")" -w $D -e $D/env363.yaml -t 600 2>&1)
    echo "$sub" >> $L; echo "$sub" | grep -oE 'Job [0-9]+' | head -1 | grep -oE '[0-9]+'; return
  done
}
job() {  # name, run363.sh args; sets JOK
  local name=$1; shift; JOK=0
  for att in 1 2; do
    rm -f $D/run.log
    J=$(submit "$@"); [ -n "$J" ] || { log "$name notsubmitted"; echo "$name notsubmitted" >> $D/jobs.txt; return; }
    log "$name attempt=$att job=$J"
    s=$(waitjob $J); log "$name job=$J status=$s"
    cp $D/run.log $R/run_${name}_job$J.log 2>/dev/null
    sleep 10; log "job=$J leftover: $(ps -u $(id -u) -o pid=,args= | grep -E 'pytest|run363' | grep -v grep | tr '\n' ';')"
    echo "$name job=$J status=$s exit=$(grep -oE '^T363_EXIT=[0-9]+' $D/run.log 2>/dev/null)" >> $D/jobs.txt
    grep -q '^T363_EXIT=0' $D/run.log 2>/dev/null && { JOK=1; return; }
    grep -q '^T363_EXIT=' $D/run.log 2>/dev/null && return
    log "$name job=$J DROP? status=$s (no exit line) $(date -u +%FT%TZ); waiting, then rerun"; echo "$name DROP job=$J $(date -u +%FT%TZ)" >> $D/jobs.txt; sleep 240
  done
}
CAND="64,256,2,4,4;64,256,2,2,8;64,256,2,8,2;32,256,2,8,4;32,256,2,4,8;32,256,4,8,4;32,256,2,8,8;64,128,2,8,4;64,128,2,8,8;64,128,4,8,4;64,128,2,4,8;32,256,4,4,8"
: # sweeps done (jobs 457/462/463/465); drv363b reruns only the A/B

# (3) winners, (4) A/B
source $F/t48/python_env/bin/activate
NEW=$(cat $R/winners.txt); log "winners (from sweep): $NEW"
AB=skip
if [ -n "$NEW" ]; then job ab ab "$NEW"; AB=$JOK; fi

# (5) cleanup
cp $D/build.log $R/build_b.log 2>/dev/null && gzip -f $R/build_b.log
rm -rf $D/jit $B; log "cleaned jit b; df / $(df --output=pcent / | tail -1 | tr -dc 0-9)%"
echo "DONE ab=$AB $(tr '\n' ' ' < $D/jobs.txt)" > $M
