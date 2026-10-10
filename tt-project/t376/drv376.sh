#!/bin/bash
# t376 driver on blx01 (no device work itself): (1) build t376/b = reflink of t48 at the t48 tip 90ed8257bac
# + the t376 sweep file and A/B test; (2) one broker halo sweep job per layer (s4res_x s1up_x s0res_x s0up_x
# s4out_x: res128, up_all_x1, res1024, up_all/2, conv_out), 40 candidates each with the table C_in_block,
# -t 600 each (unmeasured), rerun once on a drop; (3) pick md5-identical winners >= 3% faster; (4) if any, one A/B
# full-decode job (old vs new, 3 warm replays; rerun once if it times out on a cold JIT); (5) delete jit and b.
# Submitted only when the broker shows no upgrade/hold/health/reset/fabric-check job and no other smarton job.
# Marker t376/drv376.done.
set -o pipefail
F=/var/tmp/fasth3; D=$F/t376; B=$D/b; M=$D/drv376.done; L=$D/drv376.log; R=$D/res
REV=90ed8257bac  # ttp/t48-ltx25-integrated tip: the A/B runs on the land target
trap 'rc=$?; echo "exit=$rc $(date -u +%T)" >> $L; [ -e $M ] || echo "DRIVER_EXIT rc=$rc" > $M' EXIT
log() { echo "$(date -u '+%F %T') $*" >> $L; }
fail() { log "FAIL $*"; rm -rf $D/jit $B; echo "FAIL $*" > $M; exit 1; }
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp
mkdir -p $TMPDIR $R
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); log "start df / $use%"; [ "$use" -le 70 ] || fail "df / $use%"
gb=$(timeout 120 du -sxBG $F | cut -f1 | tr -dc 0-9); log "footprint ${gb}G"; [ "${gb:-999}" -le 136 ] || fail "footprint ${gb}G"

# (1) build
# A finished build leaves a stamp: a killed copy of t48 also has a _ttnn.so (job 473 ran on such a tree).
if [ "$(cat $B/.t376_built 2>/dev/null)" != "$REV" ]; then
  rm -rf $B
  cp -a --reflink=always $F/t48 $B || fail "copy rc=$?"
  rm -rf $B/build $B/build_Release $B/python_env $B/tmp
  ( cd $B && timeout 600 git fetch -q origin ttp/t48-ltx25-integrated && git checkout -q -f $REV && git submodule update --init --recursive -q && git clean -fdq ) >> $L 2>&1 || fail "checkout rc=$?"
  ( cd $B && CMAKE_BUILD_PARALLEL_LEVEL=32 timeout 5400 ./build_metal.sh --build-type Release ) > $D/build.log 2>&1; rc=$?
  log "build rc=$rc"; [ $rc = 0 ] || { tail -40 $D/build.log > $R/build_tail.log; fail "build rc=$rc"; }
  echo $REV > $B/.t376_built
fi
cp $D/bruteforce_conv3d_sweep_ltx.py $D/test_vae_ltx_blk_ab_4x8.py $B/models/tt_dit/tests/models/ltx/ || fail "test copy"
log "checkout $(git -C $B log -1 --format='%h %s' | cut -c1-80) dirty=$(git -C $B status --porcelain | tr '\n' ' ')"
[ -r $B/ttnn/ttnn/_ttnn.so ] || fail "missing _ttnn.so"

# (2) device jobs
st() { tt-device-mcp status -j $1 2>&1 | awk '/^Status:/{print $2}'; }
waitjob() { for i in $(seq 1 240); do s=$(st $1); case $s in running|queued|pending|"") sleep 30;; *) echo $s; return;; esac; done; echo stillrunning; }
submit() {  # args: the run376.sh arguments
  for i in $(seq 1 720); do
    out=$(tt-device-mcp status 2>&1)
    run=$(echo "$out" | sed -n '/^RUNNING/,/^QUEUED/p'); q=$(echo "$out" | sed -n '/^QUEUED/,/^RECENT/p')
    if echo "$out" | grep -qi upgrade || echo "$run$q" | grep -qiE 'hold|health|reset|fabric-check|smarton'; then sleep 30; continue; fi
    bash $D/lint.sh --device --env $D/env376.yaml --timeout 600 t376 bash $D/run376.sh >> $L 2>&1 || { log "lint refused"; return; }
    sub=$(tt-device-mcp run-bg "bash $D/run376.sh $(printf '%q ' "$@")" -w $D -e $D/env376.yaml -t 600 2>&1)
    echo "$sub" >> $L; echo "$sub" | grep -oE 'Job [0-9]+' | head -1 | grep -oE '[0-9]+'; return
  done
}
job() {  # name, run376.sh args; sets JOK
  local name=$1; shift; JOK=0
  for att in 1 2; do
    rm -f $D/run.log
    J=$(submit "$@"); [ -n "$J" ] || { log "$name notsubmitted"; echo "$name notsubmitted" >> $D/jobs.txt; return; }
    log "$name attempt=$att job=$J"
    s=$(waitjob $J); log "$name job=$J status=$s"
    cp $D/run.log $R/run_${name}_job$J.log 2>/dev/null
    sleep 10; log "job=$J leftover: $(ps -u $(id -u) -o pid=,args= | grep -E 'pytest|run376' | grep -v grep | tr '\n' ';')"
    echo "$name job=$J status=$s exit=$(grep -oE '^T376_EXIT=[0-9]+' $D/run.log 2>/dev/null)" >> $D/jobs.txt
    grep -q '^T376_EXIT=0' $D/run.log 2>/dev/null && { JOK=1; return; }
    grep -q '^T376_EXIT=' $D/run.log 2>/dev/null && return
    log "$name job=$J DROP? status=$s (no exit line) $(date -u +%FT%TZ); waiting, then rerun"; echo "$name DROP job=$J $(date -u +%FT%TZ)" >> $D/jobs.txt; sleep 240
  done
}
CAND="64,256,2,4,4;64,256,2,2,8;64,256,2,8,2;32,256,2,8,4;32,256,2,4,8;32,256,4,8,4;32,256,2,8,8;64,128,2,8,4;64,128,2,8,8;64,128,4,8,4;64,128,2,4,8;32,256,4,4,8"
C_s4res_x="128,64,6,4,8;128,64,5,4,8;128,64,7,4,8;128,64,4,4,8;128,64,6,2,16;128,64,6,8,4;128,64,6,16,2;128,64,3,4,8;128,64,5,2,16;128,64,5,8,4;128,64,5,16,2;128,64,6,2,8;128,64,6,4,4;128,64,6,4,16;128,64,6,8,2;128,64,6,8,8;128,64,6,16,4;128,64,7,2,16;128,64,7,8,4;128,64,7,16,2;128,64,2,4,8;128,64,4,2,16;128,64,4,8,4;128,64,4,16,2;128,64,5,2,8;128,64,5,2,32;128,64,5,4,4;128,64,5,4,16;128,64,5,8,2;128,64,5,8,8;128,64,5,16,4;128,64,5,32,2;128,64,7,2,8;128,64,7,4,4;128,64,7,4,16;128,64,7,8,2;128,64,7,8,8;128,64,7,16,4;128,64,1,4,8;128,64,3,2,16"
C_s1up_x="128,64,5,2,16;128,64,4,2,16;128,64,6,2,16;128,32,5,2,16;128,64,3,2,16;128,64,5,4,8;128,64,5,8,4;128,64,5,16,2;128,64,7,2,16;128,32,4,2,16;128,32,6,2,16;128,64,2,2,16;128,64,4,4,8;128,64,4,8,4;128,64,4,16,2;128,64,5,2,8;128,64,5,4,4;128,64,5,8,2;128,64,6,4,8;128,64,6,8,4;128,64,6,16,2;128,32,3,2,16;128,32,5,4,8;128,32,5,8,4;128,32,5,16,2;128,32,7,2,16;128,64,1,2,16;128,64,3,4,8;128,64,3,8,4;128,64,3,16,2;128,64,4,1,16;128,64,4,2,8;128,64,4,4,4;128,64,4,4,16;128,64,4,8,2;128,64,4,8,8;128,64,4,16,1;128,64,4,16,4;128,64,6,1,16;128,64,6,2,8"
C_s0res_x="128,64,5,4,8;128,64,4,4,8;128,64,6,4,8;128,32,5,4,8;128,64,3,4,8;128,64,5,8,4;128,64,7,4,8;128,32,4,4,8;128,32,6,4,8;128,64,2,4,8;128,64,4,8,4;128,64,5,2,8;128,64,5,4,4;128,64,5,8,2;128,64,6,8,4;128,32,3,4,8;128,32,5,8,4;128,32,7,4,8;128,64,1,4,8;128,64,3,8,4;128,64,4,2,8;128,64,4,4,4;128,64,4,8,2;128,64,4,8,8;128,64,6,2,8;128,64,6,4,4;128,64,6,8,2;128,64,7,8,4;128,32,2,4,8;128,32,4,8,4;128,32,5,2,8;128,32,5,4,4;128,32,5,8,2;128,32,5,8,8;128,32,6,8,4;128,64,2,8,4;128,64,3,2,8;128,64,3,4,4;128,64,3,8,2;128,64,3,8,8"
C_s0up_x="128,64,5,4,8;128,64,4,4,8;128,64,6,4,8;128,32,5,4,8;128,64,3,4,8;128,64,5,8,4;128,64,7,4,8;128,32,4,4,8;128,32,6,4,8;128,64,2,4,8;128,64,4,8,4;128,64,5,2,8;128,64,5,4,4;128,64,5,8,2;128,64,6,8,4;128,32,3,4,8;128,32,5,8,4;128,32,7,4,8;128,64,1,4,8;128,64,3,8,4;128,64,4,2,8;128,64,4,4,4;128,64,4,8,2;128,64,4,8,8;128,64,6,2,8;128,64,6,4,4;128,64,6,8,2;128,64,7,8,4;128,32,2,4,8;128,32,4,8,4;128,32,5,2,8;128,32,5,4,4;128,32,5,8,2;128,32,5,8,8;128,32,6,8,4;128,64,2,8,4;128,64,3,2,8;128,64,3,4,4;128,64,3,8,2;128,64,3,8,8"
C_s4out_x="128,64,6,2,16;128,64,5,2,16;128,64,7,2,16;128,32,6,2,16;128,64,4,2,16;128,64,6,4,8;128,64,6,8,4;128,64,6,16,2;128,32,5,2,16;128,32,7,2,16;128,64,3,2,16;128,64,5,4,8;128,64,5,8,4;128,64,5,16,2;128,64,6,2,8;128,64,6,4,4;128,64,6,4,16;128,64,6,8,2;128,64,6,8,8;128,64,6,16,4;128,64,7,4,8;128,64,7,8,4;128,64,7,16,2;128,32,4,2,16;128,32,6,4,8;128,32,6,8,4;128,32,6,16,2;128,64,2,2,16;128,64,4,4,8;128,64,4,8,4;128,64,4,16,2;128,64,5,2,8;128,64,5,2,32;128,64,5,4,4;128,64,5,4,16;128,64,5,8,2;128,64,5,8,8;128,64,5,16,4;128,64,5,32,2;128,64,7,2,8"
for layer in s4res_x s1up_x s0res_x s0up_x s4out_x; do v=C_$layer; job $layer sweep $layer "${!v}"; done

# (3) winners, (4) A/B
source $F/t48/python_env/bin/activate
NEW=$(python $D/pick376.py $R 2>> $L); log "winners: ${NEW:-none}"
echo "${NEW:-none}" > $R/winners.txt
AB=skip
if [ -n "$NEW" ]; then
  job ab ab "$NEW"; AB=$JOK
  if [ $AB = 0 ] && grep -q "^T376_EXIT=124" $D/run.log 2>/dev/null; then log "ab timed out (cold JIT?): rerun warm"; job ab2 ab "$NEW"; AB=$JOK; fi
fi

# (5) cleanup
cp $D/build.log $R/build.log 2>/dev/null && gzip -f $R/build.log
rm -rf $D/jit $B; log "cleaned jit b; df / $(df --output=pcent / | tail -1 | tr -dc 0-9)%"
echo "DONE ab=$AB $(tr '\n' ' ' < $D/jobs.txt)" > $M
