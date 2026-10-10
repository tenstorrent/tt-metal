#!/bin/bash
# t362 driver on blx01 (no device work itself): (1) worktree of /var/tmp/fasth3/t48's repo at b5d8b7f567eb9acca5f342b83f2260a0331413db (from t362.bundle)
# + Release build in t362/b; (2) one broker job (-t 600, unmeasured) run362.sh, submitted only when the broker shows no
# upgrade/hold/health/reset/fabric-check job and no other smarton job. No T362_EXIT line = drop, rerun once.
# Logs: t362/drv362.log, t362/run_job<id>.log. Marker: t362/drv362.done.
set -o pipefail
F=/var/tmp/fasth3; A=$F/t48; D=$F/t362; B=$D/b; M=$D/drv362.done; L=$D/drv362.log
REV=b5d8b7f567eb9acca5f342b83f2260a0331413db
trap 'rc=$?; echo "exit=$rc $(date -u +%T)" >> $L; [ -e $M ] || echo "DRIVER_EXIT rc=$rc" > $M' EXIT
log() { echo "$(date -u '+%F %T') $*" >> $L; }
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp CPM_SOURCE_CACHE=$F/.cpmcache
build() {
  set -x
  git -C $A cat-file -e $REV^{commit} 2>/dev/null || git -C $A fetch -q $D/t362.bundle HEAD || return 11
  git -C $A cat-file -e $REV^{commit} || return 12
  [ -d $B ] || git -C $A worktree add --detach $B $REV || return 13
  cd $B || return 14
  git checkout -q --detach $REV || return 21
  [ "$(git rev-parse HEAD)" = $REV ] || return 15
  git submodule update --init tt_metal/third_party/tracy tt_metal/third_party/tt-cluster-descriptors \
    tt_metal/third_party/umd || return 16
  ./build_metal.sh --build-type Release || return 17
  test -f ttnn/ttnn/_ttnn.so || return 18
  source $A/python_env/bin/activate || return 19
  TT_METAL_HOME=$B PYTHONPATH=$B:$B/ttnn:$B/tools python -c "import ttnn; from models.tt_dit.models.vae import vae_ltx; import inspect; assert 'input_patch_size' in ttnn.experimental.rgb_to_yuv.__doc__; print('IMPORT_OK')" || return 20
}
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 60 ] || { echo "DISK / $use%" > $M; exit 1; }
build > $D/build.log 2>&1; brc=$?; log "build rc=$brc"
[ $brc = 0 ] || { echo "BUILD_FAILED rc=$brc" > $M; exit 1; }
st() { tt-device-mcp status -j $1 2>&1 | awk '/^Status:/{print $2}'; }
waitjob() { for i in $(seq 1 120); do s=$(st $1); case $s in running|queued|pending|"") sleep 30;; *) echo $s; return;; esac; done; echo stillrunning; }
submit() {
  for i in $(seq 1 720); do
    out=$(tt-device-mcp status 2>&1)
    run=$(echo "$out" | sed -n '/^RUNNING/,/^QUEUED/p'); q=$(echo "$out" | sed -n '/^QUEUED/,/^RECENT/p')
    if echo "$out" | grep -qi upgrade || echo "$run$q" | grep -qiE 'hold|health|reset|fabric-check|smarton'; then sleep 30; continue; fi
    sub=$(tt-device-mcp run-bg "bash $D/run362.sh" -w $D -e $D/env362.yaml -t 600 2>&1)
    echo "$sub" >> $L; echo "$sub" | grep -oE 'Job [0-9]+' | head -1 | grep -oE '[0-9]+'; return
  done
}
OUT=""
for att in 1 2; do
  J=$(submit); [ -n "$J" ] || { OUT="$OUT notsubmitted"; break; }
  log "attempt=$att job=$J"
  s=$(waitjob $J); log "job=$J status=$s"
  cp $D/out/run.log $D/run_job$J.log 2>/dev/null
  for f in unit arm0 arm1; do cp $D/out/$f.log $D/${f}_job$J.log 2>/dev/null; done
  sleep 10; log "job=$J leftover: $(ps -u $(id -u) -o pid=,pgid=,args= | grep -E 'pytest|run362' | grep -v grep | tr '\n' ';')"
  OUT="$OUT job=$J:$s"
  { [ "$s" = completed ] || grep -q '^T362_EXIT=' $D/out/run.log 2>/dev/null; } && break
  log "job=$J DROP? status=$s (no exit line); waiting for the broker, then rerun"; rm -f $D/out/run.log; sleep 240
done
echo "DONE$OUT" > $M
