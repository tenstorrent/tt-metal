#!/bin/bash
# t365 driver on blx01 (no device work itself): (1) worktree of /var/tmp/fasth3/t48's repo at 53da67e9c61 (from t365.bundle)
# + Release build in t365/b; (2) one broker job (-t 240: #362 job 466 took 126 s incl. cold JIT) run365.sh, submitted only when the broker shows no
# upgrade/hold/health/reset/fabric-check job and no other smarton job. No T365_EXIT line = drop, rerun once.
# Logs: t365/drv365.log, t365/run_job<id>.log. Marker: t365/drv365.done.
set -o pipefail
F=/var/tmp/fasth3; A=$F/t48; D=$F/t365; B=$D/b; M=$D/drv365.done; L=$D/drv365.log
REV=53da67e9c618b7781882bd9ac780e86aa3f49cde
trap 'rc=$?; echo "exit=$rc $(date -u +%T)" >> $L; [ -e $M ] || echo "DRIVER_EXIT rc=$rc" > $M' EXIT
log() { echo "$(date -u '+%F %T') $*" >> $L; }
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp CPM_SOURCE_CACHE=$F/.cpmcache
build() {
  set -x
  git -C $A cat-file -e $REV^{commit} 2>/dev/null || git -C $A fetch -q $D/t365.bundle HEAD || return 11
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
    sub=$(tt-device-mcp run-bg "bash $D/run365.sh" -w $D -e $D/env365.yaml -t 240 2>&1)
    echo "$sub" >> $L; echo "$sub" | grep -oE 'Job [0-9]+' | head -1 | grep -oE '[0-9]+'; return
  done
}
OUT=""
for att in 1 2; do
  J=$(submit); [ -n "$J" ] || { OUT="$OUT notsubmitted"; break; }
  log "attempt=$att job=$J"
  s=$(waitjob $J); log "job=$J status=$s"
  cp $D/out/run.log $D/run_job$J.log 2>/dev/null
  for f in arm0 arm1; do cp $D/out/$f.log $D/${f}_job$J.log 2>/dev/null; done
  sleep 10; log "job=$J leftover: $(ps -u $(id -u) -o pid=,pgid=,args= | grep -E 'pytest|run365' | grep -v grep | tr '\n' ';')"
  OUT="$OUT job=$J:$s"
  { [ "$s" = completed ] || grep -q '^T365_EXIT=' $D/out/run.log 2>/dev/null; } && break
  log "job=$J DROP? status=$s (no exit line); waiting for the broker, then rerun"; rm -f $D/out/run.log; sleep 240
done
echo "DONE$OUT" > $M
