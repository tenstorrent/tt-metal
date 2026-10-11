#!/bin/bash
# t380 driver on blx01: (1) worktree t380/b of /var/tmp/fasth3/t48's repo at REV (from t380.bundle) + Release build;
# (2) one broker job, -t 600: run380.sh REV. Submitted only when the broker shows no upgrade/hold/health/reset/
# fabric-check job and no other smarton job. A non-completed job without its T380_EXIT line counts as a drop and is
# rerun once after 240 s. Log: t380/drv380.log, job log: t380/run_job<id>.log. Marker: t380/drv380.done.
set -o pipefail
F=/var/tmp/fasth3; A=$F/t48; D=$F/t380; B=$D/b; M=$D/drv380.done; L=$D/drv380.log
REV=${1:?rev}
trap 'rc=$?; echo "exit=$rc $(date -u +%T)" >> $L; [ -e $M ] || echo "DRIVER_EXIT rc=$rc" > $M' EXIT
log() { echo "$(date -u '+%F %T') $*" >> $L; }
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp CPM_SOURCE_CACHE=$F/.cpmcache
build() {
  set -x
  git -C $A cat-file -e $REV^{commit} 2>/dev/null || git -C $A fetch -q $D/t380.bundle || return 11
  git -C $A cat-file -e $REV^{commit} || return 12
  [ -d $B ] || git -C $A worktree add --detach $B $REV || return 13
  cd $B || return 14
  git checkout -q --detach $REV || return 21
  [ "$(git rev-parse HEAD)" = "$(git rev-parse $REV)" ] || return 15
  git submodule update --init tt_metal/third_party/tracy tt_metal/third_party/tt-cluster-descriptors \
    tt_metal/third_party/umd || return 16
  ./build_metal.sh --build-type Release || return 17
  test -f ttnn/ttnn/_ttnn.so || return 18
  source $A/python_env/bin/activate || return 19
  TT_METAL_HOME=$B PYTHONPATH=$B:$B/ttnn:$B/tools python -c "import ttnn; print('IMPORT_OK')" || return 20
}
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 60 ] || { echo "DISK / $use%" > $M; exit 1; }
log "build start $REV"
build > $D/build.log 2>&1; brc=$?; log "build rc=$brc"
[ $brc = 0 ] || { echo "BUILD_FAILED rc=$brc" > $M; exit 1; }
st() { tt-device-mcp status -j $1 2>&1 | awk '/^Status:/{print $2}'; }
waitjob() { for i in $(seq 1 120); do s=$(st $1); case $s in running|queued|pending|"") sleep 30;; *) echo $s; return;; esac; done; echo stillrunning; }
submit() {
  for i in $(seq 1 720); do
    out=$(tt-device-mcp status 2>&1)
    run=$(echo "$out" | sed -n '/^RUNNING/,/^QUEUED/p'); q=$(echo "$out" | sed -n '/^QUEUED/,/^RECENT/p')
    if echo "$out" | grep -qi upgrade || pgrep -f autoupdate.sh >/dev/null || echo "$run$q" | grep -qiE 'hold|health|reset|fabric-check|smarton'; then sleep 30; continue; fi
    sub=$(tt-device-mcp run-bg "bash $D/run380.sh $(git -C $B rev-parse --short=11 HEAD)" -w $D -e $D/env380.yaml -t 600 2>&1)
    echo "$sub" >> $L; echo "$sub" | grep -oE 'Job [0-9]+' | head -1 | grep -oE '[0-9]+'; return
  done
}
OUT=""
for att in 1 2; do
  J=$(submit); [ -n "$J" ] || { OUT="$OUT notsubmitted"; break; }
  log "attempt=$att job=$J"
  s=$(waitjob $J); log "job=$J status=$s"
  cp $D/out/run.log $D/run_job$J.log 2>/dev/null
  OUT="$OUT job=$J:$s:$(grep -o '^T380_EXIT=[0-9]*' $D/run_job$J.log 2>/dev/null)"
  grep -q '^T380_EXIT=' $D/run_job$J.log 2>/dev/null && break
  log "job=$J DROP? status=$s (no exit line); waiting for the broker, then rerun"; sleep 240
done
echo "DONE$OUT" > $M
