#!/bin/bash
# t374 driver on blx01 (no device work itself): (1) worktree of /var/tmp/fasth3/t48's repo at the t374 head (from
# t374.bundle) + Release build in t374/b; (2) broker jobs one at a time, -t 600 each (unmeasured): run374.sh <layer>
# for each layer in $LAYERS (default s3_res). Submitted only when the broker shows no upgrade/hold/health/reset/
# fabric-check job and no other smarton job. A non-completed job without its T374_EXIT line counts as a drop and is
# rerun once. Logs: t374/drv374.log, t374/run_<layer>_job<id>.log. Marker: t374/drv374.done.
set -o pipefail
F=/var/tmp/fasth3; A=$F/t48; D=$F/t374; B=$D/b; M=$D/drv374.done; L=$D/drv374.log
REV=8a365da812999c731e2206eaba764bc6c4c37f3e
trap 'rc=$?; echo "exit=$rc $(date -u +%T)" >> $L; [ -e $M ] || echo "DRIVER_EXIT rc=$rc" > $M' EXIT
log() { echo "$(date -u '+%F %T') $*" >> $L; }
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp CPM_SOURCE_CACHE=$F/.cpmcache
build() {
  set -x
  git -C $A cat-file -e $REV^{commit} 2>/dev/null || git -C $A fetch -q $D/t374.bundle refs/heads/t374 || return 11
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
  TT_METAL_HOME=$B PYTHONPATH=$B:$B/ttnn:$B/tools python -c "import ttnn; from models.tt_dit.models.vae import vae_ltx; print('IMPORT_OK')" || return 20
}
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 60 ] || { echo "DISK / $use%" > $M; exit 1; }
build > $D/build.log 2>&1; brc=$?; log "build rc=$brc"
[ $brc = 0 ] || { echo "BUILD_FAILED rc=$brc" > $M; exit 1; }
st() { tt-device-mcp status -j $1 2>&1 | awk '/^Status:/{print $2}'; }
waitjob() { for i in $(seq 1 120); do s=$(st $1); case $s in running|queued|pending|"") sleep 30;; *) echo $s; return;; esac; done; echo stillrunning; }
submit() {  # layer -> job id
  for i in $(seq 1 720); do
    out=$(tt-device-mcp status 2>&1)
    run=$(echo "$out" | sed -n '/^RUNNING/,/^QUEUED/p'); q=$(echo "$out" | sed -n '/^QUEUED/,/^RECENT/p')
    if echo "$out" | grep -qi upgrade || echo "$run$q" | grep -qiE 'hold|health|reset|fabric-check|smarton'; then sleep 30; continue; fi
    sub=$(tt-device-mcp run-bg "bash $D/run374.sh $1" -w $D -e $D/env374.yaml -t 600 2>&1)
    echo "$sub" >> $L; echo "$sub" | grep -oE 'Job [0-9]+' | head -1 | grep -oE '[0-9]+'; return
  done
}
OUT=""
for spec in ${LAYERS:-s3_res}; do
  tag=$spec
  for att in 1 2; do
    J=$(submit $tag); [ -n "$J" ] || { OUT="$OUT $tag:notsubmitted"; break; }
    log "tag=$tag attempt=$att job=$J"
    s=$(waitjob $J); log "tag=$tag job=$J status=$s"
    cp $D/out_$tag/run.log $D/run_${tag}_job$J.log 2>/dev/null
    sleep 10; log "tag=$tag job=$J leftover: $(ps -u $(id -u) -o pid=,pgid=,args= | grep -E 'pytest|run374' | grep -v grep | tr '\n' ';')"
    OUT="$OUT $tag:job=$J:$s"
    { [ "$s" = completed ] || grep -q '^T374_EXIT=' $D/out_$tag/run.log 2>/dev/null; } && break
    log "tag=$tag job=$J DROP? status=$s (no exit line); waiting for the broker, then rerun"; sleep 240
  done
done
echo "DONE$OUT" > $M
