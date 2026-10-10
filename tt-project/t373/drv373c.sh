#!/bin/bash
# drv373c (run 1445): arm B at bf45db27371 (yuv host assembly deferred to the export worker by default), reusing
# drv373's build; dm = timing split, dw = plain standard run (headline). Arm A = jobs 535 (m) / 538 (w) at e823d13b945.
# drv373b (run 1441): the build from drv373 is reused; jobs 523/525 hit pytest-timeout 540 s in a cold-JIT warm-up
# (525: warm-up done 426 s, then killed at gen#0). JIT cache now filled, so m (timing split) runs first, then w (plain).
# t373 driver on blx01 (no device work itself): (1) worktree of /var/tmp/fasth3/t48's repo at e823d13b945 (from
# t373.bundle) + Release build in t373/b; (2) broker jobs one at a time, -t 570 each: w (plain standard run, cold JIT
# fill; its timing counts only if it finishes), m (same run with TT_DIT_STAGE_TIMING=1 LTX_PERF_BREAKDOWN=2
# TT_DIT_STAGE_LOG=1: splits the VAE decode row into upload / device decode / yuv / readback / host assemble).
# Submitted only when the broker shows no upgrade/hold/health/reset/fabric-check job and no other smarton job.
# A non-completed job without its T373_EXIT line counts as a drop and is rerun once (two in a row: skipped).
# Logs: t373/drv373.log, t373/run_<tag>_job<id>.log. Marker: t373/drv373b.done.
set -o pipefail
F=/var/tmp/fasth3; A=$F/t48; D=$F/t373; B=$D/b; M=$D/drv373c.done; L=$D/drv373c.log
REV=bf45db273714490184923a5f4a13221bb434c281
trap 'rc=$?; echo "exit=$rc $(date -u +%T)" >> $L; [ -e $M ] || echo "DRIVER_EXIT rc=$rc" > $M' EXIT
log() { echo "$(date -u '+%F %T') $*" >> $L; }
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp CPM_SOURCE_CACHE=$F/.cpmcache
build() {
  set -x
  git -C $A cat-file -e $REV^{commit} 2>/dev/null || git -C $A fetch -q $D/t373.bundle refs/heads/ttp/t373-conv-vae-measure-and-cut-the-non-device- || return 11
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
# Python-only commits on top of the built e823d13b945: fetch and check out, keep the build.
git -C $A cat-file -e $REV^{commit} 2>/dev/null || git -C $A fetch -q $D/t373c.bundle HEAD || { echo "FETCH_FAILED" > $M; exit 1; }
git -C $B checkout -q --detach $REV && [ "$(git -C $B rev-parse HEAD)" = $REV ] || { echo "CHECKOUT_FAILED" > $M; exit 1; }
[ -z "$(git -C $B diff --name-only e823d13b945 $REV | grep -vE '\.(py|md)$')" ] || { echo "NON_PY_CHANGE" > $M; exit 1; }
brc=0; test -f $B/ttnn/ttnn/_ttnn.so || brc=18; log "reuse build at $REV rc=$brc"
[ $brc = 0 ] || { echo "BUILD_FAILED rc=$brc" > $M; exit 1; }
st() { tt-device-mcp status -j $1 2>&1 | awk '/^Status:/{print $2}'; }
waitjob() { for i in $(seq 1 120); do s=$(st $1); case $s in running|queued|pending|"") sleep 30;; *) echo $s; return;; esac; done; echo stillrunning; }
submit() {  # tag fuse -> job id
  for i in $(seq 1 720); do
    out=$(tt-device-mcp status 2>&1)
    run=$(echo "$out" | sed -n '/^RUNNING/,/^QUEUED/p'); q=$(echo "$out" | sed -n '/^QUEUED/,/^RECENT/p')
    if echo "$out" | grep -qi upgrade || echo "$run$q" | grep -qiE 'hold|health|reset|fabric-check|smarton'; then sleep 30; continue; fi
    sub=$(tt-device-mcp run-bg "bash $D/run373c.sh $1 $2" -w $D -e $D/env373.yaml -t 570 2>&1)
    echo "$sub" >> $L; echo "$sub" | grep -oE 'Job [0-9]+' | head -1 | grep -oE '[0-9]+'; return
  done
}
OUT=""
for spec in dm:1 dw:0; do
  tag=${spec%%:*}; fuse=${spec##*:}
  for att in 1 2; do
    J=$(submit $tag $fuse); [ -n "$J" ] || { OUT="$OUT $tag:notsubmitted"; break; }
    log "tag=$tag attempt=$att job=$J"
    s=$(waitjob $J); log "tag=$tag job=$J status=$s"
    cp $D/out_$tag/run.log $D/run_${tag}_job$J.log 2>/dev/null
    sleep 10; log "tag=$tag job=$J leftover: $(ps -u $(id -u) -o pid=,pgid=,args= | grep -E 'pytest|run373' | grep -v grep | tr '\n' ';')"
    OUT="$OUT $tag:job=$J:$s"
    { [ "$s" = completed ] || grep -q '^T373_EXIT=' $D/out_$tag/run.log 2>/dev/null; } && break
    log "tag=$tag job=$J DROP? status=$s (no exit line); waiting for the broker, then rerun"; sleep 240
  done
done
echo "DONE$OUT" > $M
