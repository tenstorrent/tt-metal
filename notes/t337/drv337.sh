#!/bin/bash
# t337 driver on blx01 (no device work itself): (1) worktree of /var/tmp/fasth3/t48's repo at 63e54d98036 + Release build
# in t337/b; (2) broker jobs one at a time, -t 570: conv25 (headline), then conv23 (2.3 monolith, md5 check). Submitted
# only when the broker shows no upgrade/hold/health/reset/fabric-check job and no other smarton job. A job without its
# T337_EXIT line counts as a drop and is rerun once. Logs: t337/run_<arm>_job<id>.log. Marker: t337/drv337.done.
set -o pipefail
F=/var/tmp/fasth3; A=$F/t48; D=$F/t337; B=$D/b; M=$D/drv337b.done; L=$D/drv337b.log
REV=63e54d98036e3e954f3292a7d70dd0fef575fafd
trap 'rc=$?; echo "exit=$rc $(date -u +%T)" >> $L; [ -e $M ] || echo "DRIVER_EXIT rc=$rc" > $M' EXIT
log() { echo "$(date -u '+%F %T') $*" >> $L; }
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp CPM_SOURCE_CACHE=$F/.cpmcache
build() {
  set -x
  git -C $A cat-file -e $REV^{commit} 2>/dev/null || timeout 1800 git -C $A fetch -q origin ttp/t48-ltx25-integrated || return 11
  git -C $A cat-file -e $REV^{commit} || return 12
  [ -d $B ] || git -C $A worktree add --detach $B $REV || return 13
  cd $B || return 14
  [ "$(git rev-parse HEAD)" = $REV ] || return 15
  git submodule update --init tt_metal/third_party/tracy tt_metal/third_party/tt-cluster-descriptors \
    tt_metal/third_party/umd || return 16
  ./build_metal.sh --build-type Release || return 17
  test -f ttnn/ttnn/_ttnn.so || return 18
  source $A/python_env/bin/activate || return 19
  TT_METAL_HOME=$B PYTHONPATH=$B:$B/ttnn:$B/tools python -c "import ttnn, models.tt_dit.pipelines.ltx.pipeline_ltx25_distilled as p; from models.tt_dit.models.vae.vae_ltx import vae_key_map; print('IMPORT_OK', p.__file__)" || return 20
}
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 60 ] || { echo "DISK / $use%" > $M; exit 1; }
# rerun (b): build already done; both arms of run 1 hit the 540 s pytest timeout on cold JIT (audio warmup 262 s).
[ "$(git -C $B rev-parse HEAD)" = $REV ] && [ -f $B/ttnn/ttnn/_ttnn.so ] || { echo "BUILD_MISSING" > $M; exit 1; }
V=$F/ltx25_vae/ltx-2.5-video-vae-conv-bf16.safetensors
vs=$(stat -c %s $V); vh=$(sha256sum < $V | cut -c1-64); log "vae25 size=$vs sha256=$vh"
[ "$vs" = 1452269922 ] && [ "$vh" = 685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d ] || { echo "VAE_MISMATCH $vs $vh" > $M; exit 1; }
st() { tt-device-mcp status -j $1 2>&1 | awk '/^Status:/{print $2}'; }
waitjob() { for i in $(seq 1 120); do s=$(st $1); case $s in running|queued|pending|"") sleep 30;; *) echo $s; return;; esac; done; echo stillrunning; }
submit() {
  for i in $(seq 1 720); do
    out=$(tt-device-mcp status 2>&1)
    run=$(echo "$out" | sed -n '/^RUNNING/,/^QUEUED/p'); q=$(echo "$out" | sed -n '/^QUEUED/,/^RECENT/p')
    if echo "$out" | grep -qi upgrade || echo "$run$q" | grep -qiE 'hold|health|reset|fabric-check|smarton'; then sleep 30; continue; fi
    sub=$(tt-device-mcp run-bg "bash $D/run337.sh $1" -w $D -e $D/env337.yaml -t 570 2>&1)
    echo "$sub" >> $L; echo "$sub" | grep -oE 'Job [0-9]+' | head -1 | grep -oE '[0-9]+'; return
  done
}
OUT=""
for arm in conv25 conv23; do
  for att in 1 2; do
    J=$(submit $arm); [ -n "$J" ] || { OUT="$OUT $arm:notsubmitted"; break; }
    log "arm=$arm attempt=$att job=$J"
    s=$(waitjob $J); log "arm=$arm job=$J status=$s"
    cp $D/out_$arm/run.log $D/run_${arm}_job$J.log 2>/dev/null
    sleep 10; log "arm=$arm job=$J leftover: $(ps -u $(id -u) -o pid=,pgid=,args= | grep -E 'pytest|run337' | grep -v grep | tr '\n' ';')"
    OUT="$OUT $arm:job=$J:$s"
    { [ "$s" = completed ] || grep -q '^T337_EXIT=' $D/out_$arm/run.log 2>/dev/null; } && break
    log "arm=$arm job=$J DROP? status=$s (no exit line); waiting for the broker, then rerun"; sleep 240
  done
done
echo "DONE$OUT" > $M
