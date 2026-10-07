#!/bin/bash
# t220 CPU-only setup on blx03: ltx-rt worktree + Release build under ~/fasth3/t220, and a project copy
# of the live service's LTX-2.3 tt_dit_cache (read-only source) under /var/tmp/fasth3/cache/dit-ltx23.
set -uo pipefail
V=/var/tmp/fasth3/t220; W=/home/smarton/fasth3/t220; A=/home/smarton/fasth3/tt-metal; REV=b9f8587ce6c
C=/var/tmp/fasth3/cache/dit-ltx23; S=/home/sulphur/tt_dit_cache
mkdir -p $V $C
log() { echo "$(date -u '+%F %T') $*" >> $V/driver.log; }
copy() {
  for d in gemma-3-12b-it-qat-q4_0-unquantized ltx-2.3-22b-distilled-1.1 ltx-2.3-spatial-upscaler-x2-1.1 ltx-embeddings-v2; do
    [ -d $S/$d ] && cp -a $S/$d $C/ || return 21
  done
}
build() {
  git -C $A cat-file -e $REV^{commit} 2>/dev/null || git -C $A fetch -q origin ltx-rt || return 11
  [ -d $W ] || git -C $A worktree add --detach $W $REV || return 12
  cd $W || return 13
  [ "$(git rev-parse --short=11 HEAD)" = $REV ] || return 14
  for n in tracy umd tt-cluster-descriptors; do
    git submodule update --init --depth 1 -- tt_metal/third_party/$n || return 15
  done
  bash build_metal.sh --release --cpm-source-cache $A/.cpmcache > $V/build.log 2>&1 || return 16
  test -f ttnn/ttnn/_ttnn.so || return 17
}
log start
copy & CP=$!
build; rb=$?; log "build rc=$rb"
wait $CP; rc=$?; log "copy rc=$rc"; du -sh $C >> $V/driver.log
log "T220_DRIVER_DONE setup build=$rb copy=$rc"
