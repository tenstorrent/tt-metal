#!/bin/bash
# t479 (copy of t466/cp466.sh): copy the MiniMax-H3 files the Turbo loader reads (test_pipeline_turbo_minimax_h3.py:171) from the
# read-only weka copy to blx01 local disk. Not a device job. Throttled per tt-buddy #311 (nice, ionice -c3,
# rsync --bwlimit=100M). Space gate (coordinator 10-13 05:50 PDT, #466 only): projected / free after weights +
# h3hf cache (115 GiB) + JIT cache (16 GiB) stays >= 250 GiB, FastH3 total <= 320 GB. A watcher re-checks the
# projection every 20 s; if it fails, the copy stops and everything it wrote is deleted.
set -o pipefail
F=/var/tmp/fasth3; S=/mnt/MLPerf/tt-shield/persistent-volume/volume_id_tt_transformers-MiniMax-H3-v0.22.0/weights/MiniMax-H3
D=$F/models/MiniMax-H3; LOG=$F/t479/cp479.log; WGB=136; CACHE=131; FLOOR=250
log() { echo "[cp479] $(date -u '+%F %T') $*" | tee -a $LOG; }
case "$(readlink -f $F)" in /mnt/*|"") echo "$F resolves under /mnt"; exit 4;; esac
mkdir -p $D
free() { df -BG --output=avail / | tail -1 | tr -dc 0-9; }
copied() { timeout 60 du -sxBG $D 2>/dev/null | cut -f1 | tr -dc 0-9; }
ok() { local f c; f=$(free); c=$(copied); [ $((f - (WGB - ${c:-0}) - CACHE)) -ge $FLOOR ]; }
tot=$(timeout 120 du -sxBG $F 2>/dev/null | cut -f1 | tr -dc 0-9)
log "start: free=$(free)G footprint=${tot}G copied=$(copied)G"
ok || { log "gate fails before copy"; exit 5; }
[ $(( ${tot:-999} - $(copied) + WGB + CACHE)) -le 320 ] || { log "footprint gate fails"; exit 5; }
nice -n19 ionice -c3 rsync -a --bwlimit=100M --exclude=transformer_ref --exclude=.cache --exclude=README.md \
  --exclude=LICENSE --exclude=.gitattributes $S/ $D/ >> $LOG 2>&1 & RP=$!
while kill -0 $RP 2>/dev/null; do
  sleep 20
  if ! ok; then
    log "space gate failed mid-copy: free=$(free)G copied=$(copied)G; stopping and deleting $D"
    kill $RP; wait $RP; rm -rf $D; exit 5
  fi
done
wait $RP; rc=$?
log "rsync rc=$rc free=$(free)G copied=$(copied)G"
[ $rc = 0 ] || exit $rc
ls $D/text_encoder/config.json $D/transformer/config.json $D/vae/config.json $D/audio_vae/config.json >> $LOG || exit 6
log done
