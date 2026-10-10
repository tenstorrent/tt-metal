#!/bin/bash
# t333 (non-device, blx01): (1) copy the LTX-2.5 split checkpoints from the MLPerf snapshot (network fs) to local
# /var/tmp/fasth3/models/ltx-2.5 with a streamed sha256 check, so device jobs never read /mnt/MLPerf;
# (2) worktree of /var/tmp/fasth3/t48's repo at ttp/t48-ltx25-integrated f6547442b30 + Release build in t333/b.
# Everything under /var/tmp/fasth3 (blx01 /home is full). Marker line in t333/driver.log.
F=/var/tmp/fasth3; A=$F/t48; T=$F/t333; B=$T/b; M=$F/models/ltx-2.5
S=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941
REV=f6547442b304d744711e80e2281f6bd368291673
FILES="diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors
latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors
text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
vae/ltx-2.5-audio-vae-bf16.safetensors"
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp CPM_SOURCE_CACHE=$F/.cpmcache
mkdir -p $T $M $TMPDIR
copy() {
  # 72G to write: / must stay at or under 70% used, and /var/tmp/fasth3 under 150G (memory, 2026-10-08).
  use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 55 ] || { echo "[t333] / at $use%"; return 21; }
  gb=$(timeout 300 du -sxBG $F | cut -f1 | tr -dc 0-9); [ "${gb:-999}" -le 70 ] || { echo "[t333] $F ${gb}G"; return 21; }
  for f in $FILES; do
    mkdir -p $M/$(dirname $f)
    if [ -f $M/$f ] && grep -q " $f\$" $M/SHA256SUMS 2>/dev/null; then echo "[t333] have $f"; continue; fi
    s1=$(timeout 5400 nice ionice -c3 cat $S/$f | tee $M/$f.part | sha256sum | cut -c1-64) || return 22
    s2=$(sha256sum < $M/$f.part | cut -c1-64)
    [ -n "$s1" ] && [ "$s1" = "$s2" ] && [ "$(stat -c %s $M/$f.part)" = "$(stat -Lc %s $S/$f)" ] \
      || { echo "[t333] verify failed $f $s1 $s2"; return 23; }
    mv $M/$f.part $M/$f; echo "$s2  $f" >> $M/SHA256SUMS; echo "[t333] copied $f $s2 $(date -u +%T)"
  done
}
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
  TT_METAL_HOME=$B PYTHONPATH=$B:$B/ttnn:$B/tools python -c "import ttnn, models.tt_dit.pipelines.ltx.pipeline_ltx25_distilled as p; print('IMPORT_OK', ttnn.__file__, p.__file__)" || return 20
}
copy > $T/copy.log 2>&1 & CP=$!
build > $T/build.log 2>&1; brc=$?
wait $CP; crc=$?
rc=$brc; [ $rc = 0 ] && rc=$crc
echo "T333_DRIVER_DONE setup $rc build=$brc copy=$crc $(date -u '+%F %T')" >> $T/driver.log; exit $rc
