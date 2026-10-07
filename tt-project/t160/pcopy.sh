#!/bin/bash
# pcopy.sh: copy the two big LTX-2.5 files to exabox in 1 GiB chunks over P parallel ssh streams
# (one stream through the Mac tunnel ran at ~1.5 MB/s). Resumable: finished chunks are listed in $M.
# Verifies each file by sha256 against its HF blob name, then prints PCOPY_OK.
set -eo pipefail
S=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941
R=/data/smarton/fasth3/models/ltx-2.5
M=${M:-/home/smarton/fasth3/tt-metal/tt-project/t160/pcopy.done}; P=${P:-8}; CH=$((1<<30))
touch $M
chunk() {  # $1=file $2=index
  grep -qx "$1 $2" $M && return 0
  local t=$(date +%s)
  dd if=$S/$1 bs=4M skip=$(( $2*256 )) count=256 status=none |
    ssh -o BatchMode=yes -o ControlMaster=no -o ControlPath=none exabox-login \
      "dd of=$R/$1 bs=4M seek=$(( $2*256 )) conv=notrunc status=none"
  echo "$1 $2" >> $M
  echo "$(date -u +%T) $1 chunk $2 $(( $(date +%s)-t ))s"
}
export -f chunk; export S R M
for f in text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors; do
  sz=$(stat -L -c %s $S/$f); n=$(( (sz+CH-1)/CH ))
  echo "$(date -u +%T) $f size=$sz chunks=$n P=$P"
  seq 0 $((n-1)) | xargs -P $P -I{} bash -c "chunk $f {}"
  ssh -o BatchMode=yes exabox-login "truncate -s $sz $R/$f; chmod 644 $R/$f"
done
for f in text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors; do
  want=$(basename "$(readlink $S/$f)")
  got=$(ssh -o BatchMode=yes exabox-login "sha256sum $R/$f" | cut -d' ' -f1)
  echo "$f sha256 want=$want got=$got"; [ "$want" = "$got" ]
done
echo PCOPY_OK
