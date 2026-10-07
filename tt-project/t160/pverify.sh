#!/bin/bash
# pverify.sh: hash the transformer file per 1 GiB chunk on both sides, drop mismatched chunks from
# pcopy.done, then rerun pcopy.sh (recopies only those chunks and re-checks the full sha256).
set -eo pipefail
D=/home/smarton/fasth3/tt-metal/tt-project/t160
S=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941
R=/data/smarton/fasth3/models/ltx-2.5
f=diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors
n=$(( ($(stat -L -c %s $S/$f) + (1<<30) - 1) >> 30 ))
seq 0 $((n-1)) | xargs -P 8 -I{} bash -c "echo {} \$(dd if=$S/$f bs=4M skip=\$(({}*256)) count=256 status=none | sha256sum | cut -c1-16)" | sort -n > $D/sha.local
ssh -o BatchMode=yes -o ServerAliveInterval=30 exabox-login \
  "for i in \$(seq 0 $((n-1))); do echo \$i \$(dd if=$R/$f bs=4M skip=\$((i*256)) count=256 status=none | sha256sum | cut -c1-16); done" | sort -n > $D/sha.remote
[ $(wc -l < $D/sha.remote) = $n ]
bad=$(join $D/sha.local $D/sha.remote | awk '$2!=$3{print $1}')
echo "$(date -u +%T) bad chunks: ${bad:-none}"
for i in $bad; do sed -i "\|^$f $i\$|d" $D/pcopy.done; done
bash $D/pcopy.sh
