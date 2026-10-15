#!/bin/bash
# t503: stage the LTX-2.5 A/B on blx03 (non-device). Weights -> /var/tmp/fasth3/t503 (throttled), code+build -> ~/fasth3/t503.
set -eo pipefail
H=g14blx03; S=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941
NEW=/home/smarton/fasth3/data/t498/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
P=/home/smarton/fasth3/tt-metal/tt-project; R=/var/tmp/fasth3/t503
RS=(ionice -c3 nice -n19 rsync -a --bwlimit=100M --rsync-path="ionice -c3 nice -n19 rsync")
use=$(ssh $H "df --output=pcent / | tail -1" | tr -dc 0-9); [ "$use" -le 70 ] || { echo "blx03 / at $use%"; exit 5; }
ssh $H "mkdir -p $R/ltx25/text_encoders $R/ltx25/diffusion_models $R/ltx25/vae $R/ltx25/latent_upscale_models $R/te0817/text_encoders $R/cache $R/jit $R/tmp ~/fasth3/t503/tt-metal"
echo "code $(date -u +%T)"
"${RS[@]}" --exclude=.git --exclude=.cpmcache --exclude=tt-project --exclude=build_Release/CMakeFiles --exclude=build_Release/Testing --include='/runtime/**' --exclude='*.o' $P/worktrees/t313/ $H:fasth3/t503/tt-metal/
git -C $P/worktrees/t503 ls-files -z -- . ':!tt-project' ':!NOTES.md' ':!tmp' > /tmp/t503.files
"${RS[@]}" --from0 --files-from=/tmp/t503.files $P/worktrees/t503/ $H:fasth3/t503/tt-metal/
git -C $P/worktrees/t503 rev-parse HEAD | ssh $H "cat > ~/fasth3/t503/tt-metal/T503_COMMIT"
for f in vae/ltx-2.5-audio-vae-bf16.safetensors latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors; do
  echo "$f $(date -u +%T)"; "${RS[@]}" -L $S/$f $H:$R/ltx25/$f; done
ssh $H "ln -sfn /var/tmp/fasth3/ltx25_vae/ltx-2.5-video-vae-conv-bf16.safetensors $R/ltx25/vae/ltx-2.5-video-vae-conv-bf16.safetensors"
echo "new $(date -u +%T)"; "${RS[@]}" $NEW $H:$R/te0817/text_encoders/
ssh $H "cd $R && ionice -c3 nice sha256sum te0817/text_encoders/*.safetensors && stat -c '%s %n' ltx25/*/* te0817/*/*" | tee /dev/stderr | grep ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1 >/dev/null
echo "STAGE503 OK $(date -u +%T)"
