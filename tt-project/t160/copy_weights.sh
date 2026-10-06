#!/bin/bash
set -eo pipefail
S=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941
D=exabox-login:/data/smarton/fasth3/models/ltx-2.5
for f in vae/ltx-2.5-audio-vae-bf16.safetensors latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors; do
  rsync -L --times --partial --inplace "$S/$f" "$D/$f"
  echo "copied $f rc=$?"
done
ssh -o BatchMode=yes exabox-login 'ls -lL /data/smarton/fasth3/models/ltx-2.5/*/'
echo COPY_OK
