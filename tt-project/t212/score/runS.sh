#!/bin/bash
# t212 score job (blx01 broker): #214's decode_ref.py (production options, host noise, yuv out) on the
# #214 seed latents, with the t212 port (tree b, own build), so the yuv compares byte-for-byte with ref/.
set -o pipefail
F=/var/tmp/fasth3; T=$F/t212; W=$T/b; D=$F/diffvae; O=$T/score/out
mkdir -p $O $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface
source $F/t48/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools HF_HUB_OFFLINE=1
export TT_METAL_CACHE=$F/cache/tt-metal-cache
export DIFFVAE_CHECKPOINT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941/vae/ltx-2.5-video-vae-bf16.safetensors
L=$O/run.log
echo "[t212S] host=$(hostname) build=$(git -C $W rev-parse --short=11 HEAD) $(date -u '+%F %T') UTC" | tee -a $L
cd $O
T0=$(date +%s)
timeout ${PY_S:-560} python -u $D/scripts/decode_ref.py $D/latents $O 2>&1 | tee -a $L
rc=${PIPESTATUS[0]}
echo "[t212S] process wall $(( $(date +%s) - T0 )) s, jit compiles $(grep -c 'BuildKernels | compiled' $L)" | tee -a $L
echo "T212S_EXIT=$rc" | tee -a $L
exit $rc
