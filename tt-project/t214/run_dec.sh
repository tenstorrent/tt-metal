#!/bin/bash
# Job B (blx01 broker): decode_ref.py on the saved latents. Same tree and env as job A.
set -o pipefail
F=/var/tmp/fasth3; W=$F/t48; O=$F/t208/tree; D=$F/diffvae
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface
source $W/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$O:$W:$W/ttnn:$W/tools HF_HUB_OFFLINE=1
export TT_METAL_CACHE=$F/cache/tt-metal-cache
export DIFFVAE_CHECKPOINT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941/vae/ltx-2.5-video-vae-bf16.safetensors
mkdir -p $D/ref
L=$D/ref/run_dec.$(date -u +%H%M%S).log
echo "[t214] job B host=$(hostname) build=$(git -C $W rev-parse --short=11 HEAD) py=$(cat $O/OVERLAY_COMMIT) $(date -u '+%F %T')" | tee $L
cd $D/ref
T0=$(date +%s)
timeout ${PY_S:-560} python -u $D/scripts/decode_ref.py $D/latents $D/ref 2>&1 | tee -a $L
rc=${PIPESTATUS[0]}
echo "[t214] process wall $(( $(date +%s) - T0 )) s, jit compiles $(grep -c 'BuildKernels | compiled' $L)" | tee -a $L
echo "T214B_EXIT=$rc" | tee -a $L
exit $rc
