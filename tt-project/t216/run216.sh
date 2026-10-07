#!/bin/bash
# #216 broker job on blx01: ab216.py on the t212 port build (= t48 a40d78b8bae), both stride arms.
set -o pipefail
F=/var/tmp/fasth3; W=$F/t212/b; D=$F/t216; mkdir -p $D/out $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface
source $F/t48/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools HF_HUB_OFFLINE=1
export TT_METAL_CACHE=$F/cache/tt-metal-cache
export DIFFVAE_CHECKPOINT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941/vae/ltx-2.5-video-vae-bf16.safetensors
L=$D/out/run.$(date -u +%H%M%S).log
echo "[t216] host=$(hostname) build=$(git -C $W rev-parse --short=11 HEAD) attempt=${T216_ATTEMPT:-?} $(date -u '+%F %T') UTC" | tee $L
cd $D/out
T0=$(date +%s)
timeout ${PY_S:-560} python -u $D/ab216.py $F/diffvae/latents $D/out 2>&1 | tee -a $L
rc=${PIPESTATUS[0]}
echo "[t216] process wall $(( $(date +%s) - T0 )) s, jit compiles $(grep -c 'BuildKernels | compiled' $L)" | tee -a $L
echo "T216_EXIT=$rc" | tee -a $L
exit $rc
