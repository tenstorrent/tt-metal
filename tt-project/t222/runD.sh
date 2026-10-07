#!/bin/bash
# t222 job D (blx01 broker): decode the #214 latents with DIFFVAE_S5_2D=1 (yuv, host noise), then one profiled decode.
set -o pipefail
source /var/tmp/fasth3/t219/drv/common.sh
D=$F/diffvae; O=$T/out; mkdir -p $O; L=$O/run.log
export DIFFVAE_S5_2D=1
export DIFFVAE_CHECKPOINT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941/vae/ltx-2.5-video-vae-bf16.safetensors
echo "[t222D] host=$(hostname) DIFFVAE_S5_2D=$DIFFVAE_S5_2D $(date -u '+%F %T') UTC" | tee -a $L
cd $O
T0=$(date +%s)
timeout ${PY_S:-560} python -u $T/drv/decode222.py $D/latents $O 2>&1 | tee -a $L
rc=${PIPESTATUS[0]}
echo "[t222D] process wall $(( $(date +%s) - T0 )) s, jit compiles $(grep -c 'BuildKernels | compiled' $L)" | tee -a $L
echo "T222_EXIT=$rc" | tee -a $L
exit $rc
