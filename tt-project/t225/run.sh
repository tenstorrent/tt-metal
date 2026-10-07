#!/bin/bash
# t225 broker job (blx01): production decode (device stage-5 noise) of the #214 latents, seeds 0-4,
# arm 1d (default) then arm 2d (DIFFVAE_S5_2D=1), each its own process. Saves yuv per arm and seed.
set -o pipefail
source /var/tmp/fasth3/t225/drv/common.sh
O=$T/out; mkdir -p $O; L=$O/run.log
export DIFFVAE_CHECKPOINT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941/vae/ltx-2.5-video-vae-bf16.safetensors
cd $O
rc=0
for arm in 1d 2d; do
  if [ $arm = 2d ]; then export DIFFVAE_S5_2D=1; else unset DIFFVAE_S5_2D; fi
  echo "[t225] arm=$arm host=$(hostname) DIFFVAE_S5_2D=${DIFFVAE_S5_2D:-0} attempt=${T225_ATTEMPT:-?} $(date -u '+%F %T') UTC" | tee -a $L
  T0=$(date +%s)
  ARM=$arm timeout ${PY_S:-140} python -u $T/drv/decode225.py $F/diffvae/latents $O 2>&1 | tee -a $L
  r=${PIPESTATUS[0]}
  echo "[t225] arm=$arm rc=$r process wall $(( $(date +%s) - T0 )) s" | tee -a $L
  [ $r = 0 ] || rc=$r
done
echo "T225_EXIT=$rc" | tee -a $L
exit $rc
