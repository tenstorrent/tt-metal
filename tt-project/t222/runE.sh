#!/bin/bash
# t222 job E (blx01 broker): production-path A/B, device stage-5 noise. Arm 1d (C1 default) then arm 2d (DIFFVAE_S5_2D=1),
# each its own process: SEEDS timed decodes after a warm-up, then one profiled decode (stage tree).
set -o pipefail
source /var/tmp/fasth3/t219/drv/common.sh
D=$F/diffvae; O=$T/outE; mkdir -p $O; L=$O/run.log
export DIFFVAE_CHECKPOINT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941/vae/ltx-2.5-video-vae-bf16.safetensors
export SEEDS=0,1,2
cd $O
rc=0
for arm in 1d 2d; do
  if [ $arm = 2d ]; then export DIFFVAE_S5_2D=1; else unset DIFFVAE_S5_2D; fi
  echo "[t222E] arm=$arm host=$(hostname) DIFFVAE_S5_2D=${DIFFVAE_S5_2D:-0} $(date -u '+%F %T') UTC" | tee -a $L
  T0=$(date +%s)
  ARM=$arm timeout 230 python -u $T/drv/decodeE.py $D/latents $O 2>&1 | tee -a $L
  r=${PIPESTATUS[0]}
  echo "[t222E] arm=$arm rc=$r process wall $(( $(date +%s) - T0 )) s" | tee -a $L
  [ $r = 0 ] || rc=$r
done
echo "T222_EXIT=$rc" | tee -a $L
exit $rc
