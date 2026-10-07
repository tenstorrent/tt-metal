#!/bin/bash
# t227 job F (blx01 broker): DIFFVAE_S5_2D=1, SDPA fidelity A/B. Arm hifi2 (today's default) then arm lofi, each its own
# process: warm-up, SEEDS timed decodes (device noise), one deep-profiled decode; the lofi arm also writes host-noise
# yuv for the 5 seeds (scored against diffvae/ref on the host afterwards).
set -o pipefail
source /var/tmp/fasth3/t227/drv/common.sh
D=$F/diffvae; O=$T/outF; mkdir -p $O; L=$O/run.log
export DIFFVAE_CHECKPOINT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941/vae/ltx-2.5-video-vae-bf16.safetensors
export DIFFVAE_S5_2D=1 SEEDS=0,1
cd $O
rc=0
for arm in hifi2 lofi; do
  if [ $arm = lofi ]; then export HOST_SEEDS=0,1,2,3,4 LIM=290; else unset HOST_SEEDS; LIM=170; fi
  echo "[t227F] arm=$arm host=$(hostname) $(date -u '+%F %T') UTC" | tee -a $L
  T0=$(date +%s)
  ARM=$arm DIFFVAE_NA_FIDELITY=$arm timeout $LIM python -u $T/drv/decode227.py $D/latents $O 2>&1 | tee -a $L
  r=${PIPESTATUS[0]}
  echo "[t227F] arm=$arm rc=$r process wall $(( $(date +%s) - T0 )) s" | tee -a $L
  [ $r = 0 ] || rc=$r
done
echo "T227_EXIT=$rc" | tee -a $L
exit $rc
