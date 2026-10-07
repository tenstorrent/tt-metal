#!/bin/bash
# t232 job (blx01 broker): host-only planner tests, then DIFFVAE_S5_2D=1 key-phase A/B. Arm kp0
# (DIFFVAE_NA_KEY_PHASE=0, today's 2-D split) then kp1, each its own process: warm-up, SEEDS timed decodes
# (device noise), one deep-profiled decode, then host-noise yuv for seeds 0-4 (scored vs diffvae/ref afterwards).
set -o pipefail
source /var/tmp/fasth3/t232/drv/common.sh
D=$F/diffvae; O=$T/out; mkdir -p $O; L=$O/run.log
export DIFFVAE_CHECKPOINT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941/vae/ltx-2.5-video-vae-bf16.safetensors
export DIFFVAE_S5_2D=1 SEEDS=0,1 HOST_SEEDS=0,1,2,3,4
cd $B
echo "[t232] host=$(hostname) $(date -u '+%F %T') UTC tree=$(git rev-parse --short=11 HEAD)" | tee -a $L
timeout 30 python -u $T/drv/hosttests.py 2>&1 | tee -a $L
echo "[t232] hosttests rc=${PIPESTATUS[0]}" | tee -a $L
timeout 60 python -m pytest -q -p no:cacheprovider models/tt_dit/tests/unit/test_neighborhood_sdpa.py \
  -k 'refuses_stride_with_h_split or key_phase_pins_brick_and_gather or divides_h_shard' 2>&1 | tail -15 | tee -a $L
echo "[t232] pytest host-only rc=${PIPESTATUS[0]}" | tee -a $L
cd $O
rc=0
for arm in kp0 kp1; do
  if [ $arm = kp0 ]; then LIM=270; else LIM=200; fi
  echo "[t232] arm=$arm $(date -u '+%F %T') UTC" | tee -a $L
  T0=$(date +%s)
  ARM=$arm DIFFVAE_NA_KEY_PHASE=${arm#kp} timeout $LIM python -u $T/drv/decode232.py $D/latents $O 2>&1 | tee -a $L
  r=${PIPESTATUS[0]}
  echo "[t232] arm=$arm rc=$r process wall $(( $(date +%s) - T0 )) s" | tee -a $L
  [ $r = 0 ] || rc=$r
done
echo "T232_EXIT=$rc" | tee -a $L
exit $rc
