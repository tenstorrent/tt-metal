#!/bin/bash
# R1f: perf-counter multipass captures on the unmodified kernel (zones compiled out), calibration checkout, one capture each.
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$SD/campaign_paths.sh"; cd $DD   # DD is the data directory in both layouts (campaign_paths.sh)
$SD/set_zone_config.sh 0 0 0 0 0 >/dev/null
A="SDPA_SEQ=4096"
for QK in 128:128 64:128 256:128 128:256 128:512 512:128 512:512; do QC=${QK%%:*}; KC=${QK##*:}
  for MODE in causal noncausal; do C=1; [ $MODE = noncausal ] && C=0
    env $A SDPA_CAUSAL=$C SDPA_QCHUNK=$QC SDPA_KCHUNK=$KC $SD/run_zone.sh r1f_${MODE}_q${QC}k${KC}_zoff_mp mp || true
  done
done
env $A SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 SDPA_FIDELITY=LoFi $SD/run_zone.sh r1f_causal_q128k128_lofi_zoff_mp mp || true
env $A SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 SDPA_FIDELITY=HiFi3 $SD/run_zone.sh r1f_causal_q128k128_hifi3_zoff_mp mp || true
env $A SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 SDPA_EXP_APPROX=0 $SD/run_zone.sh r1f_causal_q128k128_expaccurate_zoff_mp mp || true
env $A SDPA_CAUSAL=1 SDPA_FIDELITY=HiFi4 SDPA_EXP_APPROX=0 SDPA_MATH_APPROX=0 SDPA_FP32_ACC=1 SDPA_PACKER_L1_ACC=1 SDPA_QCHUNK=256 SDPA_KCHUNK=256 $SD/run_zone.sh r1f_prod_causal_S4096_q256k256_g110_zoff_mp mp || true
env $A SDPA_CAUSAL=1 SDPA_FIDELITY=HiFi4 SDPA_EXP_APPROX=0 SDPA_MATH_APPROX=0 SDPA_FP32_ACC=0 SDPA_PACKER_L1_ACC=1 SDPA_GRID=8x8 SDPA_QCHUNK=256 SDPA_KCHUNK=256 $SD/run_zone.sh r1f_a10_causal_S4096_q256k256_g64_fp32off_zoff_mp mp || true
echo "### R1f done $(date -u +%FT%TZ)"
