#!/bin/bash
# T2.4 production Llama SDPA config, zones off: 64 cores, HiFi4, exp accurate, fp32 acc, packer_l1_acc, Q bf16, K/V per PROD_KV_DTYPE
set -e
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$SD/campaign_paths.sh"; cd $DD   # DD is the data directory in both layouts (campaign_paths.sh)
$SD/set_zone_config.sh 0 0 0 0 0 >/dev/null
export SDPA_CAUSAL=1 SDPA_FIDELITY=HiFi4 SDPA_EXP_APPROX=0 SDPA_MATH_APPROX=0 SDPA_FP32_ACC=1 SDPA_PACKER_L1_ACC=1 SDPA_DTYPE=${PROD_Q_DTYPE:-bfloat16} SDPA_KV_DTYPE=${PROD_KV_DTYPE:-bfloat8_b}
for S in 2048 4096 8192; do SDPA_GRID=8x8 SDPA_SEQ=$S SDPA_QCHUNK=256 SDPA_KCHUNK=256 $SD/run_zone.sh t24_prod_causal_S${S}_q256k256_g64_zoff; done
SDPA_GRID=8x8 SDPA_SEQ=1024 SDPA_QCHUNK=64 SDPA_KCHUNK=64 $SD/run_zone.sh t24_prod_causal_S1024_q64k64_g64_zoff
SDPA_GRID=8x8 SDPA_SEQ=4096 SDPA_QCHUNK=256 SDPA_KCHUNK=256 SDPA_FP32_ACC=0 $SD/run_zone.sh t24_prod_causal_S4096_q256k256_g64_fp32off_zoff
SDPA_SEQ=4096 SDPA_QCHUNK=256 SDPA_KCHUNK=256 $SD/run_zone.sh t24_prod_causal_S4096_q256k256_g110_zoff
echo "### T2.4 zoff done $(date -u +%H:%M:%SZ)"
