#!/bin/bash
set -e
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$SD/campaign_paths.sh"; cd $DD   # DD is the data directory in both layouts (campaign_paths.sh)
export SDPA_CAUSAL=1 SDPA_FIDELITY=HiFi4 SDPA_EXP_APPROX=0 SDPA_MATH_APPROX=0 SDPA_FP32_ACC=1 SDPA_PACKER_L1_ACC=1 SDPA_DTYPE=bfp8_b SDPA_KV_DTYPE=bfp8_b SDPA_SEQ=4096 SDPA_QCHUNK=256 SDPA_KCHUNK=256
$SD/set_zone_config.sh 1 0 0 0 0 >/dev/null
SDPA_GRID=8x8 $SD/run_zone.sh t24_prod_causal_S4096_q256k256_g64_zon
SDPA_GRID=8x8 $SD/run_zone.sh t24_prod_causal_S4096_q256k256_g64_zon_mp mp
$SD/run_zone.sh t24_prod_causal_S4096_q256k256_g110_zon
SDPA_GRID=8x8 SDPA_FP32_ACC=0 $SD/run_zone.sh t24_prod_causal_S4096_q256k256_g64_fp32off_zon
$SD/set_zone_config.sh 0 0 0 0 0 >/dev/null
echo "### T2.4 zon done $(date -u +%H:%M:%SZ)"
