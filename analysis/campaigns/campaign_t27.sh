#!/bin/bash
# T2.7 drift check on the fresh main-tip build: causal and non-causal anchors, production S4096 point (zones off).
set -e
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$SD/campaign_paths.sh"; cd $DD   # DD is the data directory in both layouts (campaign_paths.sh)
SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 $SD/run_fresh.sh t27_fresh_causal_q128k128_zoff
SDPA_CAUSAL=0 SDPA_QCHUNK=128 SDPA_KCHUNK=128 $SD/run_fresh.sh t27_fresh_noncausal_q128k128_zoff
SDPA_CAUSAL=1 SDPA_FIDELITY=HiFi4 SDPA_EXP_APPROX=0 SDPA_MATH_APPROX=0 SDPA_FP32_ACC=1 SDPA_PACKER_L1_ACC=1 SDPA_GRID=8x8 SDPA_SEQ=4096 SDPA_QCHUNK=256 SDPA_KCHUNK=256 $SD/run_fresh.sh t27_fresh_prod_causal_S4096_q256k256_g64_zoff
SDPA_CAUSAL=1 SDPA_FIDELITY=HiFi4 SDPA_EXP_APPROX=0 SDPA_MATH_APPROX=0 SDPA_FP32_ACC=1 SDPA_PACKER_L1_ACC=1 SDPA_GRID=8x8 SDPA_SEQ=1024 SDPA_QCHUNK=64 SDPA_KCHUNK=64 $SD/run_fresh.sh t27_fresh_prod_causal_S1024_q64k64_g64_zoff
echo "### T2.7 done $(date -u +%H:%M:%SZ)"
