#!/bin/bash
# R1g: head_dim 64 hold-outs (coordinator request 2026-09-12). Calibration checkout, zones off, 3 invocations first discarded,
# 110 cores, nh32 nkv8 bfp8 HiFi2 exp approx, head_dim 64. Zones-on twin for the non-causal S2048 point (PACK lane composition).
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$SD/campaign_paths.sh"; cd $DD   # DD is the data directory in both layouts (campaign_paths.sh)
Z(){ $SD/set_zone_config.sh $1 0 0 0 0 >/dev/null; }
Z 0
SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 SDPA_SEQ=2048 SDPA_HEAD_DIM=64 $SD/run_zone.sh r1g_causal_S2048_q128k128_hd64_zoff || true
SDPA_CAUSAL=0 SDPA_QCHUNK=128 SDPA_KCHUNK=128 SDPA_SEQ=2048 SDPA_HEAD_DIM=64 $SD/run_zone.sh r1g_noncausal_S2048_q128k128_hd64_zoff || true
Z 1
SDPA_CAUSAL=0 SDPA_QCHUNK=128 SDPA_KCHUNK=128 SDPA_SEQ=2048 SDPA_HEAD_DIM=64 $SD/run_zone.sh r1g_noncausal_S2048_q128k128_hd64_zon || true
Z 0
SDPA_CAUSAL=1 SDPA_QCHUNK=256 SDPA_KCHUNK=128 SDPA_SEQ=4096 SDPA_HEAD_DIM=64 $SD/run_zone.sh r1g_causal_S4096_q256k128_hd64_zoff || true
SDPA_CAUSAL=0 SDPA_QCHUNK=128 SDPA_KCHUNK=256 SDPA_SEQ=4096 SDPA_HEAD_DIM=64 $SD/run_zone.sh r1g_noncausal_S4096_q128k256_hd64_zoff || true
Z 0
echo "### R1g done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
