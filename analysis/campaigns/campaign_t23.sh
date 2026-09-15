#!/bin/bash
# T2.3 DRAM-law runs at the causal anchor (zones off, 3 iterations each), requested by the coordinator after T2.1.
set -e
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$SD/campaign_paths.sh"   # DD is the data directory in both layouts (campaign_paths.sh)
cd $DD
$SD/set_zone_config.sh 0 0 0 0 0 >/dev/null
SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 SDPA_KV_DTYPE=bfloat16 $SD/run_zone.sh t23_causal_q128k128_kvbf16_zoff
SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 SDPA_NKV=32 $SD/run_zone.sh t23_causal_q128k128_nkv32_zoff
SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 SDPA_NKV=1 $SD/run_zone.sh t23_causal_q128k128_nkv1_zoff
SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 SDPA_GRID=8x8 $SD/run_zone.sh t23_causal_q128k128_grid8x8_zoff
SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 SDPA_DTYPE=bfloat16 $SD/run_zone.sh t23_causal_q128k128_allbf16_zoff
echo "### T2.3 campaign done $(date -u +%H:%M:%SZ)"
