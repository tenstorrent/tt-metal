#!/bin/bash
# T2.8 decode calibration sweep (zones off, 3 iterations per position).
set -e
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$SD/campaign_paths.sh"; cd $DD   # DD is the data directory in both layouts (campaign_paths.sh)
$SD/set_zone_config.sh 0 0 0 0 0 >/dev/null
DEC_B=32 DEC_POS=128,512,1024,2048,4096,8192 DEC_MAXSEQ=16384 DEC_GRID=8x8 $SD/run_decode.sh t28_decode_b32_g64_pos128to8192
DEC_B=32 DEC_POS=1024,4096 DEC_MAXSEQ=16384 DEC_GRID=11x10 $SD/run_decode.sh t28_decode_b32_g110_pos1024_4096
DEC_B=8 DEC_POS=1024 DEC_MAXSEQ=16384 DEC_GRID=8x8 $SD/run_decode.sh t28_decode_b8_g64_pos1024
DEC_B=16 DEC_POS=1024 DEC_MAXSEQ=16384 DEC_GRID=8x8 $SD/run_decode.sh t28_decode_b16_g64_pos1024
DEC_B=32 DEC_POS=1024 DEC_MAXSEQ=16384 DEC_GRID=8x8 DEC_KV_DTYPE=bfloat16 $SD/run_decode.sh t28_decode_b32_g64_pos1024_kvbf16
DEC_B=32 DEC_POS=1024 DEC_MAXSEQ=16384 DEC_GRID=8x8 $SD/run_decode.sh t28_decode_b32_g64_pos1024_mp mp
echo "### T2.8 done $(date -u +%H:%M:%SZ)"
