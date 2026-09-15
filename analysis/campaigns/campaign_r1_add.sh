#!/bin/bash
# R1b addendum: MLA decode in the model's non-paged form (b8 nh16 nkv1 kv_lora512 d_rope64 KV bf16, cur_pos = cache-1) at cache
# 1024/4096/8192 via analysis/mla_decode_latency_sweep.py (3 warm-ups + MLAD_ITERS 1 = 4 invocations, first discarded), and the
# paged test geometry (nh128 kv_lora512 d_rope64, Q sharded on 64 cores, K bfp8 block 64) at batch 8.
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$SD/campaign_paths.sh"; cd $DD   # DD is the data directory in both layouts (campaign_paths.sh)
$SD/set_zone_config.sh 0 0 0 0 0 >/dev/null
MLAD_B=8 MLAD_NH=16 MLAD_KVLORA=512 MLAD_DROPE=64 MLAD_SEQS=1024,4096,8192 MLAD_ITERS=1 $SD/run_regime.sh r1b_mla_decode_nonpaged_b8_nh16_kvbf16_cache1024_4096_8192_zoff plain "analysis/mla_decode_latency_sweep.py::test_mla_decode_latency" || true
MLAD_POS=1024,4096,8192 MLAD_CACHE=16384 MLAD_B=8 MLAD_NH=128 MLAD_KVLORA=512 MLAD_DROPE=64 MLAD_QCORES=64 MLAD_ITERS=3 $SD/run_regime.sh r1b_mla_decode_paged_b8_nh128_pos1024_4096_8192_zoff plain "analysis/r1_mla_decode.py::test_r1_mla_decode" || true
echo "### R1 addendum done $(date -u +%FT%TZ)"
