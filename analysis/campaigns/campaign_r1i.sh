#!/bin/bash
# R1i: the cores-per-group axis of paged MLA decode. Everything is held at the R1h geometry (kv_lora 512,
# d_rope 64, cache 4096, position 1024, Q sharded on 64 cores, K bfp8 in 64-row pages, reuse_k) and only
# max_cores_per_head_batch moves, so the attended bytes and the slice count are identical inside each family
# and only the number of cores that read for one head group changes.
#   family A, batch 4 nh128: 16 head groups, cores per group 1 / 2 / 3 / 4 / 6 (6 is the R1h point, a repeat)
#   family B, batch 4 nh32:   4 head groups, cores per group 4 / 8 / 27 (16 is the R1h point)
# Three invocations per run, the first discarded, zones off.
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$SD/campaign_paths.sh"; cd $DD
$SD/set_zone_config.sh 0 0 0 0 0 >/dev/null
COMMON="MLAD_POS=1024 MLAD_CACHE=4096 MLAD_KVLORA=512 MLAD_DROPE=64 MLAD_QCORES=64 MLAD_ITERS=3"
T="analysis/r1_mla_decode.py::test_r1_mla_decode"
for M in 1 2 3 4 6; do
  env $COMMON MLAD_B=4 MLAD_NH=128 MLAD_MCPHB=$M $SD/run_regime.sh r1i_mla_decode_paged_b4_nh128_mcphb${M}_zoff plain "$T" || true
done
for M in 4 8 27; do
  env $COMMON MLAD_B=4 MLAD_NH=32 MLAD_MCPHB=$M $SD/run_regime.sh r1i_mla_decode_paged_b4_nh32_mcphb${M}_zoff plain "$T" || true
done
echo "### R1i done $(date -u +%FT%TZ)"
