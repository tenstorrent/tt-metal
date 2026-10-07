#!/usr/bin/env bash
# Round 3 eltwise binary (#58723 review, twin round): Blackhole Galaxy, the strided matmul reduce-scatter with tt_dit's fused
# addcmul (Wan Galaxy case, broadcast and full gate, both cluster axes): device time A/B with the opt-in, three passes of
# main optin optin main, then bits.
cd /work
export EB_SHOW_ERR=1 EB_RUN_LIMIT=1500 EB_REPS=3
for p in 1 2 3; do
  echo "##### srs addcmul pass $p"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_srs.txt tests/eb_r3_ci/test_eb_srs_addcmul.py
done
echo "##### bits srs addcmul"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_srs.txt -p eb_seed_plugin tests/eb_r3_ci/test_eb_srs_addcmul.py
