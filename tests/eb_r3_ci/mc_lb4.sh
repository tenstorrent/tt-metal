#!/usr/bin/env bash
# Round 3 eltwise binary (#58723 review): LoudBox on the CI branch merged with current main (zero_padded_kv_cache's dataflow
# kernels include a header the older base lacks, run 37573070751): its opt-in A/B and bits.
cd /work
export EB_SHOW_ERR=1 EB_RUN_LIMIT=900
ZP=models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_zero_padded_kv_cache.py
echo "##### zero_padded_kv_cache"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_zpkv.txt $ZP -k "2x4"
echo "##### bits zpkv"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_zpkv.txt -p eb_seed_plugin $ZP -k "2x4"
