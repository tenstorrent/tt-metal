#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58723 review): Blackhole Galaxy again: the all-gather minimal matmul's fused addcmul (tt_dit's
# bh4x8links2 gate test) and the strided matmul reduce-scatter with tt_dit's fused addcmul (the Wan Galaxy case); device time
# A/B with the opt-in, then bits.
cd /work
export EB_SHOW_ERR=1 EB_RUN_LIMIT=1500
AG=models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py
echo "##### strided reduce-scatter addcmul"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_srs.txt tests/eb_r3_ci/test_eb_srs_addcmul.py
echo "##### agmm addcmul gate"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_agmm.txt $AG -k "test_linear_addcmul_gate"
echo "##### bits strided reduce-scatter addcmul"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_srs.txt -p eb_seed_plugin tests/eb_r3_ci/test_eb_srs_addcmul.py
echo "##### bits agmm addcmul gate"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_agmm.txt -p eb_seed_plugin $AG -k "test_linear_addcmul_gate"
