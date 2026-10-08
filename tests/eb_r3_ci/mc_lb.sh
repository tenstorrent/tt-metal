#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58723, #58724 review): the multi-chip callers a Blackhole LoudBox (8 x P150, 2x4) runs: the all-gather
# minimal matmul with its fused addcmul, attn_res_gather_softmax, zero_padded_kv_cache and deepseek_v3_b1's reduce_to_one
# (dest-reuse add); device time A/B with their opt-in, then bits.
cd /work
AG=models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py
AR=tests/ttnn/unit_tests/operations/experimental/test_attn_res_gather_softmax.py
ZP=models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_zero_padded_kv_cache.py
R1=models/demos/deepseek_v3_b1/tests/unit_tests/test_reduce_to_one_b1.py
echo "##### agmm"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_agmm.txt $AG -k "2x4links1 and denseattn1"
echo "##### attn_res_gather_softmax"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_args.txt $AR -k "matches_torch and 2x4"
echo "##### zero_padded_kv_cache"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_zpkv.txt $ZP -k "2x4"
echo "##### reduce_to_one"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_r21.txt $R1
echo "##### bits agmm"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_agmm.txt -p eb_seed_plugin $AG -k "2x4links1 and denseattn1"
echo "##### bits args"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_args.txt -p eb_seed_plugin $AR -k "2x4"
echo "##### bits zpkv"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_zpkv.txt -p eb_seed_plugin $ZP -k "2x4"
echo "##### bits reduce_to_one"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_r21.txt -p eb_seed_plugin $R1
