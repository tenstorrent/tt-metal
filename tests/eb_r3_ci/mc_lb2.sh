#!/usr/bin/env bash
# Round 3 eltwise binary (#58723, #58724 review): LoudBox (8 x P150, 2x4) again. attn_res_gather_softmax's direct math init now
# passes the hand-off its broadcast multiply executes with (the first pass hung with the opt-in); every fabric test in its own
# process; zero_padded_kv_cache and reduce_to_one first, the all-gather minimal matmul's fused addcmul last.
cd /work
export EB_SHOW_ERR=1 EB_RUN_LIMIT=1500
AG=models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py
AR=tests/ttnn/unit_tests/operations/experimental/test_attn_res_gather_softmax.py
ZP=models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_zero_padded_kv_cache.py
R1=models/demos/deepseek_v3_b1/tests/unit_tests/test_reduce_to_one_b1.py
echo "##### zero_padded_kv_cache"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_zpkv.txt $ZP -k "2x4"
for t in test_reduce_to_one_1d test_reduce_to_one_2d test_reduce_to_one_trace; do
  echo "##### reduce_to_one $t"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_r21.txt $R1 -k "$t"
done
echo "##### attn_res_gather_softmax"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_args.txt $AR -k "matches_torch and 2x4"
echo "##### bits zpkv"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_zpkv.txt -p eb_seed_plugin $ZP -k "2x4"
for t in test_reduce_to_one_1d test_reduce_to_one_2d test_reduce_to_one_trace; do
  echo "##### bits reduce_to_one $t"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_r21.txt -p eb_seed_plugin $R1 -k "$t"
done
echo "##### bits args"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_args.txt -p eb_seed_plugin $AR -k "matches_torch and 2x4"
echo "##### agmm fused addcmul"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_agmm.txt $AG -k "test_linear and perf and fused and 1xdenseattn1 and 2x4links1 and not separate"
