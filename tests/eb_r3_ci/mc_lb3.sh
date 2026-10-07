#!/usr/bin/env bash
# Round 3 eltwise binary (#58723, #58724 review): LoudBox, third pass: attn_res_gather_softmax (its direct init passing the
# hand-off) A/B and bits first; then reduce_to_one's bits without the profiler (under it the 1D test passes and then outlives
# a 1500 s limit, run 37573070751).
cd /work
export EB_SHOW_ERR=1 EB_RUN_LIMIT=900
AR=tests/ttnn/unit_tests/operations/experimental/test_attn_res_gather_softmax.py
R1=models/demos/deepseek_v3_b1/tests/unit_tests/test_reduce_to_one_b1.py
echo "##### attn_res_gather_softmax"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_args.txt $AR -k "matches_torch and 2x4"
echo "##### bits args"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_args.txt -p eb_seed_plugin $AR -k "matches_torch and 2x4"
echo "##### bits reduce_to_one 1d"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_r21.txt -p eb_seed_plugin $R1 -k "test_reduce_to_one_1d"
