#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58723, #58724 review): on a Blackhole Galaxy: deepseek_v3_b1's moe and moe_routed_expert fused ops
# (single device, 13x10 worker grid under slow dispatch; their dest-reuse multiply) and the strided all-gather minimal matmul
# with its fused addcmul (8x4); device time A/B with the opt-in, then bits.
cd /work
B1=models/demos/deepseek_v3_b1/tests/unit_tests
SG=tests/ttnn/unit_tests/operations/ccl/blackhole_CI/galaxy/galaxy_nightly/test_strided_all_gather_minimal_matmul_async_bh.py
echo "##### strided agmm addcmul"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_sagmm.txt $SG -k "addcmul"
echo "##### bits strided agmm"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_sagmm.txt -p eb_seed_plugin $SG -k "addcmul"
export TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_ALLOCATOR_MODE_HYBRID=1 PYTHONPATH=/work/ttnn:/work/tools:${PYTHONPATH:-}
echo "##### b1 moe_routed_expert"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_b1mre.txt $B1/test_moe_routed_expert.py -k "test_moe_routed_expert and not with_reduce"
echo "##### b1 moe fused"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_b1moe.txt $B1/test_moe_mlp.py -k "test_moe_fused and not with_reduce"
echo "##### bits b1"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_b1mre.txt -p eb_seed_plugin $B1/test_moe_routed_expert.py -k "test_moe_routed_expert and not with_reduce"
bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_b1moe.txt -p eb_seed_plugin $B1/test_moe_mlp.py -k "test_moe_fused and not with_reduce"
