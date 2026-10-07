#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58724 third review): deepseek_v3_b1's single-device fused tests whose kernels the dest-reuse opt-in
# reaches (moe_routed_expert without reduce, the dense MLP), P150 under slow dispatch (their 13x10 worker grid), the kernels with
# and without ELTWISE_BINARY_PER_TILE_HANDOFF_DEST_REUSE: outcomes and output bits, then device time (main optin optin main).
cd /work
export TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_ALLOCATOR_MODE_HYBRID=1 TT_METAL_RUNTIME_ROOT=/work EB_SHOW_ERR=1
B1=models/demos/deepseek_v3_b1/tests/unit_tests
OPT=tests/eb_r3_ci/optin_b1dr.txt
echo "##### collect"; python3 -m pytest --collect-only -q $B1/test_moe_routed_expert.py $B1/test_moe_mlp.py 2>&1 | grep -E "::" | head -40
echo "##### bits moe_routed_expert"; EB_RUN_LIMIT=1800 bash tests/eb_r3_ci/bits_ab.sh $OPT -p eb_seed_plugin $B1/test_moe_routed_expert.py -k "test_moe_routed_expert and not with_reduce"
echo "##### elf moe_routed_expert"; python3 tests/eb_r3_ci/elf_set_diff.py /tmp/ebbits/cache_main /tmp/ebbits/cache_optin "^moe_routed_expert_kernel$" 2>&1 | head -20
echo "##### bits mlp and moe fused"; EB_RUN_LIMIT=2400 bash tests/eb_r3_ci/bits_ab.sh $OPT -p eb_seed_plugin $B1/test_moe_mlp.py -k "(test_mlp or test_moe_fused) and not with_reduce"
echo "##### elf moe"; python3 tests/eb_r3_ci/elf_set_diff.py /tmp/ebbits/cache_main /tmp/ebbits/cache_optin "^moe_kernel$" 2>&1 | head -20
echo "##### time moe_routed_expert"; EB_RUN_LIMIT=1800 bash tests/eb_r3_ci/ab_set.sh $OPT $B1/test_moe_routed_expert.py -k "test_moe_routed_expert and not with_reduce"
