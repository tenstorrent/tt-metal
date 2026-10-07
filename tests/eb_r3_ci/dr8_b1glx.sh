#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58724 third review): deepseek_v3_b1's fused MoE and MLP with reduce_to_one (a 4x2 submesh) and
# moe_routed_expert (single device) on a Blackhole Galaxy, whose 13x10 worker grid they require, under slow dispatch: the kernels
# with and without ELTWISE_BINARY_PER_TILE_HANDOFF_DEST_REUSE, outcomes, output bits and the ELFs; skip reasons are printed.
cd /work
export TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_ALLOCATOR_MODE_HYBRID=1 TT_METAL_RUNTIME_ROOT=/work EB_SHOW_ERR=1
B1=models/demos/deepseek_v3_b1/tests/unit_tests
OPT=tests/eb_r3_ci/optin_b1dr.txt
K="(test_moe_fused_with_reduce and full_groups and t8_partial) or (test_mlp_with_reduce and not half and not 7sram and not all-dram)"
echo "##### bits moe and mlp with reduce"; EB_RUN_LIMIT=3600 bash tests/eb_r3_ci/bits_ab.sh $OPT -p eb_seed_plugin -rs $B1/test_moe_mlp.py -k "$K"
grep -E "SKIPPED|PASSED|FAILED" /tmp/ebbits/log_main.txt | cut -c1-300 | head -10
echo "##### elf moe"; python3 tests/eb_r3_ci/elf_set_diff.py /tmp/ebbits/cache_main /tmp/ebbits/cache_optin "^moe_kernel$" 2>&1 | head -20
echo "##### bits moe_routed_expert"; EB_RUN_LIMIT=1800 bash tests/eb_r3_ci/bits_ab.sh $OPT -p eb_seed_plugin -rs $B1/test_moe_routed_expert.py -k "test_moe_routed_expert and not with_reduce"
grep -E "SKIPPED|PASSED|FAILED" /tmp/ebbits/log_main.txt | cut -c1-300 | head -10
echo "##### elf moe_routed_expert"; python3 tests/eb_r3_ci/elf_set_diff.py /tmp/ebbits/cache_main /tmp/ebbits/cache_optin "^moe_routed_expert_kernel$" 2>&1 | head -20
