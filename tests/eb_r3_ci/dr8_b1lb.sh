#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58724 third review): deepseek_v3_b1's fused MoE and MLP with reduce_to_one on a 4x2 mesh (moe_kernel.cpp
# under ENABLE_REDUCE_TO_ONE), Blackhole LoudBox, the kernel with and without ELTWISE_BINARY_PER_TILE_HANDOFF_DEST_REUSE: outcomes and
# output bits, the ELFs, then device time (main optin optin main). Slow dispatch for the 13x10 worker grid unless EB_FAST is set.
cd /work
[[ -z ${EB_FAST:-} ]] && export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_ALLOCATOR_MODE_HYBRID=1 TT_METAL_RUNTIME_ROOT=/work EB_SHOW_ERR=1
B1=models/demos/deepseek_v3_b1/tests/unit_tests
OPT=tests/eb_r3_ci/optin_b1dr.txt
K="(test_moe_fused_with_reduce and full_groups and t8_partial) or (test_mlp_with_reduce and not half and not 7sram and not all-dram)"
echo "##### collect"; python3 -m pytest --collect-only -q $B1/test_moe_mlp.py -k "with_reduce" 2>&1 | grep -E "::" | head -40
echo "##### bits"; EB_RUN_LIMIT=3600 bash tests/eb_r3_ci/bits_ab.sh $OPT -p eb_seed_plugin $B1/test_moe_mlp.py -k "$K"
echo "##### elf"; python3 tests/eb_r3_ci/elf_set_diff.py /tmp/ebbits/cache_main /tmp/ebbits/cache_optin "^moe_kernel$" 2>&1 | head -20
echo "##### time"; EB_RUN_LIMIT=3600 bash tests/eb_r3_ci/ab_set.sh $OPT $B1/test_moe_mlp.py -k "$K"
