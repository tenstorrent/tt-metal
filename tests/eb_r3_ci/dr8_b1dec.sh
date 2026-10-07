#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58724 third review): deepseek_v3_b1's decoder block with reduce_to_one (decoder_block_kernel.cpp under
# ENABLE_REDUCE_TO_ONE, a 4x2 submesh) on a Blackhole Galaxy under slow dispatch, the kernel with and without
# ELTWISE_BINARY_PER_TILE_HANDOFF_DEST_REUSE: outcomes, output bits and the ELFs, positions 0 and 4096 of the light rigged case.
cd /work
export TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_ALLOCATOR_MODE_HYBRID=1 TT_METAL_RUNTIME_ROOT=/work EB_SHOW_ERR=1
B1=models/demos/deepseek_v3_b1/tests/unit_tests
OPT=tests/eb_r3_ci/optin_b1dr.txt
K="rigged_groups1 and t8_seven_picked and sram_bspm_off and random_weights and just_decoder_mla and just_decoder_moe and not mtp and (device_params0-0-32768 or device_params0-4096-32768)"
echo "##### collect"; python3 -m pytest --collect-only -q $B1/test_decoder_block.py -k "$K" 2>&1 | grep "::" | head
echo "##### bits decoder"; EB_RUN_LIMIT=3000 bash tests/eb_r3_ci/bits_ab.sh $OPT -p eb_seed_plugin -rs --timeout=0 $B1/test_decoder_block.py -k "$K"
grep -E "SKIPPED|PASSED|FAILED|^E  " /tmp/ebbits/log_main.txt | cut -c1-300 | head -10
grep -E "SKIPPED|PASSED|FAILED|^E  " /tmp/ebbits/log_optin.txt | cut -c1-300 | head -10
echo "##### elf decoder"; python3 tests/eb_r3_ci/elf_set_diff.py /tmp/ebbits/cache_main /tmp/ebbits/cache_optin "^decoder_block_kernel$" 2>&1 | head -20
