#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, seventh review: the per-tile dest-reuse callers (CSA compressor, welford group norm) with the LLK
# asserts on, on the PR head merged with main #59921 (the unpacker B check of DEST_TO_SRCA dest reuse). TT_METAL_LLK_ASSERTS
# turns them on for any value; without the watcher a tripped assert is an ebreak, so each test runs under a timeout.
cd /work
export PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-} TT_METAL_LLK_ASSERTS=1 TT_METAL_CACHE=/tmp/eb_asserts_cache
rm -rf $TT_METAL_CACHE; mkdir -p $TT_METAL_CACHE
run() { echo "##### $(date -u +%T) asserts: $*"; timeout -s INT -k 60 1500 python3 -m pytest -p no:cacheprovider -q -rfE --timeout=600 "$@" 2>&1 | grep -E "passed|failed|error|FAILED|ERROR|Timeout|timed out|ebreak|ASSERT" | tail -15; echo "rc=${PIPESTATUS[0]}"; }
run models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_csa_compressor.py -k "test_csa_compressor_single_device"
run tests/eb_r3_ci/test_eb_r3_ops.py -k "group_norm_dram_welford"
run tests/ttnn/unit_tests/operations/fused/test_group_norm_DRAM.py -k "test_group_norm_DRAM and two_pass"
# ELFs built with ENABLE_LLK_ASSERT, as a check the asserts were compiled in
grep -rl "ENABLE_LLK_ASSERT" $TT_METAL_CACHE 2>/dev/null | head -2; find $TT_METAL_CACHE -name "*.elf" | wc -l
echo "##### end $(date -u +%T)"
