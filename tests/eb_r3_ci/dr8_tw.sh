#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58724 third review): single-core twins of deepseek_v3_b1's dest-reuse calls that
# ELTWISE_BINARY_PER_TILE_HANDOFF_DEST_REUSE changes in moe_kernel, decoder_block_kernel and moe_routed_expert_kernel
# (ReduceToOneB1's add, EltwiseMul at 1x32 and 16x16) and GatedReduce at 8x32 with the whole-tile program for 8-row faces,
# main's per-face program against the opt-in: bits of both sides, their ELFs, then three device time passes (main optin optin
# main under the profiler). P150, the JIT reading every header from /work.
cd /work
export TT_METAL_RUNTIME_ROOT=/work EB_SHOW_ERR=1
TW=tests/eb_r3_ci/twins/test_eb_twins.py
OPT=tests/eb_r3_ci/twins/optin_dr8.txt
K="test_twin_gr or test_twin_em or test_twin_r21"
echo "##### bits"; EB_RUN_LIMIT=1200 bash tests/eb_r3_ci/bits_ab.sh $OPT -p eb_seed_plugin $TW -k "$K"
echo "##### elf"; python3 tests/eb_r3_ci/elf_set_diff.py /tmp/ebbits/cache_main /tmp/ebbits/cache_optin "^tw_(gr|em|r21)$" 2>&1
for p in 1 2 3; do
  echo "##### twins pass $p"; EB_RUN_LIMIT=1800 bash tests/eb_r3_ci/ab_set.sh $OPT $TW -k "$K"
done
