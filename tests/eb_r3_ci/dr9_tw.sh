#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58724 fourth review): single-core twins on a P150. tw_r2o: deepseek_v3_b1's ReduceToOneB1 Op itself,
# the PR head's header (main's per-face add) against the PR's (the add's direct per-tile LLK calls), ROOT3/2/1, LoFi (fused
# kernels) and HiFi4 (micro op). tw_em: EltwiseMul at 1x32 and 16x16 with ELTWISE_BINARY_PER_TILE_HANDOFF_DEST_REUSE, which the
# one-face rule returns to main's program at 16x16. Bits of both sides, their ELFs, then device time in passes of main optin
# optin main under the profiler. The JIT reads every header from /work.
cd /work
export TT_METAL_RUNTIME_ROOT=/work EB_SHOW_ERR=1
TW=tests/eb_r3_ci/twins/test_eb_twins.py
OPT=tests/eb_r3_ci/twins/optin_dr9.txt
K="test_twin_r2o or test_twin_em"
echo "##### bits"; EB_RUN_LIMIT=1500 bash tests/eb_r3_ci/bits_ab.sh $OPT -p eb_seed_plugin $TW -k "$K"
echo "##### elf"; python3 tests/eb_r3_ci/elf_set_diff.py /tmp/ebbits/cache_main /tmp/ebbits/cache_optin "^tw_(r2o|em)$" 2>&1
echo "##### elf sites tw_r2o (head header | PR header)"; python3 tests/eb_r3_ci/elfsite/r2o_sites.py /tmp/ebbits/cache_main /tmp/ebbits/cache_optin "^tw_r2o$" --a=head --b=pr --shift=1 2>&1
for p in 1 2 3 4 5; do
  echo "##### twins pass $p"; EB_RUN_LIMIT=1800 bash tests/eb_r3_ci/ab_set.sh $OPT $TW -k "$K"
done
