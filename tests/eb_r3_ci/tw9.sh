#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58723 fourth review), P150, JIT root /work: reduce_to_root's twin (the whole compute kernel, 8x32 tiles,
# HiFi4, fp32 DEST), main's program against the broadcast opt-in (optin_wt_rb.txt) on the PR commit, which gives its 8x32 column
# broadcast multiplies the whole-tile program. Bits, the ELF sets, then three device time passes (main optin optin main).
cd /work
export TT_METAL_RUNTIME_ROOT=/work EB_SHOW_ERR=1
TW=tests/eb_r3_ci/twins/test_eb_twins.py
OPT=tests/eb_r3_ci/twins/optin_wt_rb.txt
echo "##### bits r2r"; EB_RUN_LIMIT=1500 bash tests/eb_r3_ci/bits_ab.sh $OPT -p eb_seed_plugin $TW -k "test_twin_r2r"
echo "##### elf r2r"; python3 tests/eb_r3_ci/elf_set_diff.py /tmp/ebbits/cache_main /tmp/ebbits/cache_optin "^tw_r2r$" 2>&1
for p in 1 2 3; do
  echo "##### r2r pass $p"; EB_RUN_LIMIT=2400 bash tests/eb_r3_ci/ab_set.sh $OPT $TW -k "test_twin_r2r"
done
