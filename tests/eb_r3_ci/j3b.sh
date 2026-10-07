#!/usr/bin/env bash
# Round 3 eltwise binary (#58725, #58726 review): the post-activation re-init skip (p58725), passes 2 and 3 (2 runs per side
# each; pass 1 is run 37563834806), then the compute ELF diff of main against p58725; then the compute-sensitivity probe (one
# more binary init per tile) on the sharded ops that run one tile per DEST section and the interleaved controls, two passes.
cd /work
T=tests/eb_r3_ci/test_eb_bng.py
for i in 2 3; do
  echo "##### p58725 pass $i"; bash tests/eb_r3_ci/ab_env2.sh EB_R3_P58725 $T -k "post_activation or interleaved"
done
timeout 600 python3 tests/eb_r3_ci/elf_pair_diff.py /tmp/ebenv2/cache_main /tmp/ebenv2/cache_optin eltwise_binary_no_bcast eltwise_binary_row_bcast eltwise_binary_col_bcast eltwise_binary_scalar_bcast eltwise_binary
for i in 1 2; do
  echo "##### probe pass $i"; bash tests/eb_r3_ci/ab_env2.sh EB_R3_PROBE_INIT $T -k "sharded_bcast or interleaved or bng_bcast"
done
