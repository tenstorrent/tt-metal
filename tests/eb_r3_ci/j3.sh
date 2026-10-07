#!/usr/bin/env bash
# Round 3 eltwise binary (#58725, #58726 review): the post-activation re-init skip (p58725), three interleaved A/B passes (6 runs
# per side), its compute ELF diff, bits; then the compute-sensitivity probe (one more binary init per tile) on the sharded ops
# that run one tile per DEST section, two passes.
cd /work
T=tests/eb_r3_ci/test_eb_bng.py
for i in 1 2 3; do
  echo "##### p58725 pass $i"; bash tests/eb_r3_ci/ab_env2.sh EB_R3_P58725 $T -k "post_activation or interleaved"
  [[ $i == 1 ]] && python3 tests/eb_r3_ci/elf_pair_diff.py /tmp/ebenv2/cache_main /tmp/ebenv2/cache_optin eltwise_binary_no_bcast
done
for i in 1 2; do
  echo "##### probe pass $i"; bash tests/eb_r3_ci/ab_env2.sh EB_R3_PROBE_INIT $T -k "sharded_bcast or interleaved or bng_bcast"
done
