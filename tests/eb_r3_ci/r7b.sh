#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth pass (#58724): the four opted-in kernels whose math ELF the one-face rule changed, main's
# program (their defines removed) against the head, at the N2 and N11 shapes; three passes.
cd /work
for i in 1 2 3; do
  echo "##### pass $i four: main vs head"; bash tests/eb_r3_ci/ab_off.sh tests/eb_r3_ci/off_r7b.txt tests/eb_r3_ci/test_eb_r3_ops.py -k "test_group_norm_dram_welford or test_layer_norm_welford_large or test_hardswish or test_batch_norm_fpu"
done
