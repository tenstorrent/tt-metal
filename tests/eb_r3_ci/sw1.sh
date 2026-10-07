#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, fourth pass: the switch to #58816's block pack (pr/switch_58816.patch) on the head merged with
# #58816 (ff4e3f3efa0). The block sections without any block unpack or pack (EB_R3_NO_BLOCK) against the switch, outputs
# compared, three passes.
cd /work
T=tests/eb_r3_ci/test_eb_blk4.py
for i in 1 2 3; do
  echo "##### pass $i sw: no block vs the switch"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NO_BLOCK=1 EB_R3_NO_PRE_BLOCK=1" "EB_R3_NONE=1" $T -k "test_blk4_nob or test_blk4_scalar or test_blk4_bcast or test_blk4_post"
done
