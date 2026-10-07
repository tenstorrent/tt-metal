#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, fourth pass: the switch's operand activation pass (one block pack per section into a bf16 or fp32
# intermediate) against the per-tile pack, and a block-pack add followed by a small sharded binary op (the sequence's
# total), no block against the switch; three passes.
cd /work
for i in 1 2 3; do
  echo "##### pass $i pre: per-tile pack vs the block pack in the operand pass"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NO_PRE_BLOCK=1" "EB_R3_NONE=1" tests/eb_r3_ci/test_eb_r3_mp.py -k test_mp
  echo "##### pass $i seq2: main vs the switch"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NO_BLOCK=1 EB_R3_NO_PRE_BLOCK=1 EB_R3_NO_BCAST_CHUNK=1 EB_R3_MAIN_REINIT=1 EB_R3_PER_FACE=1 EB_R3_NO_HIFI3=1" "EB_R3_NONE=1" tests/eb_r3_ci/test_eb_seq2.py
done
