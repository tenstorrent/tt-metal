#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, third pass: a block-pack residual add then a small sharded binary op; the sequence's total device
# time, main's program (every round toggle off) against the head, three passes. "none" runs the follower alone.
cd /work
export EB_R3_LOG_RULE=1
M="EB_R3_NO_BLOCK=1 EB_R3_NO_BCAST_CHUNK=1 EB_R3_MAIN_REINIT=1 EB_R3_PER_FACE=1 EB_R3_NO_HIFI3=1"
for i in 1 2 3; do
  echo "##### pass $i seq2: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" tests/eb_r3_ci/test_eb_seq2.py
  echo "##### pass $i seq2: head without the block pack (EB_R3_NO_BLOCK) vs head"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NO_BLOCK=1" "EB_DUMMY=1" tests/eb_r3_ci/test_eb_seq2.py
done
