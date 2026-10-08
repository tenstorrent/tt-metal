#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, fourth pass, the head with the fourth pass's binary_ng changes (ci9): main's program (every round
# toggle off) against the head on sharded ops with an operand activation (#58725) and block or width sharded broadcasts
# (#58726), three passes; and the head without the two changes against the head.
cd /work
export EB_R3_LOG_RULE=1
T=tests/eb_r3_ci/test_eb_r3_mp.py
M="EB_R3_NO_BLOCK=1 EB_R3_NO_BCAST_CHUNK=1 EB_R3_MAIN_REINIT=1 EB_R3_PER_FACE=1 EB_R3_NO_HIFI3=1 EB_R3_NO_PRE_SECTIONS=1 EB_R3_NO_NATIVE=1"
for i in 1 2 3; do
  echo "##### pass $i mp: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" $T -k test_mp
  echo "##### pass $i nat: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" $T -k "test_nat and not test_nat2"
  echo "##### pass $i mp4: head without the two changes vs head"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NO_PRE_SECTIONS=1 EB_R3_NO_NATIVE=1" "EB_DUMMY=1" $T -k "test_mp or test_nat3"
done
