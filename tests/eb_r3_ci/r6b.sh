#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, fifth pass, final code (#58726): main's program against the head on block and width sharded column
# and scalar broadcasts, every class measured in the fifth pass (test_nat4 to test_nat7); three passes.
cd /work
export EB_R3_LOG_RULE=1
M="EB_R3_NO_BLOCK=1 EB_R3_NO_BCAST_CHUNK=1 EB_R3_MAIN_REINIT=1 EB_R3_PER_FACE=1 EB_R3_NO_HIFI3=1 EB_R3_NO_PRE_SECTIONS=1 EB_R3_NO_NATIVE=1"
for i in 1 2 3; do
  echo "##### pass $i nat47: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" tests/eb_r3_ci/test_eb_r5.py -k "test_nat4 or test_nat5 or test_nat6 or test_nat7"
done
