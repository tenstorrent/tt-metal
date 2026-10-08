#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, fifth pass, final code (#58726): main's program against the head on the fourth pass's native routing
# cases (test_nat, test_nat2, test_nat3) and the block sections' broadcast cases; three passes.
cd /work
export EB_R3_LOG_RULE=1
M="EB_R3_NO_BLOCK=1 EB_R3_NO_BCAST_CHUNK=1 EB_R3_MAIN_REINIT=1 EB_R3_PER_FACE=1 EB_R3_NO_HIFI3=1 EB_R3_NO_PRE_SECTIONS=1 EB_R3_NO_NATIVE=1"
for i in 1 2 3; do
  echo "##### pass $i nat123: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" tests/eb_r3_ci/test_eb_r3_mp.py -k "test_nat"
  echo "##### pass $i bcast: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" tests/eb_r3_ci/test_eb_blk4.py -k "test_blk4_bcast or test_blk4_opact"
done
