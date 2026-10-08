#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth pass (#58726): the class boundaries (every class native against main's routing) and main's
# program against the head on the subsets taken; three passes.
cd /work
export EB_R3_LOG_RULE=1
M="EB_R3_NO_BLOCK=1 EB_R3_NO_BCAST_CHUNK=1 EB_R3_MAIN_REINIT=1 EB_R3_PER_FACE=1 EB_R3_NO_HIFI3=1 EB_R3_NO_PRE_SECTIONS=1 EB_R3_NO_NATIVE=1"
for i in 1 2 3; do
  echo "##### pass $i bnd: current vs native"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NO_NATIVE=1" "EB_R3_NATIVE_ALL=1" tests/eb_r3_ci/test_eb_r5.py -k "test_nat8 or test_nat9"
  echo "##### pass $i n89m: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" tests/eb_r3_ci/test_eb_r5.py -k "test_nat8 or test_nat9"
done
