#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, fifth pass, final code (#58725): main's program against the head on sharded ops with an operand
# activation (multi-section, one section, bfp4 included), and the fourth pass's two sections against the head; three passes.
cd /work
export EB_R3_LOG_RULE=1
M="EB_R3_NO_BLOCK=1 EB_R3_NO_BCAST_CHUNK=1 EB_R3_MAIN_REINIT=1 EB_R3_PER_FACE=1 EB_R3_NO_HIFI3=1 EB_R3_NO_PRE_SECTIONS=1 EB_R3_NO_NATIVE=1"
for i in 1 2 3; do
  echo "##### pass $i mp: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" tests/eb_r3_ci/test_eb_r3_mp.py -k test_mp
  echo "##### pass $i one: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" tests/eb_r3_ci/test_eb_r5.py -k test_one
  echo "##### pass $i k2mp: fourth pass vs head"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_PRE_K2=1" "EB_DUMMY=1" tests/eb_r3_ci/test_eb_r3_mp.py -k test_mp
  echo "##### pass $i k2one: fourth pass vs head"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_PRE_K2=1" "EB_DUMMY=1" tests/eb_r3_ci/test_eb_r5.py -k test_one
done
