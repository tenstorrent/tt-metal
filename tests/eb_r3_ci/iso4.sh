#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, fourth pass (ci9): the r4a rows above main, alone; main's program and the head without its two
# changes against the head, three passes.
cd /work
M="EB_R3_NO_BLOCK=1 EB_R3_NO_BCAST_CHUNK=1 EB_R3_MAIN_REINIT=1 EB_R3_PER_FACE=1 EB_R3_NO_HIFI3=1 EB_R3_NO_PRE_SECTIONS=1 EB_R3_NO_NATIVE=1"
for i in 1 2 3; do
  echo "##### pass $i iso: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" tests/eb_r3_ci/test_eb_iso4.py
  echo "##### pass $i iso4: head without the two changes vs head"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NO_PRE_SECTIONS=1 EB_R3_NO_NATIVE=1" "EB_DUMMY=1" tests/eb_r3_ci/test_eb_iso4.py
done
