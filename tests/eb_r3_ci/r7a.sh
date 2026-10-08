#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth pass (#58723): logical_and off the native path on block grids around 4x4 with the grid gate,
# main's program against the head and the head without the gate against the head; logical_or the control; five passes.
cd /work
export EB_R3_LOG_RULE=1
M="EB_R3_NO_BLOCK=1 EB_R3_NO_BCAST_CHUNK=1 EB_R3_MAIN_REINIT=1 EB_R3_PER_FACE=1 EB_R3_NO_HIFI3=1 EB_R3_NO_PRE_SECTIONS=1 EB_R3_NO_NATIVE=1"
for i in 1 2 3 4 5; do
  echo "##### pass $i ma3m: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" tests/eb_r3_ci/test_eb_r5.py -k "test_mulact3"
  echo "##### pass $i ma3g: no gate vs head"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NO_MUL_GATE=1" "EB_DUMMY=1" tests/eb_r3_ci/test_eb_r5.py -k "test_mulact3"
done
