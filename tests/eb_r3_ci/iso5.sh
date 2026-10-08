#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, fourth pass (ci9): the r4b rows above main, alone; main's program, main's per-tile re-init alone and
# main's per-face multiply alone against the head, three passes.
cd /work
T=tests/eb_r3_ci/test_eb_iso5.py
M="EB_R3_NO_BLOCK=1 EB_R3_NO_BCAST_CHUNK=1 EB_R3_MAIN_REINIT=1 EB_R3_PER_FACE=1 EB_R3_NO_HIFI3=1 EB_R3_NO_PRE_SECTIONS=1 EB_R3_NO_NATIVE=1"
for i in 1 2 3; do
  echo "##### pass $i iso: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" $T
  echo "##### pass $i reinit: main's re-init vs head"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_MAIN_REINIT=1" "EB_DUMMY=1" $T
  echo "##### pass $i perface: main's per-face multiply vs head"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_PER_FACE=1" "EB_DUMMY=1" $T
done
