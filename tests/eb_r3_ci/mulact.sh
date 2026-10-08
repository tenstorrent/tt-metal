#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, fourth pass (ci9): multiplies with an operand activation where the init runs per tile, the per-face
# hand-off (main's) against the head's per-tile one, five passes.
cd /work
for i in 1 2 3 4 5; do
  echo "##### pass $i mulact: per-face multiply vs head"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_PER_FACE=1" "EB_DUMMY=1" tests/eb_r3_ci/test_eb_mulact.py
done
