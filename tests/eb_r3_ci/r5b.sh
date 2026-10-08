#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, fifth pass (#58723): the per-face multiply against the head's per-tile one where the init runs per
# tile, with logical_or (an add, the same program on both sides) as the control; five passes.
cd /work
for i in 1 2 3 4 5; do
  echo "##### pass $i ma: per-face multiply vs head"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_PER_FACE=1" "EB_R3_NONE=1" tests/eb_r3_ci/test_eb_r5.py -k "test_mulact2"
done
