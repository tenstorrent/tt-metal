#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, fifth pass (#58723): the per-face multiply against the head's per-tile one for logical_and off the
# native path on block grids of 8 to 32 cores (the same 1024x1024 tensor on 2x4, 4x4, 2x8, 4x2), with logical_or as the
# same-program control; five passes.
cd /work
for i in 1 2 3 4 5; do
  echo "##### pass $i ma3: per-face multiply vs head"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_PER_FACE=1" "EB_R3_NONE=1" tests/eb_r3_ci/test_eb_r5.py -k "test_mulact3"
done
