#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth pass (#58725): the operand pass over up to eight sections against four (64 to 256 tiles per
# core; 32 tiles, four sections, the same program on both sides as the control); three passes.
cd /work
export EB_R3_LOG_RULE=1
for i in 1 2 3; do
  echo "##### pass $i k8: four vs eight"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NONE=1" "EB_R3_PRE_MAX=8" tests/eb_r3_ci/test_eb_r5.py -k "test_k8"
done
