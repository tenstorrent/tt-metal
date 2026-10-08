#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, fifth pass (#58725): two against four sections per operand pass, and the pass structure for one
# section alone; three passes each.
cd /work
for i in 1 2 3; do
  echo "##### pass $i k4: two sections (head) vs four"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NONE=1" "EB_R3_PRE_SECTIONS=4" tests/eb_r3_ci/test_eb_r3_mp.py -k test_mp
  echo "##### pass $i one: one section, head vs the pass structure"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NONE=1" "EB_R3_PRE_ONE=1" tests/eb_r3_ci/test_eb_r5.py -k test_one
done
