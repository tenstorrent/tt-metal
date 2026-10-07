#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, third pass: a sharded residual add (main's program with EB_R3_NO_BLOCK, the head's block section
# with main's block pack without) followed by a common op; the follower's device time per setting, three passes.
cd /work
export EB_R3_LOG_RULE=1
for i in 1 2 3; do
  echo "##### pass $i seq: add per-tile vs add block section, then the follower"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NO_BLOCK=1" "EB_DUMMY=1" tests/eb_r3_ci/test_eb_seq.py
done
