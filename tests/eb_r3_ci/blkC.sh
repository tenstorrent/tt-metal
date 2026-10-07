#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, third pass (#58725, #58726): sharded column and scalar broadcasts with an activation on the per-tile
# operand, one tile per section against a DEST section per acquire (EB_R3_BCAST_OPACT), three passes.
cd /work
export EB_R3_LOG_RULE=1
for i in 1 2 3; do
  echo "##### pass $i opact: one tile vs sections"; bash tests/eb_r3_ci/ab_envs.sh "EB_DUMMY=1" "EB_R3_BCAST_OPACT=1" tests/eb_r3_ci/test_eb_blk4.py -k test_blk4_opact
done
