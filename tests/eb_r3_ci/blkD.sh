#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, third pass, the candidate head (ci5): main's program (every round toggle off) against the head's
# rule on binary_ng's sharded no-broadcast, Python-scalar and column or scalar broadcast ops, with and without activations.
cd /work
T=tests/eb_r3_ci/test_eb_blk4.py
export EB_R3_LOG_RULE=1
M="EB_R3_NO_BLOCK=1 EB_R3_NO_BCAST_CHUNK=1 EB_R3_MAIN_REINIT=1 EB_R3_PER_FACE=1"
echo "##### nob: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" $T -k test_blk4_nob
echo "##### post: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" $T -k test_blk4_post
echo "##### scalar: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" $T -k test_blk4_scalar
echo "##### bcast: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" $T -k test_blk4_bcast
echo "##### opact: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" $T -k test_blk4_opact
