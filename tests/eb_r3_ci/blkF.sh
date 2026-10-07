#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, third pass, the final candidate head (ci6): main's program (every round toggle off) against the
# head on binary_ng's sharded no-broadcast, Python-scalar, column and scalar broadcast ops with and without activations, the
# HiFi3 rule, and the rows the candidate read slower, alone.
cd /work
export EB_R3_LOG_RULE=1
T=tests/eb_r3_ci/test_eb_blk4.py
M="EB_R3_NO_BLOCK=1 EB_R3_NO_BCAST_CHUNK=1 EB_R3_MAIN_REINIT=1 EB_R3_PER_FACE=1 EB_R3_NO_HIFI3=1"
echo "##### nob: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" $T -k test_blk4_nob
echo "##### post: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" $T -k test_blk4_post
echo "##### scalar: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" $T -k test_blk4_scalar
echo "##### bcast: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" $T -k test_blk4_bcast
echo "##### opact: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" $T -k test_blk4_opact
echo "##### hifi3: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" $T -k test_blk4_hifi3
echo "##### hifi3: head at HiFi4 vs head"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NO_HIFI3=1" "EB_DUMMY=1" $T -k test_blk4_hifi3
echo "##### iso: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" tests/eb_r3_ci/test_eb_iso.py
