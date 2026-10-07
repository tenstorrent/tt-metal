#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, third pass (#58723): binary_ng's multiply with a block-float SrcB at HiFi3; main's program against the
# head, and the head with HiFi4 (EB_R3_NO_HIFI3) against the head.
cd /work
export EB_R3_LOG_RULE=1
T=tests/eb_r3_ci/test_eb_blk4.py
M="EB_R3_NO_BLOCK=1 EB_R3_NO_BCAST_CHUNK=1 EB_R3_MAIN_REINIT=1 EB_R3_PER_FACE=1 EB_R3_NO_HIFI3=1"
echo "##### hifi3: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" $T -k test_blk4_hifi3
echo "##### hifi3: head at HiFi4 vs head"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NO_HIFI3=1" "EB_DUMMY=1" $T -k test_blk4_hifi3
