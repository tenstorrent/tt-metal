#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, third pass (#58722, #58726, #58727): block unpack and block pack with post activations, in the
# Python-scalar kernel and in the column and scalar broadcast sections; the broadcast sections with post activations.
cd /work
T=tests/eb_r3_ci/test_eb_blk4.py
export EB_R3_LOG_RULE=1
echo "##### post: none vs bu"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NO_BLOCK=1" "EB_R3_BLK_POST=1 EB_R3_BU_MIN=1 EB_R3_NO_BLOCK_PACK=1" $T -k test_blk4_post
echo "##### post: none vs bu+bp2"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NO_BLOCK=1" "EB_R3_BLK_POST=1 EB_R3_BU_MIN=1 EB_R3_BP_MIN=1 EB_R3_BP_KIND=2" $T -k test_blk4_post
echo "##### scalar: none vs bu"; bash tests/eb_r3_ci/ab_envs.sh "EB_DUMMY=1" "EB_R3_BLK_SCALAR=1 EB_R3_BLK_POST=1 EB_R3_BU_MIN=1 EB_R3_NO_BLOCK_PACK=1" $T -k test_blk4_scalar
echo "##### scalar: none vs bu+bp2"; bash tests/eb_r3_ci/ab_envs.sh "EB_DUMMY=1" "EB_R3_BLK_SCALAR=1 EB_R3_BLK_POST=1 EB_R3_BU_MIN=1 EB_R3_BP_MIN=1 EB_R3_BP_KIND=2" $T -k test_blk4_scalar
echo "##### bcast: sections vs sections+bu"; bash tests/eb_r3_ci/ab_envs.sh "EB_DUMMY=1" "EB_R3_BLK_BCAST=1" $T -k "test_blk4_bcast and None"
echo "##### bcast: sections vs sections+bu+bp2"; bash tests/eb_r3_ci/ab_envs.sh "EB_DUMMY=1" "EB_R3_BLK_BCAST=1 EB_R3_BCAST_BP=1 EB_R3_BP_MIN=1" $T -k "test_blk4_bcast and None"
echo "##### bcast post: one tile vs sections"; bash tests/eb_r3_ci/ab_envs.sh "EB_DUMMY=1" "EB_R3_BCAST_POST=1" $T -k "test_blk4_bcast and (gelu or silu)"
