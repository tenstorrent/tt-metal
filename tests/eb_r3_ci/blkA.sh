#!/usr/bin/env bash
# Round 3 eltwise binary, third pass (#58722): binary_ng's sharded no-broadcast ops over tiles per core and formats; the branch
# without the block section against the block unpack alone, with main's contiguous block pack, and with #58816's block pack.
cd /work
T=tests/eb_r3_ci/test_eb_blk4.py
export EB_R3_LOG_RULE=1
echo "##### none vs bu"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NO_BLOCK=1" "EB_R3_BU_MIN=1 EB_R3_NO_BLOCK_PACK=1" $T -k test_blk4_nob
echo "##### none vs bu+bp1"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NO_BLOCK=1" "EB_R3_BU_MIN=1 EB_R3_BP_MIN=1 EB_R3_BP_KIND=1" $T -k test_blk4_nob
echo "##### none vs bu+bp2"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NO_BLOCK=1" "EB_R3_BU_MIN=1 EB_R3_BP_MIN=1 EB_R3_BP_KIND=2" $T -k test_blk4_nob
