#!/usr/bin/env bash
# Round 3 eltwise binary (#58722 review): the block section over the tiles per core; none against block unpack, block unpack
# against block unpack and block pack, none against both; two passes each.
cd /work
T=tests/eb_r3_ci/test_eb_block2.py
for i in 1 2; do
  echo "##### pass $i none vs bu"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NO_BLOCK=1" "EB_R3_NO_BLOCK_PACK=1" $T
  echo "##### pass $i bu vs bu+bp"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NO_BLOCK_PACK=1" "EB_DUMMY=1" $T
  echo "##### pass $i none vs bu+bp"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NO_BLOCK=1" "EB_DUMMY=1" $T
done
