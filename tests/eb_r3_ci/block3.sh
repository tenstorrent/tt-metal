#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58722 review): the next program after a block-pack program; identical bfp8 programs on both sides
# (block pack off, on), two passes.
cd /work
for i in 1 2; do
  echo "##### pass $i bu vs bu+bp"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NO_BLOCK_PACK=1" "EB_DUMMY=1" tests/eb_r3_ci/test_eb_block3.py -k next_unary
done
