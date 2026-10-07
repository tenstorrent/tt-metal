#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58722 review): the block section as the PR gates it (16 or more tiles per core) against none, three
# passes; then the binary modules bit for bit.
cd /work
for i in 1 2 3; do
  echo "##### pass $i none vs block"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NO_BLOCK=1" "EB_DUMMY=1" tests/eb_r3_ci/test_eb_block2.py
done
E=tests/ttnn/unit_tests/operations/eltwise
echo "##### modules"; bash tests/eb_r3_ci/bits_env.sh EB_R3_NO_BLOCK -p eb_seed_plugin $E/test_add.py $E/test_mul.py $E/test_binary_bcast.py $E/test_binaryng_fp32.py $E/test_binary_ng_sharded_fp32_batch.py $E/test_binary_scalar.py $E/test_binaryng_ND.py $E/test_binary_ng_typecast.py tests/eb_r3_ci/test_eb_block2.py
