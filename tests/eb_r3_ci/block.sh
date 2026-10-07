#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58722 review): binary_ng's block section (block unpack plus block pack) against the branch without
# it, then the block unpack alone; device time A/B and bits, and the binary modules whole bit for bit.
cd /work
export EB_R3_LOG_RULE=1
echo "##### block section against none"; bash tests/eb_r3_ci/ab_env.sh EB_R3_NO_BLOCK tests/eb_r3_ci/test_eb_block.py
echo "##### block pack against block unpack alone"; bash tests/eb_r3_ci/ab_env.sh EB_R3_NO_BLOCK_PACK tests/eb_r3_ci/test_eb_block.py
E=tests/ttnn/unit_tests/operations/eltwise
echo "##### modules"; bash tests/eb_r3_ci/bits_env.sh EB_R3_NO_BLOCK -p eb_seed_plugin $E/test_add.py $E/test_mul.py $E/test_binary_bcast.py $E/test_binaryng_fp32.py $E/test_binary_ng_sharded_fp32_batch.py $E/test_binary_scalar.py $E/test_binaryng_ND.py $E/test_binary_ng_typecast.py
