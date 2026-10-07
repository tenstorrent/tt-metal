#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58726 review): sharded ops with a column or scalar broadcast against a sharded operand compute a DEST
# section of tiles per acquire (BCAST_OTHER_CHUNK); the branch without it (EB_R3_NO_BCAST_CHUNK) against with it, three passes,
# over the sharded broadcast cases and the interleaved broadcast controls; then the binary modules bit for bit.
cd /work
T=tests/eb_r3_ci/test_eb_bng.py
for i in 1 2 3; do
  echo "##### pass $i none vs sections"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NO_BCAST_CHUNK=1" "EB_DUMMY=1" $T -k "sharded_bcast or bng_bcast"
done
E=tests/ttnn/unit_tests/operations/eltwise
echo "##### modules"; bash tests/eb_r3_ci/bits_env.sh EB_R3_NO_BCAST_CHUNK -p eb_seed_plugin $E/test_add.py $E/test_mul.py $E/test_binary_bcast.py $E/test_binaryng_fp32.py $E/test_binary_ng_sharded_fp32_batch.py $E/test_binary_scalar.py $E/test_binaryng_ND.py $E/test_binary_ng_typecast.py $E/test_binary_bcast_tcast.py $T
