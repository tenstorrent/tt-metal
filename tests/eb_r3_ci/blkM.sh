#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, third pass, the candidate head (ci5): the binary test modules whole, seeded, main's program (every
# round toggle off) against the head, every output hashed and compared per test.
cd /work
E=tests/ttnn/unit_tests/operations/eltwise
M="EB_R3_NO_BLOCK=1 EB_R3_NO_BCAST_CHUNK=1 EB_R3_MAIN_REINIT=1 EB_R3_PER_FACE=1"
echo "##### modules: main vs head"; bash tests/eb_r3_ci/bits_envs.sh "$M" "EB_DUMMY=1" -p eb_seed_plugin $E/test_add.py $E/test_mul.py $E/test_binary_bcast.py $E/test_binaryng_fp32.py $E/test_binary_ng_sharded_fp32_batch.py $E/test_binary_scalar.py $E/test_binaryng_ND.py $E/test_binary_ng_typecast.py $E/test_binary_ng_activation_mixed_dtype.py $E/test_binary_ng_bcast_fp32_dest_acc.py $E/test_binary_ng_width_padded_stride.py $E/test_binary_fp32.py $E/test_binary_bcast_tcast.py
