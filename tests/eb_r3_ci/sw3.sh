#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, fourth pass: the switch to #58816's block pack on the final head (ec7714f90f8) merged with #58816;
# outputs against the head's per-tile packs (no block section, no operand-pass block pack), and the binary modules whole
# against main's program; one pass of device time.
cd /work
M="EB_R3_NO_BLOCK=1 EB_R3_NO_BCAST_CHUNK=1 EB_R3_MAIN_REINIT=1 EB_R3_PER_FACE=1 EB_R3_NO_HIFI3=1 EB_R3_NO_PRE_SECTIONS=1 EB_R3_NO_NATIVE=1 EB_R3_NO_PRE_BLOCK=1"
echo "##### sw: no block vs the switch"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NO_BLOCK=1 EB_R3_NO_PRE_BLOCK=1" "EB_R3_NONE=1" tests/eb_r3_ci/test_eb_blk4.py -k "test_blk4_nob or test_blk4_scalar or test_blk4_bcast or test_blk4_post"
echo "##### swmp: no operand-pass block pack vs the switch"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NO_PRE_BLOCK=1" "EB_R3_NONE=1" tests/eb_r3_ci/test_eb_r3_mp.py -k "test_mp or (test_nat and not test_nat2)"
E=tests/ttnn/unit_tests/operations/eltwise
echo "##### modules: main vs the switch"; bash tests/eb_r3_ci/bits_envs.sh "$M" "EB_DUMMY=1" -p eb_seed_plugin $E/test_add.py $E/test_mul.py $E/test_binary_bcast.py $E/test_binaryng_fp32.py $E/test_binary_ng_sharded_fp32_batch.py $E/test_binary_scalar.py $E/test_binaryng_ND.py $E/test_binary_ng_typecast.py $E/test_binary_ng_activation_mixed_dtype.py $E/test_binary_ng_bcast_fp32_dest_acc.py $E/test_binary_ng_width_padded_stride.py $E/test_binary_fp32.py $E/test_binary_bcast_tcast.py $E/test_div_ops.py $E/test_binary_category4_bfloat16.py
