#!/usr/bin/env bash
# Round 3 eltwise binary (#58725 review): the re-init skip after a post activation alone (EB_R3_P58725) on the column, scalar
# and Python-scalar kernels, three passes; the compute ELF diff of every binary_ng kernel against main's program (matched by
# defines and compile-time arguments); the binary modules with fused activations bit for bit.
cd /work
T=tests/eb_r3_ci/test_eb_bng.py
for i in 1 2 3; do
  echo "##### p58725 bcast pass $i"; bash tests/eb_r3_ci/ab_env2.sh EB_R3_P58725 $T -k "bcast_post_activation or post_activation"
done
timeout 900 python3 tests/eb_r3_ci/elf_pair_diff.py /tmp/ebenv2/cache_main /tmp/ebenv2/cache_optin eltwise_binary_no_bcast eltwise_binary_col_bcast eltwise_binary_scalar_bcast eltwise_binary_scalar eltwise_binary eltwise_binary_row_bcast
E=tests/ttnn/unit_tests/operations/eltwise
echo "##### modules"; EB_VAR_ON_OPTIN=1 bash tests/eb_r3_ci/bits_env.sh EB_R3_P58725 -p eb_seed_plugin $E/test_binary_ng_activation_mixed_dtype.py $E/test_add.py $E/test_mul.py $E/test_binary_bcast.py $E/test_binaryng_fp32.py $E/test_binary_scalar.py $E/test_binary_ng_typecast.py $T
