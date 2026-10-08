#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth pass (#58724): the modules of the four kernels whole, their defines removed against the head,
# bit for bit.
cd /work
F=tests/ttnn/unit_tests/operations/fused
E=tests/ttnn/unit_tests/operations/eltwise
echo "##### modules norm: main vs head"; bash tests/eb_r3_ci/bits_strip.sh tests/eb_r3_ci/off_r7b.txt -p eb_seed_plugin $F/test_group_norm.py $F/test_group_norm_DRAM.py $F/test_layer_norm.py $F/test_batch_norm.py
echo "##### modules hardswish: main vs head"; bash tests/eb_r3_ci/bits_strip.sh tests/eb_r3_ci/off_r7b.txt -p eb_seed_plugin $E/test_unary.py $E/test_unary_category2_bfloat16.py -k hardswish
