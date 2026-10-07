#!/usr/bin/env bash
# Round 3 eltwise binary (#58723, #58724 review): moreh_sgd's bf16 cases, dit_minimal_matmul_addcmul_fused at production shapes,
# and softmax's dest-reuse add on the padded last tile (its broadcast switch in the tree on both sides): device time A/B, bits.
cd /work
S=tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_sgd.py
F=tests/ttnn/nightly/unit_tests/operations/experimental/test_dit_minimal_matmul_addcmul_fused.py
SM=tests/ttnn/unit_tests/operations/fused/test_softmax.py
K="non_divisible_width or non_tile_aligned_width"
echo "##### sgd"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_sgd.txt --nodes-file tests/eb_r3_ci/nodes_sgd.txt
echo "##### mmac production shapes"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_mmac.txt tests/eb_r3_ci/test_eb_mmac.py
echo "##### mmac module"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_mmac.txt $F
echo "##### softmax padded dest reuse"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_smdr.txt $SM -k "$K"
echo "##### bits sgd"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_sgd.txt $S
echo "##### bits mmac"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_mmac.txt -p eb_seed_plugin $F tests/eb_r3_ci/test_eb_mmac.py
echo "##### bits softmax"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_smdr.txt -p eb_seed_plugin $SM
