#!/usr/bin/env bash
# Round 3 eltwise binary: the PR head's binary_ng (block section, broadcast sections, post-activation init) from an empty
# kernel cache, the binary modules whole and the CI cases, outputs checked against torch.
cd /work
export PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-} TT_METAL_CACHE=/tmp/final_cache
E=tests/ttnn/unit_tests/operations/eltwise
timeout 3000 python3 -m pytest -q -p no:cacheprovider -o timeout_method=thread -rfE $E/test_add.py $E/test_mul.py $E/test_binary_bcast.py $E/test_binaryng_fp32.py $E/test_binary_ng_sharded_fp32_batch.py $E/test_binary_scalar.py $E/test_binaryng_ND.py $E/test_binary_ng_typecast.py $E/test_binary_ng_activation_mixed_dtype.py $E/test_binary_bcast_tcast.py 2>&1 | tail -15
for t in tests/eb_r3_ci/test_eb_bng.py tests/eb_r3_ci/test_eb_block2.py; do timeout 1800 python3 -m pytest -q -p no:cacheprovider -o timeout_method=thread -rfE $t 2>&1 | tail -6; done
