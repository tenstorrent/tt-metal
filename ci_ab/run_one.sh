#!/usr/bin/env bash
# Scratch, not for merge. One failing ttnn topk id of the checked-out tree, with the compile error printed.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work
timeout -s INT -k 60 900 python3 -m pytest -p no:cacheprovider -q -x -rfE "tests/ttnn/unit_tests/operations/reduce/test_topk.py::test_topk[sub_core_grids=None-largest=True-sorted=True-N=1-C=1-H=32-W=8192-dim=3-k=6-BFLOAT16_B]" > /tmp/one.log 2>&1
echo "ONE rc=$?"
grep -a -n -E "error:|Error|Failed to generate|static assertion|too large|overflow|undefined reference" /tmp/one.log | head -30 | cut -c1-600
echo "ONE tail"; tail -n 40 /tmp/one.log | cut -c1-400
