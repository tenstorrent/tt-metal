#!/bin/bash
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work
T=tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_large_indices.py
export TT_METAL_CACHE=$(mktemp -d)
python3 -m pytest -q -p no:cacheprovider $T::test_topk_large_indices_restricted_grid_cache_hit_rebinds_shape_and_valid_length > /tmp/t1.log 2>&1
echo "== single rc=$?: $(grep -aE '[0-9]+ (passed|failed|skipped)' /tmp/t1.log | tail -1)"; grep -aE '^(FAILED|ERROR)|assert' /tmp/t1.log | cut -c1-240
timeout -s INT -k 60 3000 python3 -m pytest -q -p no:cacheprovider -o timeout_method=thread $T -k "not production_perf" > /tmp/t2.log 2>&1
rc=$?; echo "== file rc=$rc: $(grep -aE '[0-9]+ (passed|failed)' /tmp/t2.log | tail -1)"; grep -aE '^(FAILED|ERROR)' /tmp/t2.log | cut -c1-240
exit $rc
