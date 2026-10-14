#!/usr/bin/env bash
# scratch: where / ternary tests on #58819 merged with origin/main (conflict resolved in ternary_sfpu_no_bcast_ttt.cpp)
set +e +o pipefail
export HWLOCK_HELD=ci
R=$(git rev-parse --show-toplevel); cd $R
echo "== head $(git log --oneline -1)"
export TT_METAL_CACHE=$HOME/plbits_cache; rm -rf $TT_METAL_CACHE; mkdir -p $TT_METAL_CACHE
for t in test_where.py test_ternary.py test_ternary_sharding.py test_where_unsupported_bcast.py; do
  timeout -s INT -k 60 2400 python -m pytest -q -p no:cacheprovider -o timeout_method=thread -rfE tests/ttnn/unit_tests/operations/eltwise/$t > /tmp/bits_$t.log 2>&1
  echo "== $t rc=$? $(grep -aE '[0-9]+ (passed|failed|error)' /tmp/bits_$t.log | tail -1)"
  grep -a -E "^FAILED|^ERROR" /tmp/bits_$t.log | head -20 | cut -c1-250
done
echo "== DONE"
