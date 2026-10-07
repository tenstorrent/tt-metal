#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58723 review): QuietBox 2, reduce_to_root device time A/B with the pytest timeout raised (under the
# profiler both tests pass and then exceed the 300 s ini timeout in teardown, run 37577737354). Bits: run 37577737354.
cd /work
export EB_SHOW_ERR=1 EB_RUN_LIMIT=2400
RR=tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/nightly/test_reduce_to_root_trace.py
for t in test_reduce_to_root_auto_intermediate test_reduce_to_root_with_trace; do
  echo "##### reduce_to_root $t"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_r2r.txt -o timeout=1800 $RR -k "$t"
done
