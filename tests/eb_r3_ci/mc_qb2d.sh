#!/usr/bin/env bash
# Round 3 eltwise binary (#58723 review): QuietBox 2, reduce_to_root (it builds and passes without the device profiler):
# device time A/B with its opt-in under the profiler, the build error printed if there is one; then bits.
cd /work
export EB_SHOW_ERR=1 EB_RUN_LIMIT=1200
RR=tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/nightly/test_reduce_to_root_trace.py
for t in test_reduce_to_root_with_trace test_reduce_to_root_auto_intermediate; do
  echo "##### reduce_to_root $t"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_r2r.txt $RR -k "$t"
  O=/tmp/ebset; for f in $O/log_1_main.txt; do grep -B2 -A12 "error:" $f | head -60 | cut -c1-400; done
  echo "##### bits reduce_to_root $t"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_r2r.txt -p eb_seed_plugin $RR -k "$t"
done
