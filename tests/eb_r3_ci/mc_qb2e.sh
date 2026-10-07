#!/usr/bin/env bash
# Round 3 eltwise binary (#58723 review): QuietBox 2, reduce_to_root: bits of both tests first (no profiler), then the device
# time A/B of the test without trace (the traced one timed out under the profiler and left the board unusable, run 37575323709).
cd /work
export EB_SHOW_ERR=1 EB_RUN_LIMIT=900
RR=tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/nightly/test_reduce_to_root_trace.py
echo "##### bits reduce_to_root"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_r2r.txt -p eb_seed_plugin $RR
echo "##### reduce_to_root auto intermediate"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_r2r.txt $RR -k "test_reduce_to_root_auto_intermediate"
