#!/usr/bin/env bash
# Round 3 eltwise binary (#58723 review): the multi-chip callers a Blackhole QuietBox 2 (2 x P300, 4 chips) runs: ring and
# exp-ring joint SDPA and reduce_to_root; device time A/B with their opt-in, then bits.
cd /work
RJ=tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py
ER=tests/nightly/blackhole/sdpa/test_exp_ring_joint_sdpa.py
RR=tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/nightly/test_reduce_to_root_trace.py
echo "##### ring joint"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_ring.txt $RJ -k "sdpa_accuracy"
echo "##### exp ring joint"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_expring.txt $ER -k "sweep_perf_impl or sdpa_accuracy"
echo "##### reduce_to_root"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_r2r.txt $RR
echo "##### bits ring"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_ring.txt -p eb_seed_plugin $RJ -k "sdpa_accuracy"
echo "##### bits exp ring"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_expring.txt -p eb_seed_plugin $ER -k "sdpa_accuracy"
echo "##### bits reduce_to_root"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_r2r.txt -p eb_seed_plugin $RR
