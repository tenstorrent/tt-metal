#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58723 review): QuietBox 2 again: exp-ring joint SDPA bits (no seed plugin) first, then the ring joint
# SDPA perf-check cases A/B and bits, and the exp-ring A/B a second time.
cd /work
RJ=tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py
ER=tests/nightly/blackhole/sdpa/test_exp_ring_joint_sdpa.py
echo "##### bits exp ring"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_expring.txt $ER -k "sdpa_accuracy"
echo "##### exp ring joint pass 2"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_expring.txt $ER -k "sweep_perf_impl or sdpa_accuracy"
echo "##### ring joint perf_check"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_ring.txt $RJ -k "perf_check"
echo "##### bits ring perf_check"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_ring.txt $RJ -k "perf_check"
