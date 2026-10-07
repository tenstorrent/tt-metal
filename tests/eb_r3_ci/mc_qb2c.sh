#!/usr/bin/env bash
# Round 3 eltwise binary (#58723 review): QuietBox 2, the ring joint SDPA perf cases and reduce_to_root fail to build their
# kernels; print the build error with the PR's device sources, then with main's (main_dev.tar.gz: main's versions of every
# device-side file the PR changes), each with an empty kernel cache.
cd /work
RJ=tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py
RR=tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/nightly/test_reduce_to_root_trace.py
show() { grep -E "error:|Failed to generate|TT_THROW|TT_FATAL|required from|note: " "$1" | sort | uniq -c | sort -rn | head -25 | cut -c1-600; grep -E "passed|failed|error" "$1" | tail -1; }
for side in pr main; do
  [[ $side == main ]] && tar xzf tests/eb_r3_ci/main_dev.tar.gz -C /work && git -C /work status --short 2>/dev/null | head -20
  export TT_METAL_CACHE=/tmp/qb2c_$side; rm -rf $TT_METAL_CACHE; mkdir -p $TT_METAL_CACHE
  echo "##### $side ring joint perf_check wan2_2"; timeout 900 python3 -m pytest -q -rfE -p no:cacheprovider $RJ -k "test_ring_joint_attention_perf_check and wan2_2_1xGLX" > /tmp/qb2c_rj_$side.txt 2>&1; show /tmp/qb2c_rj_$side.txt
  echo "##### $side reduce_to_root with trace"; timeout 900 python3 -m pytest -q -rfE -p no:cacheprovider $RR -k "test_reduce_to_root_with_trace" > /tmp/qb2c_rr_$side.txt 2>&1; show /tmp/qb2c_rr_$side.txt
  echo "##### $side reduce_to_root auto intermediate"; timeout 900 python3 -m pytest -q -rfE -p no:cacheprovider $RR -k "test_reduce_to_root_auto_intermediate" > /tmp/qb2c_ra_$side.txt 2>&1; show /tmp/qb2c_ra_$side.txt
done
