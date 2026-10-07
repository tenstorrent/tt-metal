#!/usr/bin/env bash
# Round 3 eltwise binary (#58723 review, twin round): Blackhole QuietBox 2 with the JIT reading every header from /work
# (TT_METAL_RUNTIME_ROOT): ring joint SDPA's perf checks (the ring-4 QuietBox entries CI's Galaxy job runs at ring 8) and ring
# joint MLA at 2x2 (CI unset, which the test's uncollect_if keys on), device time A/B with the opt-in and bits; then
# reduce_to_root's traced replay timed by the test's own BenchmarkProfiler, main against the opt-in without the profiler.
cd /work
export TT_METAL_RUNTIME_ROOT=/work EB_SHOW_ERR=1
RJ=tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py
RM=models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_joint_mla.py
RR=tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/nightly/test_reduce_to_root_trace.py
echo "##### reduce_to_root trace bench"; EB_RUN_LIMIT=600 bash tests/eb_r3_ci/bench_ab.sh tests/eb_r3_ci/optin_r2r.txt 4 --count 3 $RR -k "test_reduce_to_root_with_trace"
echo "##### ring joint perf_check pass 1"; EB_REPS=3 EB_RUN_LIMIT=1500 bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_ring.txt $RJ -k "perf_check"
echo "##### elf ring joint perf_check"; python3 tests/eb_r3_ci/elf_cache_diff.py /tmp/ebset/cache_main /tmp/ebset/cache_optin 2>&1 | head -30
echo "##### bits ring joint perf_check"; EB_RUN_LIMIT=1500 bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_ring.txt -p eb_seed_plugin $RJ -k "perf_check"
echo "##### ring mla 2x2 pass 1"; env -u CI -u TT_GH_CI_INFRA EB_REPS=2 EB_RUN_LIMIT=1200 bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_ring.txt $RM -k "test_mla_sdpa and 2x2 and pcc_check and fabric2d and single_run"
echo "##### elf ring mla 2x2"; python3 tests/eb_r3_ci/elf_cache_diff.py /tmp/ebset/cache_main /tmp/ebset/cache_optin 2>&1 | head -30
echo "##### bits ring mla 2x2"; env -u CI -u TT_GH_CI_INFRA EB_RUN_LIMIT=1200 bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_ring.txt -p eb_seed_plugin $RM -k "test_mla_sdpa and 2x2 and pcc_check and fabric2d and single_run"
for p in 2 3; do
  echo "##### ring joint perf_check pass $p"; EB_REPS=3 EB_RUN_LIMIT=1500 bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_ring.txt $RJ -k "perf_check"
  echo "##### ring mla 2x2 pass $p"; env -u CI -u TT_GH_CI_INFRA EB_REPS=2 EB_RUN_LIMIT=1200 bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_ring.txt $RM -k "test_mla_sdpa and 2x2 and pcc_check and fabric2d and single_run"
done
