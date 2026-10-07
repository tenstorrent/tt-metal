#!/usr/bin/env bash
# Round 3 eltwise binary (#58723 review, twin round): Blackhole QuietBox 2, ring joint MLA SDPA at CI's 2x2 pcc_check selection,
# device time A/B with the opt-in, one test per process under tracy, JIT root /work, CI unset; then 3 more passes of the
# reduce_to_root trace timing without the profiler.
[[ -n "${HWLOCK_HELD:-}" || -n "${GITHUB_ACTIONS:-}" || -n "${TT_METAL_MOCK_CLUSTER_DESC_PATH:-}" ]] || { echo "not under hwlock" >&2; exit 2; }
cd /work
export TT_METAL_RUNTIME_ROOT=/work
env -u CI -u TT_GH_CI_INFRA EB_RUN_LIMIT=900 bash tests/eb_r3_ci/ab_plain.sh tests/eb_r3_ci/optin_ring.txt tests/eb_r3_ci/spec_mla_qb2.txt 3 "^RingJointSDPA"
python3 tests/eb_r3_ci/elf_set_diff.py /tmp/ebplain/cache_main /tmp/ebplain/cache_optin 2>&1 | grep -v "ELFSET identical"
echo "##### reduce_to_root trace bench"; EB_RUN_LIMIT=600 bash tests/eb_r3_ci/bench_ab.sh tests/eb_r3_ci/optin_r2r.txt 4 --count 3 tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/nightly/test_reduce_to_root_trace.py -k "test_reduce_to_root_with_trace"
