#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58723 fourth review), Blackhole QuietBox 2, JIT root /work. reduce_to_root (8x32 tiles, HiFi4): main's
# program against the PR commit's broadcast opt-in (this CI branch drops the define, optin_r2r.txt adds it back): bits, the test's own traced time
# (BenchmarkProfiler, 4 passes of main optin optin main, 3 repeats each) and device time under the profiler. sdpa_reduce_to_all
# (post_sdpa's SDPA reduce worker on a 4x1 mesh, HiFi4): main's program against SDPA_BCAST_COL_REUSE_PER_TILE_HANDOFF: bits,
# the traced benchmark and device time.
cd /work
export TT_METAL_RUNTIME_ROOT=/work EB_SHOW_ERR=1
RR=tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/nightly/test_reduce_to_root_trace.py
S2A=models/demos/deepseek_v3_b1/tests/unit_tests/test_sdpa_reduce_to_all.py
if [[ -z "${PROD_ONLY:-}" || "${PROD_ONLY}" == r2r ]]; then
  echo "##### bits reduce_to_root"; EB_RUN_LIMIT=1500 bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_r2r.txt -p eb_seed_plugin -o timeout=1800 $RR
  echo "##### elf reduce_to_root"; python3 tests/eb_r3_ci/elf_set_diff.py /tmp/ebbits/cache_main /tmp/ebbits/cache_optin "compute_kernel" 2>&1 | head -20
  echo "##### reduce_to_root trace bench"; EB_RUN_LIMIT=900 bash tests/eb_r3_ci/bench_ab.sh tests/eb_r3_ci/optin_r2r.txt 4 --count 3 $RR -k "test_reduce_to_root_with_trace"
  for t in test_reduce_to_root_auto_intermediate test_reduce_to_root_with_trace; do
    echo "##### reduce_to_root device time $t"; EB_RUN_LIMIT=2400 bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_r2r.txt -o timeout=1800 $RR -k "$t"
  done
fi
if [[ -z "${PROD_ONLY:-}" || "${PROD_ONLY}" == s2a ]]; then
  echo "##### bits sdpa_reduce_to_all"; EB_RUN_LIMIT=1500 bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_s2a.txt -p eb_seed_plugin -o timeout=1800 $S2A -k "test_sdpa_reduce_to_all and not trace"
  echo "##### elf sdpa_reduce_to_all"; python3 tests/eb_r3_ci/elf_set_diff.py /tmp/ebbits/cache_main /tmp/ebbits/cache_optin "sdpa_reduce_kernel" 2>&1 | head -20
  echo "##### sdpa_reduce_to_all trace bench"; EB_RUN_LIMIT=900 bash tests/eb_r3_ci/bench_ab.sh tests/eb_r3_ci/optin_s2a.txt 4 $S2A -k "test_sdpa_reduce_to_all_trace"
  printf '%s\n' "$S2A|test_sdpa_reduce_to_all and reduce_only and pos500" "$S2A|test_sdpa_reduce_to_all and reduce_only and pos3500" "$S2A|test_sdpa_reduce_to_all and reduce_and_scatter and pos2500" > /tmp/s2a_spec.txt
  echo "##### sdpa_reduce_to_all device time"; EB_RUN_LIMIT=900 bash tests/eb_r3_ci/ab_plain.sh tests/eb_r3_ci/optin_s2a.txt /tmp/s2a_spec.txt 3 "."
fi
