#!/usr/bin/env bash
# Scratch: the ttnn topk_large_indices test file on the checked-out tree (main), three runs, failing ids and durations.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work HWLOCK_HELD=ci
mkdir -p /work/generated/pt
for a in M1 M2 M3; do export TT_METAL_CACHE=/work/generated/pt/cache_$a$RANDOM
  timeout -s INT -k 60 1500 python3 -m pytest -q -p no:cacheprovider -rf tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_large_indices.py > /work/generated/pt/$a.log 2>&1
  echo "PT $a rc=$? $(grep -a -E '[0-9]+ (passed|failed)' /work/generated/pt/$a.log | tail -n 1)"
  grep -a -o '^FAILED tests[^ ]*' /work/generated/pt/$a.log | sed "s/^/PTFAIL $a /"
  grep -a -o -E "perf check [a-z_]+: duration=[0-9.]+ ms" /work/generated/pt/$a.log | sed "s/^/PTDUR $a /" | sort -u
done
