#!/usr/bin/env bash
# Scratch, not for merge. topk_large_indices perf checks on the host-IOMMU runner, three runs, exact ns printed.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work HWLOCK_HELD=ci
mkdir -p /work/generated/pt
for i in 1 2 3; do export TT_METAL_CACHE=/work/generated/pt/cache_$i
  timeout -s INT -k 60 1500 python3 -m pytest -q -p no:cacheprovider -rfP tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_large_indices.py > /work/generated/pt/$i.log 2>&1
  echo "PT $i rc=$? $(grep -a -E '[0-9]+ (passed|failed)' /work/generated/pt/$i.log | tail -n 1)"
  grep -a -o '^FAILED tests[^ ]*' /work/generated/pt/$i.log | sed "s/^/PTFAIL $i /"
  grep -a -o -E "perf check [a-z_]+: duration=[0-9.]+ ms" /work/generated/pt/$i.log | sed "s/^/PTDUR $i /" | sort -u
  grep -a -o -E "trace perf: scalar=[^\"]*" /work/generated/pt/$i.log | head -1 | sed "s/^/PTTRACE $i /"
done
