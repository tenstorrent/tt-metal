#!/usr/bin/env bash
# Scratch: the ttnn topk_large_indices test file on A (head) and B twice each, the full list of failing node ids per run.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work HWLOCK_HELD=ci; S=/work/ci_ab
RT=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(os.path.abspath(ttnn.__file__)))")
use() { while read -r rel; do cp "$S/variants/$1/$rel" "/work/$rel"; [[ -f $RT/$rel ]] && cp "$S/variants/$1/$rel" "$RT/$rel"; done < $S/files_ops.txt; }
mkdir -p /work/generated/pt
for a in A B A B B A; do use $a; export TT_METAL_CACHE=/work/generated/pt/cache_$a$RANDOM
  timeout -s INT -k 60 1500 python3 -m pytest -q -p no:cacheprovider -rfP tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_large_indices.py > /work/generated/pt/$a.log 2>&1
  echo "PT $a rc=$? $(grep -a -E '[0-9]+ (passed|failed)' /work/generated/pt/$a.log | tail -n 1)"
  grep -a -o '^FAILED tests[^ ]*' /work/generated/pt/$a.log | sed "s/^/PTFAIL $a /"
  grep -a -o -E "perf check [a-z_]+: duration=[0-9.]+ ms" /work/generated/pt/$a.log | sed "s/^/PTDUR $a /" | sort -u
done
