#!/usr/bin/env bash
# Scratch, not for merge. ttnn functional tests of every op that calls topk merge K32/K64 (or the topk_large_indices op), one pytest per file.
# usage: run_tk_ttnn.sh [file...]  (default list below; CI runs from /work, a local dry run passes one file)
set +e +o pipefail
ROOT=${ROOT:-/work}; cd $ROOT; export TT_METAL_HOME=$ROOT HWLOCK_HELD=${HWLOCK_HELD:-ci}
FILES=("$@")
[ ${#FILES[@]} -gt 0 ] || FILES=(
 tests/ttnn/unit_tests/operations/reduce/test_topk.py
 tests/ttnn/unit_tests/operations/reduce/test_moe.py
 tests/ttnn/unit_tests/operations/reduce/test_sampling.py
 tests/ttnn/unit_tests/operations/data_movement/test_sort.py
 tests/ttnn/nightly/unit_tests/operations/reduction/test_deepseek_grouped_gate.py
 tests/ttnn/nightly/unit_tests/operations/reduction/test_topk_stable_adversarial.py
 tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_router_gpt.py
 tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/test_moe_grouped_topk.py
 tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/test_moe_grouped_topk_stable.py
 tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/test_moe_grouped_topk_ties.py
 tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_large_indices.py)
OUT=$ROOT/generated/tkt; mkdir -p $OUT
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s' 2>/dev/null)"
i=0
for f in "${FILES[@]}"; do
  i=$((i+1)); export TT_METAL_CACHE=$OUT/cache_$i
  timeout -s INT -k 60 ${PT_TMO:-2400} python3 -m pytest -q -p no:cacheprovider -rfE "$f" > $OUT/$i.log 2>&1
  echo "== TT $i rc=$? $(basename $f) $(grep -a -E '[0-9]+ (passed|failed|skipped|error)' $OUT/$i.log | tail -n 1)"
  grep -a -o '^\(FAILED\|ERROR\) [^ ]*' $OUT/$i.log | head -n 15 | sed "s/^/TTFAIL $i /"
done
echo "== done $(date -u +%FT%TZ)"
