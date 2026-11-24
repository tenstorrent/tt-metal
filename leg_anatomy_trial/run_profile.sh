#!/bin/bash
# Leg-anatomy profile of one merge-gate leg's compile phase on a CPU runner.
# Env: ARCH GROUP NGROUPS OUT; cwd = tt_metal/tt-llk/tests/python_tests
set -uo pipefail
mkdir -p "$OUT"
MARK="perf and not accuracy and not nightly"
COMMON=(-q --override-ini=log_cli=false -m "$MARK" --timeout=600 --perf-splits "$NGROUPS" --perf-group "$GROUP" -p no:cacheprovider)
T() { # T label cmd...
  local label=$1; shift
  local s=$(date +%s.%N)
  "$@" > "$OUT/$label.log" 2>&1; local rc=$?
  local e=$(date +%s.%N)
  python3 -c "import json;print(json.dumps({'label':'$label','wall_s':round($e-$s,2),'rc':$rc}))" >> "$OUT/times.jsonl"
  tail -2 "$OUT/$label.log"
}
BUILD="$RUNNER_TEMP/tt-llk-build"
T pytest_version python3 -m pytest --version
T import_conftest python3 -X importtime -c "import conftest" 
cp "$OUT/import_conftest.log" "$OUT/importtime.txt"
T collect_only_n0 python3 -m pytest "${COMMON[@]}" --compile-producer --collect-only -p no:xdist
T collect_only_n0_prof python3 -m cProfile -o "$OUT/collect.prof" -m pytest "${COMMON[@]}" --compile-producer --collect-only -p no:xdist
T collect_only_consumer_n0 python3 -m pytest "${COMMON[@]}" --compile-consumer --collect-only -p no:xdist
rm -rf "$BUILD"
T cold_n10 python3 -m pytest "${COMMON[@]}" --compile-producer -n 10 --junitxml="$OUT/cold_n10.xml" .
python3 "$RUNNER_TEMP/trial/elf_dedup.py" "$BUILD" "$OUT/dedup.json"
T warm_n10 python3 -m pytest "${COMMON[@]}" --compile-producer -n 10 --junitxml="$OUT/warm_n10.xml" .
T warm_n0_prof python3 -m cProfile -o "$OUT/warm_n0.prof" -m pytest "${COMMON[@]}" --compile-producer -p no:xdist --junitxml="$OUT/warm_n0.xml" .
T warm_n1 python3 -m pytest "${COMMON[@]}" --compile-producer -n 1 .
for p in collect warm_n0; do
  python3 -c "import pstats,sys; s=pstats.Stats('$OUT/$p.prof'); s.sort_stats('cumulative').print_stats(60); s.sort_stats('tottime').print_stats(40)" > "$OUT/$p.prof.txt" 2>&1
done
du -sh "$BUILD" > "$OUT/build_size.txt"
